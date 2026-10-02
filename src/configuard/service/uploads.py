"""Stream a multipart upload straight into a private temp directory.

The body is never buffered whole: python-multipart parses the request stream
chunk by chunk and only the bytes of the single `file` part are written to
disk. The byte cap is enforced while streaming. The client's filename is used
ONLY for its extension (a cross-check against the content signature); it is
never stored or logged.
"""

from __future__ import annotations

import re
from pathlib import Path

from python_multipart.multipart import MultipartParser, parse_options_header

_EXT = re.compile(r"\.[a-z0-9]{1,5}$")
CHUNK_LIMIT_FIELDS = 4096


class UploadError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(code)
        self.status, self.code, self.message = status, code, message


def sniff_extension(header: bytes) -> str | None:
    """Canonical extension for a recognised signature (matches configuard.validation)."""
    if header.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return ".webp"
    if header[4:8] == b"ftyp":
        return ".mov" if header[8:10] == b"qt" else ".mp4"
    if header.startswith(b"\x1a\x45\xdf\xa3"):
        return ".mkv"
    if header[:4] == b"RIFF" and header[8:12] == b"AVI ":
        return ".avi"
    return None


class _Sink:
    def __init__(self, dest: Path, max_bytes: int) -> None:
        self.dest, self.max_bytes = dest, max_bytes
        self.fh = None
        self.size = 0
        self.files = 0
        self.client_ext: str | None = None
        self._hname = b""
        self._hval = b""
        self._headers: dict[bytes, bytes] = {}
        self._in_file = False
        self._field_bytes = 0

    def on_part_begin(self) -> None:
        self._headers, self._in_file = {}, False

    def on_header_field(self, data, start, end) -> None:
        self._hname += data[start:end]

    def on_header_value(self, data, start, end) -> None:
        self._hval += data[start:end]

    def on_header_end(self) -> None:
        self._headers[self._hname.strip().lower()] = self._hval.strip()
        self._hname, self._hval = b"", b""

    def on_headers_finished(self) -> None:
        _, opts = parse_options_header(self._headers.get(b"content-disposition", b""))
        name = opts.get(b"name")
        filename = opts.get(b"filename")
        if name == b"file" and filename is not None:
            self.files += 1
            if self.files > 1:
                raise UploadError(400, "too_many_files", "Send exactly one file in the 'file' field.")
            m = _EXT.search(filename.decode("utf-8", "replace").lower())
            self.client_ext = m.group(0) if m else None
            self.fh = self.dest.open("wb")
            self._in_file = True

    def on_part_data(self, data, start, end) -> None:
        n = end - start
        if self._in_file:
            self.size += n
            if self.size > self.max_bytes:
                raise UploadError(413, "file_too_large", "File exceeds the configured size limit.")
            self.fh.write(data[start:end])
        else:
            self._field_bytes += n
            if self._field_bytes > CHUNK_LIMIT_FIELDS:
                raise UploadError(400, "unexpected_fields", "Only a 'file' field is accepted.")

    def on_part_end(self) -> None:
        if self._in_file and self.fh is not None:
            self.fh.close()
            self.fh = None
        self._in_file = False

    def close(self) -> None:
        if self.fh is not None:
            self.fh.close()
            self.fh = None


async def receive_upload(request, workdir: Path, max_bytes: int) -> tuple[Path, int]:
    """Returns (path with extension, size). Raises UploadError."""
    ctype, opts = parse_options_header(request.headers.get("content-type", ""))
    if ctype != b"multipart/form-data" or b"boundary" not in opts:
        raise UploadError(415, "unsupported_content_type", "Use multipart/form-data with a 'file' field.")
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_bytes + 64 * 1024:
        raise UploadError(413, "file_too_large", "File exceeds the configured size limit.")
    part = workdir / "upload.part"
    sink = _Sink(part, max_bytes)
    callbacks = {k: getattr(sink, k) for k in ("on_part_begin", "on_header_field", "on_header_value", "on_header_end",
                                                "on_headers_finished", "on_part_data", "on_part_end")}
    parser = MultipartParser(opts[b"boundary"], callbacks)
    total = 0
    try:
        async for chunk in request.stream():
            total += len(chunk)
            if total > max_bytes + 64 * 1024:
                raise UploadError(413, "file_too_large", "File exceeds the configured size limit.")
            parser.write(chunk)
        parser.finalize()
    except UploadError:
        raise
    except Exception as exc:  # malformed multipart bodies
        raise UploadError(400, "malformed_upload", "The multipart body could not be parsed.") from exc
    finally:
        sink.close()
    if sink.files != 1 or not part.exists():
        raise UploadError(400, "missing_file", "Send exactly one file in the 'file' field.")
    if sink.size == 0:
        raise UploadError(400, "empty_file", "File is empty.")
    with part.open("rb") as f:
        ext = sink.client_ext or sniff_extension(f.read(64)) or ".bin"
    final = workdir / f"upload{ext}"
    part.replace(final)
    return final, sink.size
