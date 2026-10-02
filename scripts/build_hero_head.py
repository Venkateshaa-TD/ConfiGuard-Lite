"""Build the landing-page hero head (GLB) from MakeHuman CC0 assets.

The head is a fictional, generic identity: the MakeHuman hm08 base mesh with an equal
African/Asian/Caucasian young-male macro blend, fitted CC0 eyes, eyebrows and eyelashes,
and a CC0 skin texture. No MakeHuman application code (AGPL) is used; this script reads the
CC0 data files directly and reimplements the documented proxy fitting rule
(weighted three-vertex barycentre plus a scaled offset).

Output is a plain GLB (no Draco/meshopt: the site CSP has no 'wasm-unsafe-eval'), JPEG
textures at most 1024 px, and a JSON provenance record with SHA-256 for every input.

Usage:
    python scripts/build_hero_head.py --src D:/path/to/makehuman_cache \
        --out frontend/public/hero/head.glb
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import struct
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

LOG = logging.getLogger("build_hero_head")

PACK = "makehuman_system_assets_cc0.zip"
RACES = ("african-male-young", "asian-male-young", "caucasian-male-young")
SKIN = "skins/young_caucasian_male/young_lightskinned_male_diffuse.png"
EYES = ("eyes/high-poly/high-poly", "eyes/materials/brown_eye.png")
BROWS = ("eyebrows/eyebrow001/eyebrow001", "eyebrows/eyebrow001/eyebrow001.png")
LASHES = ("eyelashes/eyelashes01/eyelashes01", "eyelashes/eyelashes01/eyelashes01.png")
HAIR = ("hair/short02/short02", "hair/short02/short02_diffuse.png")
UNIT = 0.1  # MakeHuman units are decimetres; glTF is metres.
MAX_TEX = 1024


@dataclass
class Obj:
    v: np.ndarray
    vt: np.ndarray
    faces: list[tuple[list[int], list[int]]]  # (vertex indices, uv indices), 0-based
    groups: dict[str, list[int]] = field(default_factory=dict)  # group -> face ids
    group_verts: dict[str, set[int]] = field(default_factory=dict)


def parse_obj(text: str) -> Obj:
    v: list[list[float]] = []
    vt: list[list[float]] = []
    faces: list[tuple[list[int], list[int]]] = []
    groups: dict[str, list[int]] = {}
    gverts: dict[str, set[int]] = {}
    group = "default"
    for line in text.splitlines():
        if line.startswith("v "):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("vt "):
            vt.append([float(x) for x in line.split()[1:3]])
        elif line.startswith("g "):
            group = line.split(maxsplit=1)[1].strip()
        elif line.startswith("f "):
            vi, ti = [], []
            for tok in line.split()[1:]:
                parts = tok.split("/")
                vi.append(int(parts[0]) - 1)
                ti.append(int(parts[1]) - 1 if len(parts) > 1 and parts[1] else int(parts[0]) - 1)
            groups.setdefault(group, []).append(len(faces))
            gverts.setdefault(group, set()).update(vi)
            faces.append((vi, ti))
    return Obj(np.asarray(v, np.float64), np.asarray(vt, np.float64), faces, groups, gverts)


def apply_target(coords: np.ndarray, text: str, weight: float) -> None:
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 4 and not line.startswith("#"):
            coords[int(parts[0])] += weight * np.array([float(p) for p in parts[1:]])


def fit_proxy(base: np.ndarray, mhclo: str) -> np.ndarray:
    """MakeHuman .mhclo rule: sum(w_i * base[r_i]) + diag(scale) @ offset."""
    scales = np.ones(3)
    rows: list[list[float]] = []
    in_verts = False
    for line in mhclo.splitlines():
        parts = line.split()
        if not parts:
            if in_verts and rows:
                in_verts = False
            continue
        key = parts[0]
        if key in ("x_scale", "y_scale", "z_scale"):
            axis = "xyz".index(key[0])
            a, b, den = int(parts[1]), int(parts[2]), float(parts[3])
            scales[axis] = abs(base[a, axis] - base[b, axis]) / den
        elif key == "verts":
            in_verts = True
        elif in_verts:
            if len(parts) == 9:
                rows.append([float(p) for p in parts])
            elif len(parts) == 1 and parts[0].lstrip("-").isdigit():
                rows.append([float(parts[0]), 0, 0, 1, 0, 0, 0, 0, 0])
            else:
                in_verts = False
    arr = np.asarray(rows)
    ref = arr[:, :3].astype(int)
    w = arr[:, 3:6]
    off = arr[:, 6:9]
    return (base[ref[:, 0]] * w[:, :1] + base[ref[:, 1]] * w[:, 1:2] + base[ref[:, 2]] * w[:, 2:3]
            + off * scales)


def joint(obj: Obj, coords: np.ndarray, name: str) -> np.ndarray:
    return coords[sorted(obj.group_verts[name])].mean(axis=0)


@dataclass
class Prim:
    name: str
    pos: np.ndarray
    nrm: np.ndarray
    uv: np.ndarray
    idx: np.ndarray
    image: bytes
    mime: str
    alpha: str  # OPAQUE | MASK | BLEND
    double_sided: bool = False
    roughness: float = 0.6


def build_prim(name: str, coords: np.ndarray, uvs: np.ndarray, faces, image: bytes, mime: str,
               alpha: str, double_sided: bool = False, roughness: float = 0.6) -> Prim:
    """Weld (vertex, uv) pairs, triangulate fans, compute smooth normals by position."""
    key_to_new: dict[tuple[int, int], int] = {}
    pos_src: list[int] = []
    uv_src: list[int] = []
    tris: list[tuple[int, int, int]] = []
    for vi, ti in faces:
        ids = []
        for a, b in zip(vi, ti):
            k = (a, b)
            if k not in key_to_new:
                key_to_new[k] = len(pos_src)
                pos_src.append(a)
                uv_src.append(b)
            ids.append(key_to_new[k])
        for i in range(1, len(ids) - 1):
            tris.append((ids[0], ids[i], ids[i + 1]))
    pos = coords[pos_src]
    tri = np.asarray(tris, np.int64)
    # Smooth normals accumulated per ORIGINAL vertex so UV seams stay invisible.
    src = np.asarray(pos_src)
    fn = np.cross(pos[tri[:, 1]] - pos[tri[:, 0]], pos[tri[:, 2]] - pos[tri[:, 0]])
    acc = np.zeros((coords.shape[0], 3))
    for c in range(3):
        np.add.at(acc, src[tri[:, c]], fn)
    nrm = acc[src]
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    uv = uvs[uv_src].copy()
    uv[:, 1] = 1.0 - uv[:, 1]  # OBJ v-up -> glTF v-down
    return Prim(name, pos.astype(np.float32), nrm.astype(np.float32), uv.astype(np.float32),
                tri.astype(np.uint32), image, mime, alpha, double_sided, roughness)


def encode_image(img: Image.Image, keep_alpha: bool, quality: int = 86) -> tuple[bytes, str]:
    if max(img.size) > MAX_TEX:
        s = MAX_TEX / max(img.size)
        img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS)
    buf = io.BytesIO()
    if keep_alpha:
        img.convert("RGBA").save(buf, "PNG", optimize=True)
        return buf.getvalue(), "image/png"
    img.convert("RGB").save(buf, "JPEG", quality=quality, optimize=True, progressive=False)
    return buf.getvalue(), "image/jpeg"


def uv_islands(faces) -> list[set[int]]:
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, t in faces:
        for a in t[1:]:
            parent[find(a)] = find(t[0])
    out: dict[int, set[int]] = {}
    for _, t in faces:
        out.setdefault(find(t[0]), set()).update(t)
    return list(out.values())


def pack_atlas(img: Image.Image, uvs: np.ndarray, faces, size: int = MAX_TEX, pad_px: int = 6) -> tuple[Image.Image, np.ndarray]:
    """Re-pack only the UV islands the head uses into a square atlas (shelf packing, one shared scale)."""
    W, H = img.size
    rects = []
    for isl in uv_islands(faces):
        ids = np.fromiter(isl, int)
        sub = uvs[ids]
        x0, x1 = int(sub[:, 0].min() * W) - pad_px, int(np.ceil(sub[:, 0].max() * W)) + pad_px
        y0, y1 = int((1 - sub[:, 1].max()) * H) - pad_px, int(np.ceil((1 - sub[:, 1].min()) * H)) + pad_px
        rects.append((max(x0, 0), max(y0, 0), min(x1, W), min(y1, H), ids))
    rects.sort(key=lambda r: -(r[3] - r[1]))
    scale = size / max(r[3] - r[1] for r in rects)
    while True:  # shrink until every island fits in columns of the atlas
        placed, cx, cy, colw, ok = [], 0, 0, 0, True
        for r in rects:
            w, h = round((r[2] - r[0]) * scale), round((r[3] - r[1]) * scale)
            if cy + h > size:
                cx, cy, colw = cx + colw, 0, 0
            if cx + w > size:
                ok = False
                break
            placed.append((cx, cy, w, h, r))
            cy += h
            colw = max(colw, w)
        if ok:
            break
        scale *= 0.97
    atlas = Image.new("RGB", (size, size), img.resize((1, 1)).getpixel((0, 0)))
    out = uvs.copy()
    for cx, cy, w, h, (x0, y0, x1, y1, ids) in placed:
        atlas.paste(img.crop((x0, y0, x1, y1)).resize((w, h), Image.LANCZOS), (cx, cy))
        px = uvs[ids, 0] * W
        py = (1 - uvs[ids, 1]) * H
        out[ids, 0] = (cx + (px - x0) / (x1 - x0) * w) / size
        out[ids, 1] = 1 - (cy + (py - y0) / (y1 - y0) * h) / size
    LOG.info("atlas: %d islands, scale %.3f", len(placed), scale)
    return atlas, out


def write_glb(path: Path, prims: list[Prim], extras: dict) -> int:
    bin_parts: list[bytes] = []
    views: list[dict] = []
    accessors: list[dict] = []
    offset = 0

    def add_view(data: bytes, target: int | None) -> int:
        nonlocal offset
        pad = (-offset) % 4
        if pad:
            bin_parts.append(b"\0" * pad)
            offset += pad
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        views.append(view)
        bin_parts.append(data)
        offset += len(data)
        return len(views) - 1

    def add_acc(arr: np.ndarray, kind: str, comp: int, target: int, minmax: bool = False) -> int:
        view = add_view(arr.tobytes(), target)
        acc = {"bufferView": view, "componentType": comp, "count": int(arr.shape[0]), "type": kind}
        if minmax:
            acc["min"] = arr.min(axis=0).tolist()
            acc["max"] = arr.max(axis=0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    meshes, nodes, materials, textures, images = [], [], [], [], []
    for p in prims:
        a_pos = add_acc(p.pos, "VEC3", 5126, 34962, minmax=True)
        a_nrm = add_acc(p.nrm, "VEC3", 5126, 34962)
        a_uv = add_acc(p.uv, "VEC2", 5126, 34962)
        idx = p.idx.reshape(-1)
        if idx.max() < 65535:
            a_idx = add_acc(idx.astype(np.uint16), "SCALAR", 5123, 34963)
        else:
            a_idx = add_acc(idx.astype(np.uint32), "SCALAR", 5125, 34963)
        images.append({"bufferView": add_view(p.image, None), "mimeType": p.mime, "name": p.name})
        textures.append({"source": len(images) - 1, "sampler": 0})
        mat = {"name": p.name, "pbrMetallicRoughness": {
            "baseColorTexture": {"index": len(textures) - 1}, "metallicFactor": 0.0,
            "roughnessFactor": p.roughness}, "alphaMode": p.alpha, "doubleSided": p.double_sided}
        if p.alpha == "MASK":
            mat["alphaCutoff"] = 0.35
        materials.append(mat)
        meshes.append({"name": p.name, "primitives": [{
            "attributes": {"POSITION": a_pos, "NORMAL": a_nrm, "TEXCOORD_0": a_uv},
            "indices": a_idx, "material": len(materials) - 1}]})
        nodes.append({"name": p.name, "mesh": len(meshes) - 1})

    blob = b"".join(bin_parts)
    blob += b"\0" * ((-len(blob)) % 4)
    gltf = {
        "asset": {"version": "2.0", "generator": "ConfiGuard-Lite build_hero_head.py",
                  "copyright": "CC0 1.0 - derived from MakeHuman CC0 assets", "extras": extras},
        "scene": 0, "scenes": [{"name": "hero-head", "nodes": list(range(len(nodes)))}],
        "nodes": nodes, "meshes": meshes, "materials": materials, "textures": textures,
        "images": images, "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(blob)}],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    total = 12 + 8 + len(js) + 8 + len(blob)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(struct.pack("<4sII", b"glTF", 2, total))
        fh.write(struct.pack("<I4s", len(js), b"JSON") + js)
        fh.write(struct.pack("<I4s", len(blob), b"BIN\0") + blob)
    return total


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, required=True, help="folder holding the MakeHuman CC0 inputs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--neck-drop", type=float, default=0.35, help="keep body faces above neck joint minus this (dm)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    inputs: dict[str, str] = {}
    base_bytes = (args.src / "base.obj").read_bytes()
    inputs["base.obj"] = sha256(base_bytes)
    base = parse_obj(base_bytes.decode("latin1"))
    coords = base.v.copy()
    for race in RACES:
        t = (args.src / f"{race}.target").read_bytes()
        inputs[f"{race}.target"] = sha256(t)
        apply_target(coords, t.decode("latin1"), 1.0 / len(RACES))

    zf = zipfile.ZipFile(args.src / PACK)
    inputs[PACK] = sha256((args.src / PACK).read_bytes())

    def read(name: str) -> bytes:
        return zf.read(name)

    neck = joint(base, coords, "joint-neck")
    head_top = coords[sorted(base.group_verts["body"])][:, 1].max()
    cut = neck[1] - args.neck_drop
    body = [base.faces[i] for i in base.groups["body"]]
    head_faces = [f for f in body if coords[f[0], 1].mean() > cut]
    LOG.info("neck y=%.3f top=%.3f cut=%.3f head faces=%d", neck[1], head_top, cut, len(head_faces))

    # Centre on the head (between the eye joints), convert to metres.
    centre = (joint(base, coords, "joint-l-eye") + joint(base, coords, "joint-r-eye")) / 2
    centre[2] = coords[np.unique(np.concatenate([f[0] for f in head_faces]))][:, 2].mean()

    def to_m(c: np.ndarray) -> np.ndarray:
        return (c - centre) * UNIT

    skin_img = Image.open(io.BytesIO(read(SKIN))).convert("RGB")
    skin_crop, skin_uv = pack_atlas(skin_img, base.vt, head_faces)
    skin_jpg, skin_mime = encode_image(skin_crop, keep_alpha=False)
    prims = [build_prim("skin", to_m(coords), skin_uv, head_faces, skin_jpg, skin_mime, "OPAQUE", roughness=0.55)]

    for name, (stem, tex), alpha, ds, rough in (
        ("eyes", EYES, "OPAQUE", False, 0.25),
        ("brows", BROWS, "BLEND", True, 0.8),
        ("lashes", LASHES, "BLEND", True, 0.8),
        ("hair", HAIR, "MASK", True, 0.7),
    ):
        mhclo = read(stem + ".mhclo").decode("latin1")
        obj = parse_obj(read(stem + ".obj").decode("latin1"))
        fitted = fit_proxy(coords, mhclo)
        if fitted.shape[0] != obj.v.shape[0]:
            raise SystemExit(f"{stem}: {fitted.shape[0]} fitted vs {obj.v.shape[0]} obj vertices")
        faces = obj.faces
        if name == "eyes":
            # The outer cornea shell maps to the transparent dot in the corner of the eye texture.
            faces = [f for f in faces if not (obj.vt[f[1], 0].min() > 0.85 and obj.vt[f[1], 1].max() < 0.15)]
        img = Image.open(io.BytesIO(read(tex)))
        data, mime = encode_image(img, keep_alpha=alpha != "OPAQUE")
        prims.append(build_prim(name, to_m(fitted), obj.vt, faces, data, mime, alpha, ds, rough))

    extras = {
        "identity": "fictional generic young adult; equal African/Asian/Caucasian MakeHuman macro blend",
        "licence": "CC0 1.0 (MakeHuman core assets)",
        "inputs_sha256": inputs,
        "components": {"skin": SKIN, "eyes": list(EYES), "brows": list(BROWS), "lashes": list(LASHES), "hair": list(HAIR),
                       "targets": [f"macrodetails/{r}.target" for r in RACES]},
    }
    size = write_glb(args.out, prims, extras)
    digest = sha256(args.out.read_bytes())
    tris = int(sum(p.idx.shape[0] for p in prims))
    LOG.info("wrote %s: %d bytes, %d triangles, sha256 %s", args.out, size, tris, digest)
    record = {"file": args.out.name, "bytes": size, "sha256": digest, "triangles": tris, **extras}
    args.out.with_suffix(".provenance.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
