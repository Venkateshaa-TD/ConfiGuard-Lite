// Minimal GLB reader for the self-hosted hero head (scripts/build_hero_head.py writes it).
// Supports exactly what that file uses: one triangle primitive per mesh, float32 attributes,
// uint16/uint32 indices and embedded PNG/JPEG images. Images are decoded with createImageBitmap
// straight from the bytes - no blob: URLs, workers, WASM or eval, so it runs under the site CSP.
import { BufferAttribute, BufferGeometry, SRGBColorSpace, Texture } from "three";

export interface GlbPart {
  name: string;
  geometry: BufferGeometry;
  map: Texture;
  alphaMode: "OPAQUE" | "MASK" | "BLEND";
  alphaCutoff: number;
  doubleSided: boolean;
  roughness: number;
}

interface Accessor { bufferView: number; componentType: number; count: number; type: "SCALAR" | "VEC2" | "VEC3" }
interface View { byteOffset?: number; byteLength: number }
interface Gltf {
  accessors: Accessor[];
  bufferViews: View[];
  images: { bufferView: number; mimeType: string }[];
  textures: { source: number }[];
  materials: { name: string; alphaMode?: GlbPart["alphaMode"]; alphaCutoff?: number; doubleSided?: boolean;
    pbrMetallicRoughness: { baseColorTexture: { index: number }; roughnessFactor?: number } }[];
  meshes: { name: string; primitives: { attributes: Record<string, number>; indices: number; material: number }[] }[];
}

const WIDTH = { SCALAR: 1, VEC2: 2, VEC3: 3 } as const;

export async function loadGlb(buffer: ArrayBuffer, anisotropy = 1): Promise<GlbPart[]> {
  const dv = new DataView(buffer);
  if (dv.getUint32(0, true) !== 0x46546c67 || dv.getUint32(4, true) !== 2) throw new Error("not a glTF 2.0 binary");
  const jsonLen = dv.getUint32(12, true);
  const json = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 20, jsonLen))) as Gltf;
  const binStart = 20 + jsonLen + 8;
  const binLen = dv.getUint32(20 + jsonLen, true);
  if (binStart + binLen > buffer.byteLength) throw new Error("truncated GLB");

  const view = (i: number) => {
    const v = json.bufferViews[i];
    if (!v) throw new Error("bad bufferView");
    return { offset: binStart + (v.byteOffset ?? 0), length: v.byteLength };
  };
  const accessor = (i: number) => {
    const a = json.accessors[i];
    if (!a) throw new Error("bad accessor");
    const { offset } = view(a.bufferView);
    const n = a.count * WIDTH[a.type];
    switch (a.componentType) {
      case 5126: return { array: new Float32Array(buffer.slice(offset, offset + n * 4)), size: WIDTH[a.type] };
      case 5123: return { array: new Uint16Array(buffer.slice(offset, offset + n * 2)), size: 1 };
      case 5125: return { array: new Uint32Array(buffer.slice(offset, offset + n * 4)), size: 1 };
      default: throw new Error("unsupported component type");
    }
  };

  const bitmaps = await Promise.all(json.images.map(async (img) => {
    const { offset, length } = view(img.bufferView);
    if (img.mimeType !== "image/jpeg" && img.mimeType !== "image/png") throw new Error("unsupported image type");
    return createImageBitmap(new Blob([new Uint8Array(buffer, offset, length)], { type: img.mimeType }));
  }));

  return json.meshes.map((mesh) => {
    const prim = mesh.primitives[0];
    if (!prim) throw new Error("empty mesh");
    const geometry = new BufferGeometry();
    for (const [key, name] of [["POSITION", "position"], ["NORMAL", "normal"], ["TEXCOORD_0", "uv"]] as const) {
      const idx = prim.attributes[key];
      if (idx === undefined) throw new Error(`missing ${key}`);
      const { array, size } = accessor(idx);
      geometry.setAttribute(name, new BufferAttribute(array, size));
    }
    geometry.setIndex(new BufferAttribute(accessor(prim.indices).array, 1));
    geometry.computeBoundingSphere();
    const mat = json.materials[prim.material];
    const tex = mat ? json.textures[mat.pbrMetallicRoughness.baseColorTexture.index] : undefined;
    const bitmap = tex ? bitmaps[tex.source] : undefined;
    if (!mat || !bitmap) throw new Error("missing material");
    const map = new Texture(bitmap);
    map.flipY = false;               // glTF UV convention
    map.colorSpace = SRGBColorSpace;
    map.anisotropy = anisotropy;
    map.needsUpdate = true;
    return {
      name: mesh.name, geometry, map,
      alphaMode: mat.alphaMode ?? "OPAQUE", alphaCutoff: mat.alphaCutoff ?? 0.5,
      doubleSided: Boolean(mat.doubleSided), roughness: mat.pbrMetallicRoughness.roughnessFactor ?? 0.6,
    };
  });
}
