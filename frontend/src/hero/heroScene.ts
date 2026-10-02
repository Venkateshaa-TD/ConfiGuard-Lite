// Hero scene: a fictional, generic human head built from MakeHuman CC0 assets (public/hero/head.glb,
// provenance in public/hero/head.provenance.json). Natural side: the textured head. Synthetic side:
// a dark surface with a cyan wireframe, landmark contours and a sparse point cloud. A vertical scan
// boundary (world-space clipping planes) decides which layer is visible.
import {
  ACESFilmicToneMapping, Box3, BufferAttribute, BufferGeometry, CanvasTexture, DirectionalLight, DoubleSide, Group, HemisphereLight,
  LineBasicMaterial, LineSegments, Mesh, MeshBasicMaterial, MeshStandardMaterial, PerspectiveCamera, Plane,
  PlaneGeometry, Points, PointsMaterial, Scene, Vector3, WebGLRenderer, WireframeGeometry, type Material, type Object3D,
} from "three";
import { loadGlb, type GlbPart } from "./glb";

export type HeroTheme = "light" | "dark";
export interface HeroOptions {
  mobile: boolean; interactive: boolean; theme: HeroTheme;
  context: WebGLRenderingContext | WebGL2RenderingContext;
}
export interface HeroStats {
  geometryBytes: number; textureBytes: number; assetBytes: number; triangles: number; dpr: number; mobile: boolean;
}
export interface HeroScene {
  render: (now: number, dt: number) => void;
  setPointer: (nx: number, ny: number) => void;
  setTheme: (theme: HeroTheme) => void;
  resize: (w: number, h: number) => void;
  dispose: () => void;
  stats: HeroStats;
}

export const HEAD_URL = "/hero/head.glb";
const SCALE = 7.5;            // the asset is in metres; the scene was laid out for a ~2-unit head
const NECK_CUT = -0.14;       // asset-space y below which nothing is drawn (clean horizontal edge)

const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));

/** Theme palettes for the 3D materials (mirrors the CSS tokens; switched without reloading the model). */
const PALETTE = {
  light: { skin: 0xf6f2ef, ink: 0x0d1015, inkMetal: 0.15, wire: 0x00b8c6, wireOpacity: 0.42, dots: 0x00c8d6,
    marks: 0x2fd6e2, sheet: "0,183,197", hemiSky: 0xffffff, hemiGround: 0x8d9aa3, hemi: 1.5, key: 2.4, rim: 0.6 },
  dark: { skin: 0xe9e3de, ink: 0x0a1118, inkMetal: 0.25, wire: 0x3fe0eb, wireOpacity: 0.34, dots: 0x5ff0f8,
    marks: 0x8ff6fb, sheet: "63,224,235", hemiSky: 0xc9d6de, hemiGround: 0x1d2834, hemi: 1.1, key: 2.2, rim: 0.8 },
} as const;

/** Facial landmark template in asset space (metres; origin between the eyes; +z towards the viewer).
 *  Each contour is snapped onto the front surface of the head at load time. */
const CONTOURS: [number, number][][] = [
  [[-0.058, 0.009], [-0.045, 0.015], [-0.03, 0.017], [-0.013, 0.013]],                         // right brow
  [[0.013, 0.013], [0.03, 0.017], [0.045, 0.015], [0.058, 0.009]],                              // left brow
  [[-0.045, 0.0], [-0.038, 0.007], [-0.03, 0.009], [-0.022, 0.006], [-0.016, 0.0], [-0.024, -0.006], [-0.031, -0.008], [-0.039, -0.005], [-0.045, 0.0]],
  [[0.016, 0.0], [0.022, 0.006], [0.03, 0.009], [0.038, 0.007], [0.045, 0.0], [0.039, -0.005], [0.031, -0.008], [0.024, -0.006], [0.016, 0.0]],
  [[0, 0.004], [0, -0.012], [0, -0.028], [0, -0.041]],                                          // nose ridge
  [[-0.018, -0.044], [-0.009, -0.051], [0, -0.053], [0.009, -0.051], [0.018, -0.044]],          // nose base
  [[-0.026, -0.076], [-0.013, -0.069], [0, -0.068], [0.013, -0.069], [0.026, -0.076], [0.012, -0.084], [0, -0.086], [-0.012, -0.084], [-0.026, -0.076]],
  [[-0.074, -0.004], [-0.072, -0.04], [-0.062, -0.078], [-0.042, -0.104], [-0.02, -0.117], [0, -0.12], [0.02, -0.117], [0.042, -0.104], [0.062, -0.078], [0.072, -0.04], [0.074, -0.004]],
];

function landmarkGeometry(skin: BufferGeometry): { lines: BufferGeometry; points: BufferGeometry } {
  const pos = skin.getAttribute("position");
  const snap = (x: number, y: number): [number, number, number] => {
    let bestZ = -Infinity;
    for (let r = 0.004; r < 0.02 && bestZ === -Infinity; r *= 1.6) {
      for (let i = 0; i < pos.count; i++) {
        const dx = pos.getX(i) - x, dy = pos.getY(i) - y;
        if (dx * dx + dy * dy < r * r && pos.getZ(i) > bestZ) bestZ = pos.getZ(i);
      }
    }
    return [x, y, (bestZ === -Infinity ? 0.04 : bestZ) + 0.0015];
  };
  const seg: number[] = [];
  const pts: number[] = [];
  for (const contour of CONTOURS) {
    const s = contour.map(([x, y]) => snap(x, y));
    s.forEach((p, i) => {
      pts.push(...p);
      const n = s[i + 1];
      if (n) seg.push(...p, ...n);
    });
  }
  return {
    lines: new BufferGeometry().setAttribute("position", new BufferAttribute(new Float32Array(seg), 3)),
    points: new BufferGeometry().setAttribute("position", new BufferAttribute(new Float32Array(pts), 3)),
  };
}

/** Split the two-eye mesh into left/right geometries (sharing attributes) so each can rotate on its own centre. */
function splitEyes(geo: BufferGeometry): { geo: BufferGeometry; centre: Vector3 }[] {
  const index = geo.getIndex()!;
  const pos = geo.getAttribute("position");
  return [1, -1].map((side) => {
    const tris: number[] = [];
    for (let i = 0; i < index.count; i += 3) {
      if (Math.sign(pos.getX(index.getX(i))) === side) tris.push(index.getX(i), index.getX(i + 1), index.getX(i + 2));
    }
    const g = new BufferGeometry();
    for (const name of ["position", "normal", "uv"]) g.setAttribute(name, geo.getAttribute(name));
    g.setIndex(tris);
    // Bounds of THIS eye's vertices (computeBoundingBox would span the whole shared attribute, i.e. both eyes).
    const box = new Box3();
    const v = new Vector3();
    for (const i of tris) box.expandByPoint(v.fromBufferAttribute(pos, i));
    // The mesh is the visible front of the eyeball, so the box centre is NOT the rotation centre:
    // put the pivot one eyeball radius behind the front-most point (radius = half the box width).
    const centre = box.getCenter(new Vector3());
    centre.z = box.max.z - (box.max.x - box.min.x) / 2;
    return { geo: g, centre };
  });
}

/** Soft vertical light sheet for the scan boundary (generated locally, no image files). */
function sheetTexture(rgb: string): CanvasTexture {
  const c = document.createElement("canvas");
  c.width = 64; c.height = 256;
  const ctx = c.getContext("2d")!;
  const gx = ctx.createLinearGradient(0, 0, 64, 0);
  gx.addColorStop(0, `rgba(${rgb},0)`); gx.addColorStop(0.5, `rgba(${rgb},0.85)`); gx.addColorStop(1, `rgba(${rgb},0)`);
  ctx.fillStyle = gx; ctx.fillRect(0, 0, 64, 256);
  ctx.globalCompositeOperation = "destination-in";
  const gy = ctx.createLinearGradient(0, 0, 0, 256);
  gy.addColorStop(0, "rgba(0,0,0,0)"); gy.addColorStop(0.2, "rgba(0,0,0,1)"); gy.addColorStop(0.8, "rgba(0,0,0,1)"); gy.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = gy; ctx.fillRect(0, 0, 64, 256);
  return new CanvasTexture(c);
}

const yieldTask = () => new Promise<void>((r) => setTimeout(r, 0));

const bytesOf = (g: BufferGeometry) => {
  let b = g.index ? g.index.array.byteLength : 0;
  for (const name of Object.keys(g.attributes)) b += (g.getAttribute(name).array as ArrayLike<number> & { byteLength: number }).byteLength;
  return b;
};

/** Built in small steps (yielding between them) so it never blocks the main thread for long. */
export async function createHeroScene(canvas: HTMLCanvasElement, opts: HeroOptions): Promise<HeroScene> {
  const renderer = new WebGLRenderer({ canvas, context: opts.context as WebGL2RenderingContext });
  const dpr = Math.min(window.devicePixelRatio || 1, opts.mobile ? 1.25 : 1.5);
  renderer.setPixelRatio(dpr);
  renderer.localClippingEnabled = true;
  const gl = renderer.getContext();
  const info = gl.getExtension("WEBGL_debug_renderer_info");
  const gpu = String(info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER));
  const fail = (msg: string): never => { renderer.dispose(); renderer.forceContextLoss(); throw new Error(msg); };
  if (/swiftshader|llvmpipe|software|basic render/i.test(gpu)) fail("software-renderer");   // too slow for a continuous 3D hero
  renderer.setClearColor(0x000000, 0);
  renderer.toneMapping = ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;

  let parts: GlbPart[] = [];
  let assetBytes = 0;
  try {
    const res = await fetch(HEAD_URL, { credentials: "same-origin" });
    if (!res.ok) throw new Error(String(res.status));
    const buf = await res.arrayBuffer();
    assetBytes = buf.byteLength;
    parts = await loadGlb(buf, opts.mobile ? 1 : Math.min(4, renderer.capabilities.getMaxAnisotropy()));
  } catch {
    fail("asset-unavailable");
  }
  const part = (name: string) => parts.find((p) => p.name === name) ?? fail("asset-unavailable");
  await yieldTask();

  const scene = new Scene();
  const camera = new PerspectiveCamera(27, 1, 0.1, 50);
  camera.position.set(0, 0.02, 6.6);

  const hemi = new HemisphereLight(0xffffff, 0x8d9aa3, 1);
  const key = new DirectionalLight(0xffffff, 2.2); key.position.set(-3.2, 2.6, 4);
  const rim = new DirectionalLight(0x00d7e5, 1.2); rim.position.set(3.5, 1.2, -2.5);
  const fill = new DirectionalLight(0xffffff, 0.35); fill.position.set(3, -0.5, 3);
  scene.add(hemi, key, rim, fill);

  const keepNatural = new Plane(new Vector3(-1, 0, 0), 0);   // keeps x <= boundary
  const keepSynthetic = new Plane(new Vector3(1, 0, 0), 0);  // keeps x >= boundary
  const neckCut = new Plane(new Vector3(0, 1, 0), 0);        // keeps y >= cut (set below, world space)
  const natural = [keepNatural, neckCut];
  const synthetic = [keepSynthetic, neckCut];

  const skin = part("skin"), eyes = part("eyes"), brows = part("brows"), lashes = part("lashes");
  const hair = parts.find((p) => p.name === "hair");
  const textures = parts.map((p) => p.map);

  const skinMat = new MeshStandardMaterial({ map: skin.map, roughness: skin.roughness, metalness: 0, clippingPlanes: natural });
  const eyeMat = new MeshStandardMaterial({ map: eyes.map, roughness: 0.18, metalness: 0, clippingPlanes: natural });
  const overlay = (p: GlbPart) => new MeshStandardMaterial({
    map: p.map, roughness: p.roughness, metalness: 0, transparent: true, depthWrite: false,
    side: DoubleSide, clippingPlanes: natural,
  });
  const browMat = overlay(brows), lashMat = overlay(lashes);
  const hairMat = hair ? new MeshStandardMaterial({
    map: hair.map, roughness: hair.roughness, metalness: 0, alphaTest: hair.alphaCutoff, side: DoubleSide,
    alphaToCoverage: !opts.mobile, clippingPlanes: natural,
  }) : null;
  const ink = new MeshStandardMaterial({ color: 0x0d1015, roughness: 0.62, metalness: 0.35, clippingPlanes: synthetic,
    polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 });
  // Line/point colours are exact brand cyan: excluded from tone mapping.
  const wire = new LineBasicMaterial({ color: 0x00a9b7, transparent: true, opacity: 0.5, depthWrite: false, clippingPlanes: synthetic, toneMapped: false });
  const markLine = new LineBasicMaterial({ color: 0x007f8a, transparent: true, opacity: 0.9, depthWrite: false, clippingPlanes: synthetic, toneMapped: false });
  const dots = new PointsMaterial({ color: 0x00b8c6, size: 0.012, sizeAttenuation: true, transparent: true, opacity: 0.6, depthWrite: false, clippingPlanes: synthetic, toneMapped: false });
  const marks = new PointsMaterial({ color: 0x007f8a, size: 0.05, sizeAttenuation: true, transparent: true, opacity: 1, depthWrite: false, clippingPlanes: synthetic, toneMapped: false });

  const head = new Group();
  const model = new Group();
  model.scale.setScalar(SCALE);
  head.add(model);

  model.add(new Mesh(skin.geometry, skinMat));
  const eyePivots = splitEyes(eyes.geometry).map(({ geo, centre }) => {
    const pivot = new Group();
    pivot.position.copy(centre);
    const natural = new Mesh(geo, eyeMat);
    const dark = new Mesh(geo, ink);
    natural.position.copy(centre).negate();
    dark.position.copy(centre).negate();
    pivot.add(natural, dark);
    model.add(pivot);
    return { pivot, geo };
  });
  const browMesh = new Mesh(brows.geometry, browMat); browMesh.renderOrder = 2;
  const lashMesh = new Mesh(lashes.geometry, lashMat); lashMesh.renderOrder = 2;
  model.add(browMesh, lashMesh);
  if (hair && hairMat) { const h = new Mesh(hair.geometry, hairMat); h.renderOrder = 1; model.add(h); }
  model.add(new Mesh(skin.geometry, ink));
  await yieldTask();

  const wireGeo = new WireframeGeometry(skin.geometry);
  model.add(new LineSegments(wireGeo, wire));
  const lm = landmarkGeometry(skin.geometry);
  await yieldTask();
  const markLines = new LineSegments(lm.lines, markLine); markLines.renderOrder = 3;
  const markPoints = new Points(lm.points, marks); markPoints.renderOrder = 3;
  model.add(markLines, markPoints);
  let cloud: BufferGeometry | null = null;
  if (!opts.mobile) {
    // Sparse point cloud: every 4th skin vertex, pushed slightly off the surface along its normal.
    const sp = skin.geometry.getAttribute("position"), sn = skin.geometry.getAttribute("normal");
    const out: number[] = [];
    for (let i = 0; i < sp.count; i += 4) out.push(sp.getX(i) + sn.getX(i) * 0.002, sp.getY(i) + sn.getY(i) * 0.002, sp.getZ(i) + sn.getZ(i) * 0.002);
    cloud = new BufferGeometry().setAttribute("position", new BufferAttribute(new Float32Array(out), 3));
    model.add(new Points(cloud, dots));
  }
  head.position.set(0, 0.02, 0);
  scene.add(head);
  neckCut.constant = -(head.position.y + NECK_CUT * SCALE);

  let sheetTex = sheetTexture(PALETTE[opts.theme].sheet);
  const sheetMat = new MeshBasicMaterial({ map: sheetTex, transparent: true, depthWrite: false, side: DoubleSide, toneMapped: false });
  const sheetGeo = new PlaneGeometry(0.14, 3.4);
  const sheet = new Mesh(sheetGeo, sheetMat);
  sheet.position.set(0, -0.05, 1.0);
  scene.add(sheet);

  let theme: HeroTheme = opts.theme;
  const setTheme = (next: HeroTheme) => {
    theme = next;
    const p = PALETTE[next];
    skinMat.color.setHex(p.skin); eyeMat.color.setHex(p.skin);
    ink.color.setHex(p.ink); ink.metalness = p.inkMetal;
    wire.color.setHex(p.wire); wire.opacity = p.wireOpacity;
    dots.color.setHex(p.dots); marks.color.setHex(p.marks); markLine.color.setHex(p.marks);
    hemi.color.setHex(p.hemiSky); hemi.groundColor.setHex(p.hemiGround); hemi.intensity = p.hemi;
    key.intensity = p.key; rim.intensity = p.rim;
    const old = sheetTex;
    sheetTex = sheetTexture(p.sheet);
    sheetMat.map = sheetTex; sheetMat.needsUpdate = true;
    old.dispose();
  };
  setTheme(theme);

  await yieldTask();
  // Compile shaders off the critical path where the browser supports parallel compilation.
  await renderer.compileAsync(scene, camera);

  let boundary = 0.12, target = 0.12, px = 0, py = 0, tx = 0, ty = 0;
  // Eye motion: gentle gaze towards the pointer plus rare, small saccades (deterministic LCG, no Math.random).
  let seed = 1234567, nextSaccade = 1800, gazeX = 0, gazeY = 0, sacX = 0, sacY = 0;
  const rand = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };

  const setBoundary = (b: number) => {
    keepNatural.constant = b;
    keepSynthetic.constant = -b;
    sheet.position.x = b;
  };
  setBoundary(boundary);

  const render = (now: number, dt: number) => {
    const moving = dt > 0;
    if (opts.interactive) {
      const k = 1 - Math.exp(-dt / 140);
      boundary += (target - boundary) * k;
      px += (tx - px) * k;
      py += (ty - py) * k;
    } else if (moving) {
      boundary = Math.sin(now * 0.00042) * 0.62;   // automatic sweep on touch devices
    }
    setBoundary(boundary);
    head.rotation.y = -0.38 + px * 0.14 + (moving ? Math.sin(now * 0.00031) * 0.045 : 0);
    head.rotation.x = 0.03 + py * 0.05 + (moving ? Math.sin(now * 0.00023) * 0.012 : 0);
    if (moving) {
      if (now > nextSaccade) {
        sacX = (rand() - 0.5) * 0.08; sacY = (rand() - 0.5) * 0.04;
        nextSaccade = now + 1600 + rand() * 2600;
      }
      const k = 1 - Math.exp(-dt / 60);
      gazeX += (clamp(px * 0.14, -0.12, 0.12) + sacX + 0.06 - gazeX) * k;   // +0.06 rad keeps the gaze towards the viewer
      gazeY += (clamp(py * 0.08, -0.06, 0.06) + sacY - gazeY) * k;
    } else {
      gazeX = 0.06; gazeY = 0;
    }
    for (const { pivot } of eyePivots) { pivot.rotation.y = gazeX; pivot.rotation.x = gazeY; }
    renderer.render(scene, camera);
  };

  const resize = (w: number, h: number) => {
    renderer.setSize(w, h, false);
    camera.aspect = w / Math.max(h, 1);
    camera.position.z = camera.aspect < 0.7 ? 6.6 : 6.0;
    camera.updateProjectionMatrix();
  };

  const geometries = [skin.geometry, eyes.geometry, brows.geometry, lashes.geometry, hair?.geometry, wireGeo,
    lm.lines, lm.points, cloud, sheetGeo, ...eyePivots.map((e) => e.geo)].filter(Boolean) as BufferGeometry[];
  const geometryBytes = [skin.geometry, eyes.geometry, brows.geometry, lashes.geometry, hair?.geometry, wireGeo, lm.lines, lm.points, cloud, sheetGeo]
    .filter(Boolean).reduce((s, g) => s + bytesOf(g as BufferGeometry), 0);
  // RGBA8 + full mip chain (x4/3) per texture: the GPU memory the textures occupy.
  const textureBytes = Math.round(textures.reduce((s, t) => {
    const img = t.image as ImageBitmap;
    return s + img.width * img.height * 4 * (4 / 3);
  }, 0));
  const tri = (g: BufferGeometry) => (g.index ? g.index.count / 3 : 0);
  const triangles = tri(skin.geometry) * 2 + tri(eyes.geometry) * 2 + tri(brows.geometry) + tri(lashes.geometry) + (hair ? tri(hair.geometry) : 0);

  const dispose = () => {
    scene.traverse((o: Object3D) => {
      const m = (o as Mesh).material as Material | Material[] | undefined;
      if (Array.isArray(m)) m.forEach((x) => x.dispose()); else m?.dispose();
    });
    geometries.forEach((g) => g.dispose());
    textures.forEach((t) => { t.dispose(); (t.image as ImageBitmap | undefined)?.close?.(); });
    sheetTex.dispose();
    renderer.dispose();
    renderer.forceContextLoss();
  };

  return {
    render, resize, dispose,
    setTheme: (t) => { if (t !== theme) setTheme(t); },
    setPointer: (nx, ny) => { target = clamp(nx * 1.05, -1, 1); tx = nx; ty = ny; },
    stats: { geometryBytes, textureBytes, assetBytes, triangles, dpr, mobile: opts.mobile },
  };
}
