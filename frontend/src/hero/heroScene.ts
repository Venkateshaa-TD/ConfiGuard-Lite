// Procedural hero: an original sculpted head (no external models, textures or assets).
// Natural side: matte porcelain surface. Synthetic side: dark surface + cyan wireframe + point cloud.
// A vertical scan boundary (clipping planes in world space) reveals one layer or the other.
import {
  BufferGeometry, CanvasTexture, CylinderGeometry, DirectionalLight, DoubleSide, Group, HemisphereLight, LineBasicMaterial,
  LineSegments, Mesh, MeshBasicMaterial, MeshStandardMaterial, PerspectiveCamera, Plane, PlaneGeometry, Points,
  PointsMaterial, Scene, SphereGeometry, Vector3, WebGLRenderer, WireframeGeometry, type Material, type Object3D,
} from "three";

export interface HeroOptions { mobile: boolean; interactive: boolean; context: WebGLRenderingContext | WebGL2RenderingContext }
export interface HeroStats { geometryBytes: number; triangles: number; dpr: number; mobile: boolean }
export interface HeroScene {
  render: (now: number, dt: number) => void;
  setPointer: (nx: number, ny: number) => void;
  resize: (w: number, h: number) => void;
  dispose: () => void;
  stats: HeroStats;
}

const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));
const smooth = (e0: number, e1: number, x: number) => { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); };
const g2 = (x: number, y: number, cx: number, cy: number, sx: number, sy: number) => Math.exp(-(((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2));

/** Displace a unit sphere into an abstract, gender-neutral head. +z faces the viewer. */
function sculptHead(w: number, h: number): BufferGeometry {
  const geo = new SphereGeometry(1, w, h);
  const p = geo.getAttribute("position");
  const v = new Vector3();
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i);
    const { x, y, z } = v;
    const front = smooth(0.05, 0.85, z);
    const ax = Math.abs(x);
    let r = 1 + front * (
      0.19 * g2(x, y, 0, -0.05, 0.085, 0.19) +                 // nose bridge and body
      0.09 * g2(x, y, 0, -0.23, 0.11, 0.055) +                 // nose tip
      -0.12 * g2(ax, y, 0.3, 0.11, 0.13, 0.075) +              // eye sockets
      0.035 * g2(ax, y, 0.3, 0.1, 0.065, 0.04) +               // eyes inside the sockets
      0.05 * g2(x, y, 0, 0.27, 0.46, 0.05) +                   // brow ridge
      0.05 * g2(ax, y, 0.43, -0.08, 0.14, 0.1) +               // cheekbones
      0.055 * g2(x, y, 0, -0.41, 0.19, 0.055) +                // lips
      -0.03 * g2(x, y, 0, -0.455, 0.2, 0.014) +                // mouth line
      0.07 * g2(x, y, 0, -0.66, 0.15, 0.08)                    // chin
    );
    r += 0.06 * g2(z, y, -0.05, 0.02, 0.1, 0.17) * smooth(0.78, 0.98, ax);   // ears
    const jaw = smooth(-0.12, -0.95, y);
    const sx = 0.8 * (1 - 0.3 * jaw);
    const sz = 0.94 * (1 - (z < 0 ? 0.14 : 0.05) * jaw) * (z < 0 ? 1.04 : 1);
    p.setXYZ(i, x * r * sx, y * r * 1.06, z * r * sz + 0.05 * jaw * front);
  }
  geo.computeVertexNormals();
  return geo;
}

function neck(seg: number): BufferGeometry {
  const g = new CylinderGeometry(0.33, 0.42, 1.1, seg, 1, true);
  g.translate(0, -1.32, -0.1);
  return g;
}

/** Soft vertical light sheet for the scan boundary (generated locally, no image files). */
function sheetTexture(): CanvasTexture {
  const c = document.createElement("canvas");
  c.width = 64; c.height = 256;
  const ctx = c.getContext("2d")!;
  const gx = ctx.createLinearGradient(0, 0, 64, 0);
  gx.addColorStop(0, "rgba(0,215,229,0)"); gx.addColorStop(0.5, "rgba(0,215,229,0.9)"); gx.addColorStop(1, "rgba(0,215,229,0)");
  ctx.fillStyle = gx; ctx.fillRect(0, 0, 64, 256);
  ctx.globalCompositeOperation = "destination-in";
  const gy = ctx.createLinearGradient(0, 0, 0, 256);
  gy.addColorStop(0, "rgba(0,0,0,0)"); gy.addColorStop(0.2, "rgba(0,0,0,1)"); gy.addColorStop(0.8, "rgba(0,0,0,1)"); gy.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = gy; ctx.fillRect(0, 0, 64, 256);
  return new CanvasTexture(c);
}

const yieldTask = () => new Promise<void>((r) => setTimeout(r, 0));

/** Built in small steps (yielding between them) so it never blocks the main thread for long. */
export async function createHeroScene(canvas: HTMLCanvasElement, opts: HeroOptions): Promise<HeroScene> {
  const renderer = new WebGLRenderer({ canvas, context: opts.context as WebGL2RenderingContext });
  const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
  renderer.setPixelRatio(dpr);
  renderer.localClippingEnabled = true;
  const gl = renderer.getContext();
  const info = gl.getExtension("WEBGL_debug_renderer_info");
  const gpu = String(info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER));
  if (/swiftshader|llvmpipe|software|basic render/i.test(gpu)) {   // too slow for a continuous 3D hero
    renderer.dispose();
    renderer.forceContextLoss();
    throw new Error("software-renderer");
  }
  renderer.setClearColor(0x000000, 0);

  const scene = new Scene();
  const camera = new PerspectiveCamera(27, 1, 0.1, 50);
  camera.position.set(0, 0.02, 6.6);

  scene.add(new HemisphereLight(0xffffff, 0x8d9aa3, 1.15));
  const key = new DirectionalLight(0xffffff, 2.1); key.position.set(-3.2, 2.6, 4); scene.add(key);
  const rim = new DirectionalLight(0x00d7e5, 1.6); rim.position.set(3.5, 1.2, -2.5); scene.add(rim);

  const keepNatural = new Plane(new Vector3(-1, 0, 0), 0);   // keeps x <= boundary
  const keepSynthetic = new Plane(new Vector3(1, 0, 0), 0);  // keeps x >= boundary

  const hi = opts.mobile ? [44, 32] : [112, 84];
  const lo = opts.mobile ? [28, 20] : [56, 40];
  await yieldTask();
  const headHi = sculptHead(hi[0]!, hi[1]!);
  await yieldTask();
  const headLo = sculptHead(lo[0]!, lo[1]!);
  const neckGeo = neck(opts.mobile ? 20 : 48);
  await yieldTask();

  const porcelain = new MeshStandardMaterial({ color: 0xdfe5e8, roughness: 0.62, metalness: 0.02, clippingPlanes: [keepNatural] });
  const ink = new MeshStandardMaterial({ color: 0x0d1015, roughness: 0.38, metalness: 0.35, clippingPlanes: [keepSynthetic],
    polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 });
  const wire = new LineBasicMaterial({ color: 0x00d7e5, transparent: true, opacity: 0.55, depthWrite: false, clippingPlanes: [keepSynthetic] });
  const dots = new PointsMaterial({ color: 0x5ff0f8, size: 0.018, sizeAttenuation: true, transparent: true, opacity: 0.95, depthWrite: false, clippingPlanes: [keepSynthetic] });

  const head = new Group();
  head.add(new Mesh(headHi, porcelain), new Mesh(neckGeo, porcelain));
  head.add(new Mesh(headHi, ink), new Mesh(neckGeo, ink));
  const wireGeo = new WireframeGeometry(headLo);
  await yieldTask();
  head.add(new LineSegments(wireGeo, wire));
  let pointGeo: BufferGeometry | null = null;
  if (!opts.mobile) {
    pointGeo = new BufferGeometry().setAttribute("position", headHi.getAttribute("position").clone());
    head.add(new Points(pointGeo, dots));
  }
  head.position.set(0, 0.32, 0);
  scene.add(head);

  const sheetTex = sheetTexture();
  const sheetGeo = new PlaneGeometry(0.16, 3.6);
  const sheet = new Mesh(sheetGeo, new MeshBasicMaterial({ map: sheetTex, transparent: true, depthWrite: false, side: DoubleSide }));
  sheet.position.set(0, 0.05, 1.2);
  scene.add(sheet);

  await yieldTask();
  // Compile shaders off the critical path where the browser supports parallel compilation.
  await renderer.compileAsync(scene, camera);

  let boundary = 0.12, target = 0.12, px = 0, py = 0, tx = 0, ty = 0;

  const setBoundary = (b: number) => {
    keepNatural.constant = b;
    keepSynthetic.constant = -b;
    sheet.position.x = b;
  };
  setBoundary(boundary);

  const render = (now: number, dt: number) => {
    if (opts.interactive) {
      const k = 1 - Math.exp(-dt / 140);
      boundary += (target - boundary) * k;
      px += (tx - px) * k;
      py += (ty - py) * k;
    } else if (dt > 0) {
      boundary = Math.sin(now * 0.00042) * 0.62;   // automatic sweep on touch devices
    }
    setBoundary(boundary);
    head.rotation.y = -0.42 + px * 0.16 + (dt > 0 ? Math.sin(now * 0.00031) * 0.05 : 0);
    head.rotation.x = 0.04 + py * 0.06;
    renderer.render(scene, camera);
  };

  const resize = (w: number, h: number) => {
    renderer.setSize(w, h, false);
    camera.aspect = w / Math.max(h, 1);
    camera.position.z = camera.aspect < 0.85 ? 7.6 : 6.6;
    camera.updateProjectionMatrix();
  };

  const geometries = [headHi, headLo, neckGeo, wireGeo, sheetGeo, pointGeo].filter(Boolean) as BufferGeometry[];
  const geometryBytes = geometries.reduce((sum, g) => {
    let b = g.index ? g.index.array.byteLength : 0;
    for (const name of Object.keys(g.attributes)) b += (g.getAttribute(name).array as ArrayLike<number> & { byteLength: number }).byteLength;
    return sum + b;
  }, 0);
  const triangles = (headHi.index ? headHi.index.count / 3 : 0) * 2 + (neckGeo.index ? neckGeo.index.count / 3 : 0) * 2;

  const dispose = () => {
    scene.traverse((o: Object3D) => {
      const m = (o as Mesh).material as Material | Material[] | undefined;
      if (Array.isArray(m)) m.forEach((x) => x.dispose()); else m?.dispose();
    });
    geometries.forEach((g) => g.dispose());
    sheetTex.dispose();
    renderer.dispose();
    renderer.forceContextLoss();
  };

  return {
    render, resize, dispose,
    setPointer: (nx, ny) => { target = clamp(nx * 1.05, -1, 1); tx = nx; ty = ny; },
    stats: { geometryBytes, triangles, dpr, mobile: opts.mobile },
  };
}
