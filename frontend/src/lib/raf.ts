// One shared requestAnimationFrame loop for every animated element on the page.
// It runs only while at least one subscriber exists and the tab is visible.
type Tick = (now: number, dt: number) => void;

const subscribers = new Set<Tick>();
let handle = 0;
let last = 0;

function frame(now: number) {
  const dt = last ? Math.min(now - last, 100) : 16;
  last = now;
  subscribers.forEach((fn) => fn(now, dt));
  handle = subscribers.size && !document.hidden ? requestAnimationFrame(frame) : 0;
  if (!handle) last = 0;
}

function start() {
  if (!handle && subscribers.size && !document.hidden) handle = requestAnimationFrame(frame);
}

if (typeof document !== "undefined") {
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && handle) {
      cancelAnimationFrame(handle);
      handle = 0;
      last = 0;
    } else start();
  });
}

export function subscribe(fn: Tick): () => void {
  subscribers.add(fn);
  start();
  return () => {
    subscribers.delete(fn);
    if (!subscribers.size && handle) {
      cancelAnimationFrame(handle);
      handle = 0;
      last = 0;
    }
  };
}

export const activeSubscribers = () => subscribers.size;
