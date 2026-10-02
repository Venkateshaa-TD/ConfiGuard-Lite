import { rmSync } from "node:fs";
import { MEDIA_DIR } from "./global-setup";

export default function globalTeardown() {
  rmSync(MEDIA_DIR, { recursive: true, force: true }); // test media copies never outlive the run
}
