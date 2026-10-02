// Runs before first paint (external file: CSP-compatible, no inline script).
// Applies the saved theme preference ("light" | "dark" | "system") so the page never flashes the wrong theme.
// Only the key "cg-theme" is ever read or written by this site.
(function () {
  var pref = "system";
  try {
    var v = window.localStorage.getItem("cg-theme");
    if (v === "light" || v === "dark" || v === "system") pref = v;
  } catch (e) { /* storage blocked: follow the system setting */ }
  var dark = pref === "dark" || (pref === "system" && window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
  var root = document.documentElement;
  root.setAttribute("data-theme", dark ? "dark" : "light");
  root.setAttribute("data-theme-pref", pref);
  root.style.colorScheme = dark ? "dark" : "light";
})();
