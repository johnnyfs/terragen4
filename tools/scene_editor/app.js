"use strict";
/* Terragen scene editor — no-build vanilla JS single-page app.
   The parsed YAML document is the source of truth held in `state.doc`;
   all edits mutate it locally and are only persisted on explicit Save. */

// ---- engine-grounded enums + friendly labels -----------------------------
const GEOMETRY_TYPES = ["box", "oriented_box", "sphere", "ellipsoid", "capsule_path", "plane"];
const CAPSULE_TYPES = new Set(["capsule_path", "capsule", "capsule_line"]);
const RADIUS_TYPES = new Set([...CAPSULE_TYPES, "sphere"]);

const INFLUENCE_FIELDS = [
  { v: "height", label: "height" },
  { v: "octave_amplitude", label: "octave amplitude" },
  { v: "warp_amount_scale", label: "warp amount" },
  { v: "ridge_amount_scale", label: "ridge amount" },
  { v: "carve_amount", label: "carve" },
  { v: "material_id", label: "material" },
  { v: "bank_steepness", label: "bank steepness (unsupported)" },
];
const BLEND_MODES = [
  { v: "add", label: "add" },
  { v: "multiply", label: "multiply" },
  { v: "pull_to_target", label: "pull to target" },
  { v: "pull_to_path_grade", label: "pull to path grade" },
  { v: "subtract", label: "subtract" },
  { v: "override", label: "override" },
  { v: "clamp_min", label: "clamp min (unsupported)" },
  { v: "clamp_max", label: "clamp max (unsupported)" },
];
const TRANSITION_TYPES = ["step", "linear", "smoothstep", "smootherstep"];
const OCTAVE_KINDS = ["fbm", "ridged"];
const FALLBACK_OCTAVES = ["broad", "detail", "ridges"];

// ---- templates (engine-grounded field names) -----------------------------
const TEMPLATES = {
  mountain: {
    label: "Add mountain root",
    make: () => ({
      kind: "mountain",
      priority: 10,
      geometry: { type: "box", center: [0, 0, 0], size: [1200, 500, 1200], mask_warp: 0.0 },
      transition: { type: "smoothstep", falloff: 120, cutoff_epsilon: 0.01 },
      influences: [
        { field: "height", mode: "add", value: 120 },
        { field: "octave_amplitude", octave: "ridges", mode: "multiply", value: 1.4 },
      ],
    }),
  },
  valley: {
    label: "Add valley child",
    make: () => ({
      kind: "valley",
      priority: 40,
      geometry: { type: "capsule_path", points: [[-300, 40, 120], [-100, 35, 30], [160, 32, -90]], radius: 160, mask_warp: 0.2 },
      transition: { type: "smoothstep", falloff: 90, cutoff_epsilon: 0.01 },
      influences: [
        { field: "height", mode: "pull_to_target", target: 32, strength: 0.8 },
        { field: "octave_amplitude", octave: "ridges", mode: "multiply", value: 0.25 },
        { field: "octave_amplitude", octave: "detail", mode: "multiply", value: 0.35 },
      ],
    }),
  },
  plateau: {
    label: "Add plateau child",
    make: () => ({
      kind: "plateau",
      priority: 60,
      geometry: { type: "box", center: [100, 90, -40], size: [250, 80, 220], mask_warp: 0.0 },
      transition: { type: "smoothstep", falloff: 45, cutoff_epsilon: 0.01 },
      influences: [
        { field: "height", mode: "pull_to_target", target: 95, strength: 0.9 },
        { field: "octave_amplitude", octave: "detail", mode: "multiply", value: 0.15 },
        { field: "octave_amplitude", octave: "ridges", mode: "multiply", value: 0.2 },
      ],
    }),
  },
  path: {
    label: "Add path/ramp child",
    make: () => ({
      kind: "path",
      priority: 80,
      geometry: { type: "capsule_path", points: [[120, 95, -10], [80, 75, -80], [20, 45, -160], [-40, 35, -220]], radius: 18, mask_warp: 0.2 },
      transition: { type: "smoothstep", falloff: 30, cutoff_epsilon: 0.01 },
      influences: [
        { field: "height", mode: "pull_to_path_grade", strength: 0.9 },
        { field: "octave_amplitude", octave: "detail", mode: "multiply", value: 0.1 },
        { field: "octave_amplitude", octave: "ridges", mode: "multiply", value: 0.05 },
      ],
    }),
  },
  riverbed: {
    label: "Add dry riverbed child",
    make: () => ({
      kind: "riverbed",
      priority: 90,
      geometry: { type: "capsule_path", points: [[-240, 65, 80], [-120, 45, -40], [20, 32, -180], [220, 25, -300]], radius: 24, mask_warp: 0.3 },
      transition: { type: "smoothstep", falloff: 36, cutoff_epsilon: 0.01 },
      influences: [
        { field: "height", mode: "add", value: -18 },
        { field: "octave_amplitude", octave: "detail", mode: "multiply", value: 0.2 },
      ],
    }),
  },
  peak: {
    label: "Add peak",
    make: () => ({
      kind: "peak",
      priority: 70,
      geometry: { type: "ellipsoid", center: [0, 120, 0], size: [180, 220, 180], mask_warp: 0.1 },
      transition: { type: "smoothstep", falloff: 50, cutoff_epsilon: 0.01 },
      influences: [
        { field: "height", mode: "add", value: 160 },
        { field: "octave_amplitude", octave: "ridges", mode: "multiply", value: 1.6 },
      ],
    }),
  },
  cave: {
    label: "Add cave/cut placeholder",
    make: () => ({
      kind: "cave",
      priority: 100,
      geometry: { type: "sphere", center: [0, 0, 0], radius: 60, mask_warp: 0.0 },
      transition: { type: "smoothstep", falloff: 20, cutoff_epsilon: 0.01 },
      influences: [
        { field: "carve_amount", mode: "add", value: 1.0 },
      ],
    }),
  },
};

// ---- state ----------------------------------------------------------------
const state = {
  scenes: [],
  path: null,
  doc: null,
  pristine: null,   // JSON string of last-saved doc
  dirty: false,
  selected: -1,     // -1 = scene settings; otherwise region index
  lastSaved: null,
  validation: { errors: [], warnings: [] },
};

// ---- tiny DOM helpers -----------------------------------------------------
function el(tag, attrs, children) {
  const e = document.createElement(tag);
  if (attrs) for (const k in attrs) {
    if (k === "class") e.className = attrs[k];
    else if (k === "text") e.textContent = attrs[k];
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), attrs[k]);
    else if (attrs[k] !== undefined && attrs[k] !== null) e.setAttribute(k, attrs[k]);
  }
  if (children) for (const c of [].concat(children)) {
    if (c == null) continue;
    e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return e;
}
const $ = (id) => document.getElementById(id);
function clear(n) { while (n.firstChild) n.removeChild(n.firstChild); }

function num(raw, integer) {
  if (raw === "" || raw == null) return undefined;
  const n = integer ? parseInt(raw, 10) : parseFloat(raw);
  return Number.isFinite(n) ? n : undefined;
}

// ---- API ------------------------------------------------------------------
const api = {
  async scenes() { return (await fetch("/api/scenes").then(r => r.json())).scenes; },
  async scene(path) { return fetch("/api/scene?path=" + encodeURIComponent(path)).then(r => r.json()); },
  async save(path, document) {
    return fetch("/api/scene", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ path, document }) }).then(r => r.json());
  },
  async validate(document) {
    return fetch("/api/validate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ document }) }).then(r => r.json());
  },
  async format(document) {
    return fetch("/api/format", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ document }) }).then(r => r.json());
  },
};

// ---- doc accessors --------------------------------------------------------
function regions() {
  if (!state.doc) return [];
  if (!Array.isArray(state.doc.regions)) state.doc.regions = [];
  return state.doc.regions;
}
function octaveNames() {
  const oc = state.doc && state.doc.scene && state.doc.scene.noise && state.doc.scene.noise.octaves;
  if (Array.isArray(oc) && oc.length) return oc.map(o => o && o.name).filter(Boolean);
  return FALLBACK_OCTAVES;
}
function hasNamedOctaves() {
  const oc = state.doc && state.doc.scene && state.doc.scene.noise && state.doc.scene.noise.octaves;
  return Array.isArray(oc) && oc.length > 0;
}
function uniqueId(base) {
  const taken = new Set(regions().map(r => r && r.id));
  if (!taken.has(base)) return base;
  let i = 2;
  while (taken.has(base + "_" + i)) i++;
  return base + "_" + i;
}

// ---- dirty / chrome -------------------------------------------------------
let syncTimer = null;
function touch() {
  state.dirty = state.doc && JSON.stringify(state.doc) !== state.pristine;
  updateChrome();
  clearTimeout(syncTimer);
  syncTimer = setTimeout(syncValidationAndPreview, 200);
}
function updateChrome() {
  $("dirty-badge").classList.toggle("hidden", !state.dirty);
  $("btn-save").disabled = !state.doc;
  $("btn-revert").disabled = !state.dirty;
  const vb = $("valid-badge");
  const errs = state.validation.errors.length;
  if (!state.doc) { vb.textContent = ""; vb.className = "badge"; }
  else if (errs === 0) { vb.textContent = "valid"; vb.className = "badge ok"; }
  else { vb.textContent = errs + " error" + (errs > 1 ? "s" : ""); vb.className = "badge bad"; }
  $("saved-at").textContent = state.lastSaved ? "saved " + state.lastSaved : "";
}

async function syncValidationAndPreview() {
  if (!state.doc) return;
  const snapshot = JSON.parse(JSON.stringify(state.doc));
  try {
    const [v, f] = await Promise.all([api.validate(snapshot), api.format(snapshot)]);
    state.validation = { errors: v.errors || [], warnings: v.warnings || [] };
    renderValidation();
    $("yaml-preview").textContent = f.ok ? f.raw : ("# format error: " + (f.error || "unknown"));
    updateChrome();
  } catch (e) { /* server offline; ignore */ }
}

function toast(msg, kind) {
  const t = $("toast");
  t.textContent = msg;
  t.className = "toast " + (kind || "");
  setTimeout(() => t.classList.add("hidden"), 2600);
  t.classList.remove("hidden");
}

// ---- field builders -------------------------------------------------------
function field(label, control, hint) {
  return el("div", { class: "field" }, [el("label", { text: label }), control, hint ? el("div", { class: "hint", text: hint }) : null]);
}
function textInput(obj, key, ph) {
  const i = el("input", { type: "text", value: obj[key] != null ? obj[key] : "", placeholder: ph || "" });
  i.addEventListener("input", () => { const v = i.value; if (v === "") delete obj[key]; else obj[key] = v; touch(); });
  return i;
}
function numInput(obj, key, opts) {
  opts = opts || {};
  const i = el("input", { type: "number", step: opts.step || "any", value: obj[key] != null ? obj[key] : "" });
  i.addEventListener("input", () => { const v = num(i.value, opts.integer); if (v === undefined) delete obj[key]; else obj[key] = v; touch(); });
  return i;
}
function selectInput(obj, key, options, onChange) {
  const s = el("select");
  for (const o of options) {
    const v = typeof o === "string" ? o : o.v;
    const label = typeof o === "string" ? o : o.label;
    s.appendChild(el("option", { value: v, text: label }));
  }
  s.value = obj[key] != null ? obj[key] : "";
  s.addEventListener("change", () => { obj[key] = s.value; touch(); if (onChange) onChange(); });
  return s;
}
function checkInput(obj, key) {
  const i = el("input", { type: "checkbox" });
  i.checked = !!obj[key];
  i.style.width = "auto";
  i.addEventListener("change", () => { if (i.checked) obj[key] = true; else delete obj[key]; touch(); });
  return el("div", {}, [i]);
}
function vec3Input(obj, key) {
  if (!Array.isArray(obj[key])) obj[key] = [0, 0, 0];
  const arr = obj[key];
  const wrap = el("div", { class: "vec3" });
  ["x", "y", "z"].forEach((axis, idx) => {
    const i = el("input", { type: "number", step: "any", value: arr[idx] != null ? arr[idx] : 0, title: axis });
    i.addEventListener("input", () => { const v = num(i.value); arr[idx] = v === undefined ? 0 : v; touch(); });
    wrap.appendChild(i);
  });
  return wrap;
}
function sliderInput(obj, key, min, max, step) {
  if (obj[key] == null) obj[key] = min;
  const wrap = el("div", { class: "slider-row" });
  const r = el("input", { type: "range", min, max, step });
  const out = el("input", { type: "number", step, class: "sval" });
  r.value = obj[key]; out.value = obj[key];
  const set = (v) => { const n = num(v); if (n !== undefined) { obj[key] = n; r.value = n; out.value = n; touch(); } };
  r.addEventListener("input", () => set(r.value));
  out.addEventListener("input", () => set(out.value));
  wrap.appendChild(r); wrap.appendChild(out);
  return wrap;
}

// ---- region tree ----------------------------------------------------------
function buildForest() {
  const rs = regions();
  const byId = new Map();
  rs.forEach((r, i) => { if (r && r.id) byId.set(r.id, i); });
  const children = new Map();
  const roots = [];
  rs.forEach((r, i) => {
    const p = r && r.parent;
    if (p && byId.has(p)) {
      if (!children.has(p)) children.set(p, []);
      children.get(p).push(i);
    } else {
      roots.push(i);
    }
  });
  return { children, roots, byId };
}
function errorIndexSet() {
  const s = new Set();
  for (const e of state.validation.errors) {
    const m = /^regions\[(\d+)\]/.exec(e.path || "");
    if (m) s.add(parseInt(m[1], 10));
  }
  return s;
}
function warnIndexSet() {
  const s = new Set();
  for (const w of state.validation.warnings) {
    const m = /^regions\[(\d+)\]/.exec(w.path || "");
    if (m) s.add(parseInt(m[1], 10));
  }
  return s;
}
function renderTree() {
  const root = $("region-tree");
  clear(root);
  const rs = regions();
  if (!state.doc) { root.appendChild(el("div", { class: "empty-hint", text: "no scene loaded" })); return; }
  const { children, roots } = buildForest();
  const errs = errorIndexSet();
  const warns = warnIndexSet();

  const sceneRow = el("div", { class: "tree-row" + (state.selected === -1 ? " selected" : ""), onclick: () => select(-1) },
    [el("span", { class: "twisty", text: "" }), el("span", { class: "label", text: "⚙ scene settings" })]);
  root.appendChild(sceneRow);

  function node(idx) {
    const r = rs[idx];
    const kids = children.get(r && r.id) || [];
    const cls = "tree-row" + (state.selected === idx ? " selected" : "") + (errs.has(idx) ? " invalid" : "");
    const row = el("div", { class: cls, onclick: (ev) => { ev.stopPropagation(); select(idx); } }, [
      el("span", { class: "twisty", text: kids.length ? "▾" : "" }),
      el("span", { class: "label", text: r && r.id ? r.id : "(no id)" }),
      r && r.kind ? el("span", { class: "kind", text: r.kind }) : null,
      warns.has(idx) ? el("span", { class: "warn", text: "⚠" }) : null,
    ]);
    const wrap = el("div", { class: "tree-node" }, [row]);
    if (kids.length) {
      const ch = el("div", { class: "tree-children" });
      kids.forEach(k => ch.appendChild(node(k)));
      wrap.appendChild(ch);
    }
    return wrap;
  }
  roots.forEach(i => root.appendChild(node(i)));
}

function renderTemplates() {
  const c = $("templates");
  clear(c);
  for (const key in TEMPLATES) {
    const t = TEMPLATES[key];
    c.appendChild(el("button", { text: t.label, onclick: () => addFromTemplate(key) }));
  }
}

// ---- selection ------------------------------------------------------------
function select(idx) {
  state.selected = idx;
  renderTree();
  renderEditor();
}

// ---- structural ops -------------------------------------------------------
function addFromTemplate(key) {
  const t = TEMPLATES[key];
  const r = t.make();
  const parentId = state.selected >= 0 ? regions()[state.selected].id : (key === "mountain" ? null : null);
  if (parentId) r.parent = parentId;
  r.id = uniqueId(r.kind || "region");
  regions().push(r);
  state.selected = regions().length - 1;
  renderTree(); renderEditor(); touch();
  toast("added " + r.id, "good");
}
function addRoot() {
  const r = { id: uniqueId("region"), kind: "", priority: 0,
    geometry: { type: "box", center: [0, 0, 0], size: [200, 100, 200], mask_warp: 0.0 },
    transition: { type: "smoothstep", falloff: 40, cutoff_epsilon: 0.01 }, influences: [] };
  regions().push(r);
  state.selected = regions().length - 1;
  renderTree(); renderEditor(); touch();
}
function duplicateRegion(idx) {
  const copy = JSON.parse(JSON.stringify(regions()[idx]));
  copy.id = uniqueId((copy.id || "region") + "_copy");
  regions().splice(idx + 1, 0, copy);
  state.selected = idx + 1;
  renderTree(); renderEditor(); touch();
}
function deleteRegion(idx) {
  const rs = regions();
  const target = rs[idx];
  // reparent children to the deleted node's parent (keeps graph valid)
  const newParent = target.parent;
  rs.forEach(r => { if (r.parent === target.id) { if (newParent) r.parent = newParent; else delete r.parent; } });
  rs.splice(idx, 1);
  state.selected = -1;
  renderTree(); renderEditor(); touch();
}

// ---- editor: dispatch -----------------------------------------------------
function renderEditor() {
  const root = $("editor");
  clear(root);
  if (!state.doc) { root.appendChild(el("div", { class: "empty-hint", text: "Select a scene above to begin." })); return; }
  if (state.selected === -1) renderSceneEditor(root);
  else renderRegionEditor(root, state.selected);
}

// ---- editor: scene settings ----------------------------------------------
function renderSceneEditor(root) {
  const doc = state.doc;
  if (!doc.scene) doc.scene = {};
  const s = doc.scene;
  s.height = s.height || {};
  s.noise = s.noise || {};
  s.warp = s.warp || {};
  s.material = s.material || {};

  const sec = (title, controls, extra) => {
    const body = el("div", { class: "form-section" }, [el("h3", {}, [el("span", { text: title }), extra || null])]);
    controls.forEach(c => body.appendChild(c));
    return body;
  };

  root.appendChild(sec("Scene", [
    field("id", textInput(s, "id")),
    field("name", textInput(s, "name")),
    field("version", numInput(doc, "version", { integer: true })),
  ]));

  root.appendChild(sec("Height bounds", [
    field("min", numInput(s.height, "min")),
    field("max", numInput(s.height, "max")),
    field("base", numInput(s.height, "base")),
  ]));

  const noise = s.noise;
  noise.legacy = noise.legacy || {};
  root.appendChild(sec("Noise", [
    field("seed", numInput(noise, "seed", { integer: true })),
    el("div", { class: "hint", text: "legacy (single-band fBm):" }),
    field("legacy.frequency", numInput(noise.legacy, "frequency")),
    field("legacy.amplitude", numInput(noise.legacy, "amplitude")),
    field("legacy.octaves", numInput(noise.legacy, "octaves", { integer: true })),
    field("legacy.lacunarity", numInput(noise.legacy, "lacunarity")),
    field("legacy.gain", numInput(noise.legacy, "gain")),
  ]));

  root.appendChild(renderOctaveTable(noise));

  root.appendChild(sec("Warp", [
    field("frequency", numInput(s.warp, "frequency")),
    field("amplitude", numInput(s.warp, "amplitude")),
  ]));

  root.appendChild(sec("Material", [
    field("default_vibe", textInput(s.material, "default_vibe")),
    field("default_flat_material", textInput(s.material, "default_flat_material")),
    field("default_slope_material", textInput(s.material, "default_slope_material")),
  ]));
}

function renderOctaveTable(noise) {
  const sec = el("div", { class: "form-section" });
  const addBtn = el("button", { text: "+ octave", onclick: () => {
    noise.octaves = noise.octaves || [];
    noise.octaves.push({ name: "octave_" + (noise.octaves.length + 1), kind: "fbm", frequency: 0.05, amplitude: 4.0 });
    renderEditor(); touch();
  } });
  sec.appendChild(el("h3", {}, [el("span", { text: "Named octaves" }), addBtn]));

  const legacyId = state.doc.scene && state.doc.scene.id;
  if (legacyId === "legacy_default" && Array.isArray(noise.octaves) && noise.octaves.length) {
    sec.appendChild(el("div", { class: "warn-inline", text: "legacy_default should keep only noise.legacy and define no named octaves." }));
  }

  const octs = noise.octaves || [];
  if (!octs.length) { sec.appendChild(el("div", { class: "hint", text: "none — influences will use legacy/fallback octave names (" + FALLBACK_OCTAVES.join(", ") + ")." })); return sec; }
  octs.forEach((o, i) => {
    const card = el("div", { class: "card" });
    card.appendChild(el("div", { class: "card-head" }, [
      el("span", { class: "title", text: o.name || "octave " + (i + 1) }),
      el("div", { class: "ctrls" }, [el("button", { text: "✕", title: "remove", onclick: () => { octs.splice(i, 1); if (!octs.length) delete noise.octaves; renderEditor(); touch(); } })]),
    ]));
    card.appendChild(field("name", textInput(o, "name")));
    card.appendChild(field("kind", selectInput(o, "kind", OCTAVE_KINDS)));
    card.appendChild(field("frequency", numInput(o, "frequency")));
    card.appendChild(field("amplitude", numInput(o, "amplitude")));
    card.appendChild(field("sharpness", numInput(o, "sharpness"), "ridged only"));
    sec.appendChild(card);
  });
  return sec;
}

// ---- editor: region -------------------------------------------------------
function renderRegionEditor(root, idx) {
  const r = regions()[idx];
  if (!r) { root.appendChild(el("div", { class: "empty-hint", text: "region not found" })); return; }

  // header + structural actions
  const head = el("div", { class: "form-section" });
  head.appendChild(el("h3", {}, [el("span", { text: "Region: " + (r.id || "(no id)") })]));
  head.appendChild(el("div", { class: "row-actions" }, [
    el("button", { text: "+ child", onclick: () => addChild(idx) }),
    el("button", { text: "duplicate", onclick: () => duplicateRegion(idx) }),
    el("button", { text: "delete", onclick: () => { if (confirm("Delete region '" + r.id + "'? Children will be reparented.")) deleteRegion(idx); } }),
  ]));
  root.appendChild(head);

  // identity
  const parentOptions = [{ v: "", label: "(none — root)" }].concat(
    regions().filter((o, i) => i !== idx && o.id).map(o => ({ v: o.id, label: o.id })));
  const idCtl = textInput(r, "id");
  idCtl.addEventListener("input", () => { renderTree(); });
  const ident = el("div", { class: "form-section" }, [el("h3", {}, [el("span", { text: "Identity" })])]);
  ident.appendChild(field("id", idCtl));
  ident.appendChild(field("parent", (() => {
    const sel = selectInput(r, "parent", parentOptions, () => renderTree());
    if (!r.parent) sel.value = "";
    sel.addEventListener("change", () => { if (sel.value === "") delete r.parent; });
    return sel;
  })()));
  const kindCtl = textInput(r, "kind");
  kindCtl.addEventListener("input", () => renderTree());
  ident.appendChild(field("kind", kindCtl));
  ident.appendChild(field("priority", numInput(r, "priority", { integer: true }), "higher wins where regions overlap"));
  ident.appendChild(field("allow_outside_parent", checkInput(r, "allow_outside_parent")));
  root.appendChild(ident);

  // geometry
  root.appendChild(renderGeometry(r));
  // transition
  root.appendChild(renderTransition(r));
  // influences
  root.appendChild(renderInfluences(r));
}

function addChild(idx) {
  const parent = regions()[idx];
  const r = { id: uniqueId((parent.id || "region") + "_child"), parent: parent.id, kind: "",
    geometry: { type: "box", center: [0, 0, 0], size: [200, 100, 200], mask_warp: 0.0 },
    transition: { type: "smoothstep", falloff: 40, cutoff_epsilon: 0.01 }, influences: [] };
  regions().push(r);
  state.selected = regions().length - 1;
  renderTree(); renderEditor(); touch();
}

function renderGeometry(r) {
  r.geometry = r.geometry || { type: "box" };
  const g = r.geometry;
  const sec = el("div", { class: "form-section" }, [el("h3", {}, [el("span", { text: "Geometry" })])]);
  sec.appendChild(field("type", selectInput(g, "type", GEOMETRY_TYPES, () => { renderEditor(); })));
  const t = g.type;

  if (CAPSULE_TYPES.has(t)) {
    sec.appendChild(renderPoints(g));
    sec.appendChild(field("radius", numInput(g, "radius")));
    sec.appendChild(field("mask_warp", numInput(g, "mask_warp")));
    sec.appendChild(pathPreview(g));
  } else if (t === "sphere") {
    sec.appendChild(field("center", vec3Input(g, "center")));
    sec.appendChild(field("radius", numInput(g, "radius")));
    sec.appendChild(field("mask_warp", numInput(g, "mask_warp")));
  } else if (t === "ellipsoid") {
    sec.appendChild(field("center", vec3Input(g, "center")));
    sec.appendChild(field("size", vec3Input(g, "size")));
    sec.appendChild(field("mask_warp", numInput(g, "mask_warp")));
  } else if (t === "oriented_box" || t === "plane" || t === "ramp") {
    sec.appendChild(field("center", vec3Input(g, "center")));
    sec.appendChild(field("size", vec3Input(g, "size")));
    sec.appendChild(field("rotation_y", numInput(g, "rotation_y"), "degrees"));
    sec.appendChild(field("mask_warp", numInput(g, "mask_warp")));
  } else { // box and fallback
    sec.appendChild(field("center", vec3Input(g, "center")));
    sec.appendChild(field("size", vec3Input(g, "size")));
    sec.appendChild(field("mask_warp", numInput(g, "mask_warp")));
  }
  return sec;
}

function renderPoints(g) {
  if (!Array.isArray(g.points)) g.points = [];
  const wrap = el("div", {});
  wrap.appendChild(el("label", { text: "points (x, y, z)", class: "hint" }));
  const list = el("div", { class: "points-list" });
  g.points.forEach((p, i) => {
    if (!Array.isArray(p)) { p = [0, 0, 0]; g.points[i] = p; }
    const row = el("div", { class: "point-row" });
    row.appendChild(el("span", { class: "idx", text: i }));
    [0, 1, 2].forEach(axis => {
      const inp = el("input", { type: "number", step: "any", value: p[axis] != null ? p[axis] : 0 });
      inp.addEventListener("input", () => { const v = num(inp.value); p[axis] = v === undefined ? 0 : v; touch(); refreshPreview(g); });
      row.appendChild(inp);
    });
    row.appendChild(el("div", { class: "pt-ctrls" }, [
      el("button", { text: "↑", title: "move up", onclick: () => { if (i > 0) { [g.points[i - 1], g.points[i]] = [g.points[i], g.points[i - 1]]; renderEditor(); touch(); } } }),
      el("button", { text: "↓", title: "move down", onclick: () => { if (i < g.points.length - 1) { [g.points[i + 1], g.points[i]] = [g.points[i], g.points[i + 1]]; renderEditor(); touch(); } } }),
      el("button", { text: "✕", title: "delete", onclick: () => { g.points.splice(i, 1); renderEditor(); touch(); } }),
    ]));
    list.appendChild(row);
  });
  wrap.appendChild(list);
  wrap.appendChild(el("div", { class: "row-actions" }, [
    el("button", { text: "+ point", onclick: () => {
      const last = g.points[g.points.length - 1] || [0, 0, 0];
      g.points.push([last[0] + 20, last[1], last[2] + 20]);
      renderEditor(); touch();
    } }),
  ]));
  return wrap;
}

let _previewG = null;
function pathPreview(g) {
  _previewG = g;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "preview2d");
  svg.setAttribute("id", "path-preview");
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", "160");
  drawPreview(svg, g);
  return svg;
}
function refreshPreview(g) {
  const svg = $("path-preview");
  if (svg && g === _previewG) drawPreview(svg, g);
}
function drawPreview(svg, g) {
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  const pts = (g.points || []).filter(p => Array.isArray(p));
  if (!pts.length) return;
  const W = svg.clientWidth || 320, H = 160, pad = 16;
  const xs = pts.map(p => p[0]), zs = pts.map(p => p[2]);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minZ = Math.min(...zs), maxZ = Math.max(...zs);
  const sx = v => pad + (maxX === minX ? (W - 2 * pad) / 2 : ((v - minX) / (maxX - minX)) * (W - 2 * pad));
  const sz = v => pad + (maxZ === minZ ? (H - 2 * pad) / 2 : ((v - minZ) / (maxZ - minZ)) * (H - 2 * pad));
  const NS = "http://www.w3.org/2000/svg";
  const poly = document.createElementNS(NS, "polyline");
  poly.setAttribute("points", pts.map(p => sx(p[0]) + "," + sz(p[2])).join(" "));
  poly.setAttribute("fill", "none"); poly.setAttribute("stroke", "#2563eb"); poly.setAttribute("stroke-width", "2");
  svg.appendChild(poly);
  pts.forEach((p, i) => {
    const c = document.createElementNS(NS, "circle");
    c.setAttribute("cx", sx(p[0])); c.setAttribute("cy", sz(p[2])); c.setAttribute("r", "4");
    c.setAttribute("fill", "#1d4ed8");
    svg.appendChild(c);
    const tx = document.createElementNS(NS, "text");
    tx.setAttribute("x", sx(p[0]) + 6); tx.setAttribute("y", sz(p[2]) - 6);
    tx.setAttribute("font-size", "9"); tx.setAttribute("fill", "#6b7280");
    tx.textContent = i + " (y=" + p[1] + ")";
    svg.appendChild(tx);
  });
}

function renderTransition(r) {
  r.transition = r.transition || {};
  const t = r.transition;
  const sec = el("div", { class: "form-section" }, [el("h3", {}, [el("span", { text: "Transition" })])]);
  sec.appendChild(field("type", selectInput(t, "type", TRANSITION_TYPES), "engine treats all as smoothstep"));
  sec.appendChild(field("falloff", numInput(t, "falloff")));
  sec.appendChild(field("cutoff_epsilon", numInput(t, "cutoff_epsilon")));
  return sec;
}

// ---- influences -----------------------------------------------------------
function renderInfluences(r) {
  if (!Array.isArray(r.influences)) r.influences = [];
  const sec = el("div", { class: "form-section" });
  sec.appendChild(el("h3", {}, [
    el("span", { text: "Influences" }),
    el("button", { text: "+ influence", onclick: () => { r.influences.push({ field: "height", mode: "add", value: 0 }); renderEditor(); touch(); } }),
  ]));
  r.influences.forEach((inf, i) => sec.appendChild(renderInfluenceCard(r, inf, i)));
  if (!r.influences.length) sec.appendChild(el("div", { class: "hint", text: "no influences" }));
  return sec;
}

function renderInfluenceCard(r, inf, i) {
  const card = el("div", { class: "card" });
  const fieldLabel = (INFLUENCE_FIELDS.find(f => f.v === inf.field) || {}).label || inf.field || "?";
  card.appendChild(el("div", { class: "card-head" }, [
    el("span", { class: "title", text: "#" + i + "  " + fieldLabel }),
    el("div", { class: "ctrls" }, [
      el("button", { text: "↑", onclick: () => { if (i > 0) { [r.influences[i - 1], r.influences[i]] = [r.influences[i], r.influences[i - 1]]; renderEditor(); touch(); } } }),
      el("button", { text: "↓", onclick: () => { if (i < r.influences.length - 1) { [r.influences[i + 1], r.influences[i]] = [r.influences[i], r.influences[i + 1]]; renderEditor(); touch(); } } }),
      el("button", { text: "⧉", title: "duplicate", onclick: () => { r.influences.splice(i + 1, 0, JSON.parse(JSON.stringify(inf))); renderEditor(); touch(); } }),
      el("button", { text: "✕", onclick: () => { r.influences.splice(i, 1); renderEditor(); touch(); } }),
    ]),
  ]));

  card.appendChild(field("field", selectInput(inf, "field", INFLUENCE_FIELDS, () => { renderEditor(); })));
  card.appendChild(field("mode", selectInput(inf, "mode", BLEND_MODES, () => { renderEditor(); })));

  if (inf.field === "octave_amplitude") {
    const opts = octaveNames().map(n => ({ v: n, label: n }));
    if (inf.octave && !opts.find(o => o.v === inf.octave)) opts.unshift({ v: inf.octave, label: inf.octave });
    card.appendChild(field("octave", selectInput(inf, "octave", opts), hasNamedOctaves() ? "" : "legacy/fallback octave"));
  } else {
    delete inf.octave;
  }

  const mode = inf.mode;
  if (mode === "pull_to_target") {
    card.appendChild(field("target", numInput(inf, "target")));
    card.appendChild(field("strength", sliderInput(inf, "strength", 0, 1, 0.01)));
    delete inf.value;
  } else if (mode === "pull_to_path_grade") {
    card.appendChild(field("strength", sliderInput(inf, "strength", 0, 1, 0.01)));
    delete inf.value; delete inf.target;
  } else {
    card.appendChild(field("value", numInput(inf, "value")));
    delete inf.target; delete inf.strength;
  }

  if (inf.field === "bank_steepness") card.appendChild(el("div", { class: "warn-inline", text: "bank_steepness is preserved but not supported by the engine." }));
  if (mode === "clamp_min" || mode === "clamp_max") card.appendChild(el("div", { class: "warn-inline", text: mode + " is preserved but not supported by the engine." }));
  return card;
}

// ---- validation panel -----------------------------------------------------
function renderValidation() {
  const root = $("validation");
  clear(root);
  const { errors, warnings } = state.validation;
  if (!errors.length && !warnings.length) {
    root.appendChild(el("div", { class: "v-empty", text: "✓ valid — no issues" }));
    return;
  }
  errors.forEach(e => root.appendChild(vItem(e, "error")));
  warnings.forEach(w => root.appendChild(vItem(w, "warn")));
}
function vItem(item, kind) {
  return el("div", { class: "v-item " + kind, onclick: () => jumpTo(item.path) }, [
    el("div", { text: item.message }),
    item.path ? el("div", { class: "vp", text: item.path }) : null,
  ]);
}
function jumpTo(path) {
  const m = /^regions\[(\d+)\]/.exec(path || "");
  if (m) select(parseInt(m[1], 10));
  else if (path && path.startsWith("scene")) select(-1);
}

// ---- save / revert / load -------------------------------------------------
async function doSave() {
  if (!state.doc) return;
  const res = await api.save(state.path, state.doc);
  if (res.saved) {
    state.pristine = JSON.stringify(state.doc);
    state.dirty = false;
    state.lastSaved = new Date().toLocaleTimeString();
    state.validation = { errors: [], warnings: res.warnings || [] };
    if (res.raw) $("yaml-preview").textContent = res.raw;
    renderValidation(); renderTree(); updateChrome();
    toast("saved → engine will hot-reload", "good");
  } else {
    state.validation = { errors: res.errors || [], warnings: res.warnings || [] };
    renderValidation(); renderTree(); updateChrome();
    toast(res.message || "save refused — fix validation errors", "bad");
  }
}
function doRevert() {
  if (!state.pristine) return;
  if (state.dirty && !confirm("Discard unsaved changes?")) return;
  state.doc = JSON.parse(state.pristine);
  state.dirty = false;
  renderTree(); renderEditor();
  touch();
  toast("reverted", "");
}
async function loadScene(path) {
  const res = await api.scene(path);
  if (res.error) { toast(res.error, "bad"); return; }
  state.path = path;
  state.doc = res.document || {};
  state.pristine = JSON.stringify(state.doc);
  state.dirty = false;
  state.selected = -1;
  state.lastSaved = null;
  state.validation = { errors: res.errors || [], warnings: res.warnings || [] };
  $("scene-name").textContent = path;
  $("yaml-preview").textContent = res.raw || "";
  renderTree(); renderEditor(); renderValidation(); updateChrome();
}

// ---- init -----------------------------------------------------------------
async function init() {
  renderTemplates();
  $("btn-save").addEventListener("click", doSave);
  $("btn-revert").addEventListener("click", doRevert);
  $("btn-add-root").addEventListener("click", () => { if (state.doc) addRoot(); });
  $("btn-copy").addEventListener("click", () => { navigator.clipboard.writeText($("yaml-preview").textContent); toast("copied", ""); });

  document.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "s") { e.preventDefault(); doSave(); }
    if (e.key === "Escape" && document.activeElement && document.activeElement.blur) document.activeElement.blur();
  });
  window.addEventListener("beforeunload", (e) => { if (state.dirty) { e.preventDefault(); e.returnValue = ""; } });

  const picker = $("scene-picker");
  try {
    state.scenes = await api.scenes();
  } catch (e) { toast("cannot reach server", "bad"); return; }
  clear(picker);
  picker.appendChild(el("option", { value: "", text: state.scenes.length ? "— choose a scene —" : "(no scenes found)" }));
  state.scenes.forEach(s => picker.appendChild(el("option", { value: s, text: s })));
  picker.addEventListener("change", () => {
    if (!picker.value) return;
    if (state.dirty && !confirm("Discard unsaved changes and switch scene?")) { picker.value = state.path || ""; return; }
    loadScene(picker.value);
  });
  renderEditor();
}

init();
