# Terragen Scene Editor

A local, no-build web tool for editing terrain scene YAML (`res/scenes/*.yaml`).
It is a structured manipulator for the YAML files the engine already
hot-reloads — **the YAML on disk stays the source of truth**. There is no
separate editor database.

```
browser editor  ↕ HTTP  scene_server.py  ↕ filesystem  res/scenes/*.yaml  ↕ hot reload  terragen
```

## Run

Requires Python 3 with PyYAML (already present on this machine). No `npm install`,
no build step — the server serves the static editor directly.

```bash
python tools/scene_server/scene_server.py \
  --scene-root res/scenes --host 127.0.0.1 --port 7810
# add --open to launch the browser automatically
```

Then open <http://127.0.0.1:7810/>.

Run it from the repository root so scene paths (`res/scenes/...`) resolve.

## Workflow

1. Pick a scene from the dropdown (left pane).
2. Click **scene settings** or any region in the tree to edit it (middle pane).
3. Use the **Templates** buttons to add common nodes as children of the
   selected region (mountain root, valley, plateau, path, riverbed, peak, cave).
4. Watch the validation panel and YAML preview (right pane) update live.
5. Click **Save** or press **Ctrl/⌘+S**. Sliders and inputs only change local
   state — nothing is written until you save. On save the file is written
   atomically (temp file + rename) and terragen picks up the new mtime.

## What it knows about the engine

The schema and validator are grounded in the engine's parser
(`src/main/terrain_scene.c`), so saved YAML stays loadable:

- **Geometry**: `box`, `oriented_box`, `sphere`, `ellipsoid`, `capsule_path`
  (aliases `capsule`/`capsule_line`), `plane`/`ramp`.
- **Influence fields**: `height`, `octave_amplitude`, `warp_amount_scale`,
  `ridge_amount_scale`, `carve_amount`, `material_id`/`vibe`. Friendly UI labels
  map to these engine names. `bank_steepness` is offered but flagged unsupported.
- **Blend modes**: `add`, `multiply`, `pull_to_target`, `pull_to_path_grade`,
  `subtract`, `override`. `clamp_min`/`clamp_max` are offered but flagged unsupported.
- **Octave kinds**: `fbm`, `ridged`.

### Serialization notes (important)

- `vec3` values and capsule points are emitted inline (`[x, y, z]`) because the
  engine parser requires the `[` form — block-style sequences would break it.
- Region scalars (`parent`, `kind`, `priority`, `allow_outside_parent`) are
  emitted **before** `geometry:`. The engine's line-based parser only reads
  these while in the region section, so placing `priority:` after the
  `transition:` block (as the original hand-written files do) causes it to be
  silently ignored. The editor's ordering makes `priority` actually take effect.
- `transition.type` is preserved for readability but the engine currently treats
  every transition as `smoothstep` (only `falloff` / `cutoff_epsilon` are used).
- Unknown fields round-trip: anything the editor doesn't recognize is kept in
  the document object and re-serialized, never dropped.

## Endpoints

| Method | Path            | Purpose                                  |
|--------|-----------------|------------------------------------------|
| GET    | `/api/scenes`   | list scene files under the scene root    |
| GET    | `/api/scene`    | load + parse + validate one scene        |
| POST   | `/api/scene`    | validate then atomically save            |
| GET    | `/api/schema`   | JSON Schema for the scene structure      |
| POST   | `/api/validate` | validate a document without saving       |
| POST   | `/api/format`   | serialize a document to YAML (preview)   |
| GET    | `/api/status`   | cwd / scene root                         |

## Layout

```
tools/scene_server/scene_server.py   # stdlib HTTP server: API + validator + YAML dumper
tools/scene_editor/
  index.html                         # three-pane shell
  styles.css
  app.js                             # the editor (vanilla JS, no build)
  src/schema/scene.schema.json       # scene JSON Schema (served at /api/schema)
```

> Implementation note: the task suggested a React/Vite + TypeScript app. A
> no-build vanilla-JS app served by the Python server was chosen instead because
> it is the fastest, least-invasive way to satisfy the requirements — one
> command, no toolchain install — matching the "boring local tool" goal. All API
> contracts, the schema, and the widget set from the brief are implemented.
```
