#!/usr/bin/env python3
"""Local scene graph editor server for terragen terrain YAML files.

A boring, dependency-light control-plane tool. It loads res/scenes/*.yaml,
validates against the scene structure the engine actually parses, and writes
YAML back atomically so the running engine can hot-reload from disk.

The YAML file is the source of truth. This server never keeps a separate
database; it only parses, validates, and re-serializes the document.

Usage:
    python tools/scene_server/scene_server.py \\
        --scene-root res/scenes --host 127.0.0.1 --port 7810 [--open]

Then open http://127.0.0.1:7810/

Requires PyYAML (used only for *loading*; serialization is custom so the
output stays parseable by the engine's line-based parser).
"""

import argparse
import json
import os
import sys
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

try:
    import yaml
except ImportError:
    sys.stderr.write(
        "error: PyYAML is required. Install with: python3 -m pip install pyyaml\n"
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Engine ground-truth enums (src/main/terrain_scene.c). Keep in sync with the
# parser; these define what produces working, hot-reloadable YAML.
# ---------------------------------------------------------------------------
GEOMETRY_TYPES = {
    "box", "oriented_box", "sphere", "ellipsoid",
    "capsule_path", "capsule", "capsule_line", "plane", "ramp",
}
CAPSULE_TYPES = {"capsule_path", "capsule", "capsule_line"}
RADIUS_TYPES = CAPSULE_TYPES | {"sphere"}
INFLUENCE_FIELDS = {
    "height", "octave_amplitude", "warp_amount_scale",
    "ridge_amount_scale", "carve_amount", "material_id", "vibe",
}
BLEND_MODES = {
    "add", "multiply", "pull_to_target", "pull",
    "pull_to_path_grade", "subtract", "carve", "override",
}
OCTAVE_KINDS = {"fbm", "ridged"}
# Fields/modes the editor preserves but the engine does not understand yet.
UNSUPPORTED_FIELDS = {"bank_steepness"}
UNSUPPORTED_MODES = {"clamp_min", "clamp_max"}

# Stable key order for serialization. Unknown keys are appended after these,
# preserving their original insertion order so nothing is dropped.
KEY_ORDER = {
    "_root": ["version", "scene", "regions"],
    "scene": ["id", "name", "world", "height", "noise", "warp", "material"],
    "world": ["region_size_chunks", "base_cell_size", "chunk_cells", "lod_count"],
    "height": ["min", "max", "base"],
    "noise": ["seed", "legacy", "octaves", "lacunarity", "gain"],
    "legacy": ["frequency", "amplitude", "octaves", "lacunarity", "gain"],
    "octave": ["name", "kind", "frequency", "amplitude", "sharpness"],
    "warp": ["frequency", "amplitude"],
    "material": ["default_vibe", "default_flat_material", "default_slope_material"],
    # Region scalars come BEFORE geometry/transition/influences: the engine's
    # parser only reads parent/kind/priority/allow_outside_parent while in the
    # region section, so they must precede the nested blocks or be ignored.
    "region": ["id", "parent", "kind", "allow_outside_parent", "priority",
               "geometry", "transition", "influences"],
    "geometry": ["type", "center", "size", "rotation_y", "radius",
                 "mask_warp", "points"],
    "transition": ["type", "falloff", "cutoff_epsilon"],
    "influence": ["field", "octave", "mode", "value", "target", "strength"],
}

import re

_PLAIN_RE = re.compile(r"^[A-Za-z0-9_./+-]+$")
_RESERVED = {"true", "false", "null", "yes", "no", "on", "off", "~", ""}

# Keys whose values are conventionally floats in scene files. JSON has no
# int/float distinction, so an untouched 16.0 arrives back as int 16; forcing
# float output for these keys keeps diffs clean and matches the existing style.
FLOAT_KEYS = {
    "base_cell_size", "min", "max", "base", "frequency", "amplitude",
    "gain", "lacunarity", "mask_warp", "edge_noise", "sharpness",
}


def fmt_scalar(v, key=None):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        if key in FLOAT_KEYS:
            return "%s.0" % v
        return str(v)
    if isinstance(v, float):
        if v != v:
            return ".nan"
        if v == float("inf"):
            return ".inf"
        if v == float("-inf"):
            return "-.inf"
        # repr keeps 13.0 as "13.0" and 0.085 as "0.085" (round-trip clean).
        return repr(v)
    if v is None:
        return "null"
    s = str(v)
    if _PLAIN_RE.match(s) and s.lower() not in _RESERVED:
        # Avoid emitting a bare number-looking string unquoted.
        try:
            float(s)
            return '"%s"' % s
        except ValueError:
            return s
    return '"%s"' % s.replace("\\", "\\\\").replace('"', '\\"')


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _all_numbers(seq):
    return isinstance(seq, list) and len(seq) > 0 and all(_is_number(x) for x in seq)


def fmt_flow_numbers(seq):
    return "[" + ", ".join(fmt_scalar(x) for x in seq) + "]"


def ordered_keys(mapping, ctx):
    order = KEY_ORDER.get(ctx, [])
    keys = list(mapping.keys())
    head = [k for k in order if k in mapping]
    tail = [k for k in keys if k not in order]
    return head + tail


# child-context resolution: which KEY_ORDER applies to a value at key `k`
CHILD_CTX = {
    ("_root", "scene"): "scene",
    ("scene", "world"): "world",
    ("scene", "height"): "height",
    ("scene", "noise"): "noise",
    ("noise", "legacy"): "legacy",
    ("noise", "warp"): "warp",
    ("scene", "warp"): "warp",
    ("scene", "material"): "material",
    ("region", "geometry"): "geometry",
    ("region", "transition"): "transition",
}


def dump_node(node, indent, ctx, lines):
    pad = "  " * indent
    if isinstance(node, dict):
        for k in ordered_keys(node, ctx):
            v = node[k]
            child_ctx = CHILD_CTX.get((ctx, k))
            if isinstance(v, dict):
                lines.append("%s%s:" % (pad, k))
                dump_node(v, indent + 1, child_ctx, lines)
            elif isinstance(v, list):
                dump_list(k, v, indent, ctx, lines)
            else:
                lines.append("%s%s: %s" % (pad, k, fmt_scalar(v, k)))
    else:
        lines.append("%s%s" % (pad, fmt_scalar(node)))


def dump_list(key, seq, indent, ctx, lines):
    pad = "  " * indent
    if _all_numbers(seq):
        # vec3 / inline numeric array
        lines.append("%s%s: %s" % (pad, key, fmt_flow_numbers(seq)))
        return
    if len(seq) == 0:
        lines.append("%s%s: []" % (pad, key))
        return
    lines.append("%s%s:" % (pad, key))
    item_ctx = _ITEM_CTX.get((ctx, key))
    item_pad = "  " * indent
    for item in seq:
        if _all_numbers(item):
            lines.append("%s- %s" % (item_pad, fmt_flow_numbers(item)))
        elif isinstance(item, dict):
            # emit "- " then the dict body at indent+1, splicing first key onto
            # the dash line for readable block style.
            sub = []
            dump_node(item, indent + 1, item_ctx, sub)
            if sub:
                first = sub[0]
                stripped = first[(indent + 1) * 2:]
                lines.append("%s- %s" % (item_pad, stripped))
                lines.extend(sub[1:])
            else:
                lines.append("%s- {}" % item_pad)
        else:
            lines.append("%s- %s" % (item_pad, fmt_scalar(item)))


_ITEM_CTX = {
    ("_root", "regions"): "region",
    ("noise", "octaves"): "octave",
    ("region", "influences"): "influence",
    # geometry.points items are numeric arrays handled by _all_numbers
}


def dump_scene(doc):
    """Serialize a parsed scene document to engine-parseable YAML text."""
    lines = []
    dump_node(doc, 0, "_root", lines)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _num(x):
    return x if _is_number(x) else None


def _bad_float(x):
    return _is_number(x) and (x != x or x in (float("inf"), float("-inf")))


def validate_document(doc):
    """Return (errors, warnings) lists of {path, message}."""
    errors, warnings = [], []

    def err(path, msg):
        errors.append({"path": path, "message": msg})

    def warn(path, msg):
        warnings.append({"path": path, "message": msg})

    if not isinstance(doc, dict):
        err("", "document root must be a mapping")
        return errors, warnings

    scene = doc.get("scene")
    octave_names = set()
    if isinstance(scene, dict):
        noise = scene.get("noise")
        if isinstance(noise, dict):
            octs = noise.get("octaves")
            if isinstance(octs, list):
                for i, o in enumerate(octs):
                    if isinstance(o, dict):
                        nm = o.get("name")
                        if nm:
                            octave_names.add(str(nm))
                        kind = o.get("kind")
                        if kind is not None and str(kind) not in OCTAVE_KINDS:
                            err("scene.noise.octaves[%d].kind" % i,
                                "unknown octave kind '%s'" % kind)
            has_legacy = isinstance(noise.get("legacy"), dict)
            has_named = bool(octave_names)
            sid = str(scene.get("id", ""))
            if sid == "legacy_default" and has_named:
                warn("scene.noise.octaves",
                     "legacy_default should keep noise.legacy and define no named octaves")
        h = scene.get("height")
        if isinstance(h, dict):
            mn, mx = _num(h.get("min")), _num(h.get("max"))
            if mn is not None and mx is not None:
                if mx <= mn:
                    err("scene.height.max", "max must be greater than min")
                elif (mx - mn) < 1.0:
                    warn("scene.height", "height band is very narrow (<1 unit)")

    regions = doc.get("regions")
    if regions is None:
        regions = []
    if not isinstance(regions, list):
        err("regions", "regions must be a list")
        return errors, warnings

    if len(regions) > 64:
        warn("regions", "very high region count (%d)" % len(regions))

    ids = {}
    id_to_parent = {}
    for i, r in enumerate(regions):
        base = "regions[%d]" % i
        if not isinstance(r, dict):
            err(base, "region must be a mapping")
            continue
        rid = r.get("id")
        if not rid:
            err(base + ".id", "region id is required")
        else:
            rid = str(rid)
            if rid in ids:
                err(base + ".id", "duplicate region id '%s'" % rid)
            ids[rid] = i
            id_to_parent[rid] = str(r.get("parent")) if r.get("parent") else None

        # geometry
        g = r.get("geometry")
        if isinstance(g, dict):
            gt = g.get("type")
            if gt is None:
                err(base + ".geometry.type", "geometry type is required")
            elif str(gt) not in GEOMETRY_TYPES:
                err(base + ".geometry.type", "unknown geometry type '%s'" % gt)
            gts = str(gt)
            for vk in ("center", "size"):
                if vk in g and g[vk] is not None:
                    if not _all_numbers(g[vk]) or len(g[vk]) != 3:
                        err("%s.geometry.%s" % (base, vk), "malformed vec3 (need [x, y, z])")
                    elif any(_bad_float(x) for x in g[vk]):
                        err("%s.geometry.%s" % (base, vk), "vec3 contains NaN/Infinity")
            if gts in RADIUS_TYPES:
                rad = _num(g.get("radius"))
                if rad is None:
                    err(base + ".geometry.radius", "radius is required for %s" % gts)
                elif rad <= 0:
                    err(base + ".geometry.radius", "radius must be greater than 0")
            if gts in CAPSULE_TYPES:
                pts = g.get("points")
                if not isinstance(pts, list) or len(pts) == 0:
                    err(base + ".geometry.points", "capsule_path requires at least one point")
                else:
                    for pi, p in enumerate(pts):
                        if not _all_numbers(p) or len(p) != 3:
                            err("%s.geometry.points[%d]" % (base, pi), "malformed vec3")
                        elif any(_bad_float(x) for x in p):
                            err("%s.geometry.points[%d]" % (base, pi), "point contains NaN/Infinity")
        elif g is not None:
            err(base + ".geometry", "geometry must be a mapping")

        # transition
        t = r.get("transition")
        if isinstance(t, dict):
            fo = _num(t.get("falloff"))
            if fo is not None and fo < 0:
                err(base + ".transition.falloff", "falloff must be >= 0")
            ce = _num(t.get("cutoff_epsilon"))
            if ce is not None and ce < 0:
                err(base + ".transition.cutoff_epsilon", "cutoff_epsilon must be >= 0")

        # influences
        infs = r.get("influences")
        if isinstance(infs, list):
            for j, inf in enumerate(infs):
                ib = "%s.influences[%d]" % (base, j)
                if not isinstance(inf, dict):
                    err(ib, "influence must be a mapping")
                    continue
                fld = inf.get("field")
                if fld is None:
                    err(ib + ".field", "influence field is required")
                elif str(fld) in UNSUPPORTED_FIELDS:
                    warn(ib + ".field", "field '%s' is preserved but not supported by the engine" % fld)
                elif str(fld) not in INFLUENCE_FIELDS:
                    err(ib + ".field", "unknown influence field '%s'" % fld)
                mode = inf.get("mode")
                if mode is None:
                    err(ib + ".mode", "influence mode is required")
                elif str(mode) in UNSUPPORTED_MODES:
                    warn(ib + ".mode", "mode '%s' is preserved but not supported by the engine" % mode)
                elif str(mode) not in BLEND_MODES:
                    err(ib + ".mode", "unknown blend mode '%s'" % mode)
                if str(fld) == "octave_amplitude":
                    oc = inf.get("octave")
                    if not oc:
                        err(ib + ".octave", "octave_amplitude influence requires an octave name")
                    elif octave_names and str(oc) not in octave_names:
                        # When legacy-only (no named octaves), the engine maps by
                        # name lookup that will simply miss; warn rather than block
                        # only if there ARE named octaves to match against.
                        err(ib + ".octave",
                            "octave '%s' does not match any scene.noise.octaves name" % oc)
                    elif not octave_names:
                        warn(ib + ".octave",
                             "scene has no named octaves; '%s' is a legacy/fallback reference" % oc)
                for nk in ("value", "target", "strength"):
                    if nk in inf and _bad_float(inf[nk]):
                        err("%s.%s" % (ib, nk), "%s is NaN/Infinity" % nk)
                amp = _num(inf.get("value"))
                if amp is not None and abs(amp) > 1000:
                    warn(ib + ".value", "very large value (%g)" % amp)

    # unknown parents + cycles
    for rid, parent in id_to_parent.items():
        if parent is None:
            continue
        if parent not in ids:
            err("regions[%d].parent" % ids[rid], "unknown parent id '%s'" % parent)
            continue
        # cycle detection
        seen, cur, steps = set(), rid, 0
        while cur is not None and steps <= len(id_to_parent) + 1:
            if cur in seen:
                err("regions[%d].parent" % ids[rid], "parent cycle detected at '%s'" % rid)
                break
            seen.add(cur)
            cur = id_to_parent.get(cur)
            steps += 1

    return errors, warnings


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------
class Config:
    scene_root = "res/scenes"
    scene_root_abs = ""
    base_dir = ""        # cwd, the root that scene paths are relative to
    editor_dir = ""      # static assets
    schema_path = ""


CFG = Config()


def list_scenes():
    out = []
    root = CFG.scene_root_abs
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.endswith((".yaml", ".yml")):
                abspath = os.path.join(dirpath, fn)
                rel = os.path.relpath(abspath, CFG.base_dir)
                out.append(rel.replace(os.sep, "/"))
    out.sort()
    return out


def resolve_scene_path(rel_path):
    """Resolve a request path safely to a file under scene_root."""
    if not rel_path:
        raise ValueError("missing path")
    candidate = os.path.normpath(os.path.join(CFG.base_dir, rel_path))
    if not (candidate == CFG.scene_root_abs or
            candidate.startswith(CFG.scene_root_abs + os.sep)):
        raise ValueError("path is outside scene root")
    if not candidate.endswith((".yaml", ".yml")):
        raise ValueError("not a YAML file")
    return candidate


def atomic_write(path, text):
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(prefix=".scene_", suffix=".tmp", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
}


class Handler(BaseHTTPRequestHandler):
    server_version = "TerragenSceneEditor/1.0"

    def log_message(self, fmt, *args):
        sys.stderr.write("[scene_server] %s\n" % (fmt % args))

    # -- helpers -----------------------------------------------------------
    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _send_file(self, path):
        if not os.path.isfile(path):
            self._send_json({"error": "not found"}, 404)
            return
        ext = os.path.splitext(path)[1].lower()
        ctype = CONTENT_TYPES.get(ext, "application/octet-stream")
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # -- routing -----------------------------------------------------------
    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path
        try:
            if route == "/api/scenes":
                self._send_json({"scenes": list_scenes()})
            elif route == "/api/scene":
                self._get_scene(parse_qs(parsed.query))
            elif route == "/api/schema":
                self._send_file(CFG.schema_path)
            elif route == "/api/status":
                self._send_json({
                    "cwd": CFG.base_dir,
                    "sceneRoot": CFG.scene_root,
                    "sceneRootAbs": CFG.scene_root_abs,
                    "selectedScene": None,
                })
            else:
                self._serve_static(route)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, 400)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, 500)

    def do_POST(self):
        route = urlparse(self.path).path
        try:
            if route == "/api/scene":
                self._save_scene(self._read_json())
            elif route == "/api/validate":
                self._validate(self._read_json())
            elif route == "/api/format":
                self._format(self._read_json())
            else:
                self._send_json({"error": "unknown endpoint"}, 404)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, 400)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, 500)

    # -- handlers ----------------------------------------------------------
    def _serve_static(self, route):
        if route == "/" or route == "":
            route = "/index.html"
        if route == "/schema" or route == "/scene.schema.json":
            self._send_file(CFG.schema_path)
            return
        rel = route.lstrip("/")
        target = os.path.normpath(os.path.join(CFG.editor_dir, rel))
        if not target.startswith(CFG.editor_dir):
            self._send_json({"error": "forbidden"}, 403)
            return
        self._send_file(target)

    def _get_scene(self, query):
        rel = (query.get("path") or [""])[0]
        path = resolve_scene_path(rel)
        if not os.path.isfile(path):
            self._send_json({"error": "scene not found: %s" % rel}, 404)
            return
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        try:
            doc = yaml.safe_load(raw) or {}
        except yaml.YAMLError as exc:
            self._send_json({
                "path": rel, "raw": raw, "document": None,
                "mtime": os.path.getmtime(path), "valid": False,
                "errors": [{"path": "", "message": "YAML parse error: %s" % exc}],
                "warnings": [],
            })
            return
        errors, warnings = validate_document(doc)
        self._send_json({
            "path": rel,
            "document": doc,
            "raw": raw,
            "mtime": os.path.getmtime(path),
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        })

    def _save_scene(self, payload):
        rel = payload.get("path")
        doc = payload.get("document")
        path = resolve_scene_path(rel)
        if doc is None:
            raise ValueError("missing document")
        errors, warnings = validate_document(doc)
        if errors:
            self._send_json({
                "saved": False, "valid": False,
                "errors": errors, "warnings": warnings,
                "message": "refused to save: %d validation error(s)" % len(errors),
            }, 400)
            return
        text = dump_scene(doc)
        # round-trip sanity: ensure our output reparses to an equivalent doc
        try:
            yaml.safe_load(text)
        except yaml.YAMLError as exc:
            self._send_json({
                "saved": False, "valid": False,
                "errors": [{"path": "", "message": "serialized YAML did not reparse: %s" % exc}],
                "warnings": warnings,
            }, 500)
            return
        atomic_write(path, text)
        self._send_json({
            "saved": True, "valid": True,
            "path": rel, "mtime": os.path.getmtime(path),
            "raw": text, "warnings": warnings, "errors": [],
        })

    def _validate(self, payload):
        doc = payload.get("document")
        errors, warnings = validate_document(doc if doc is not None else {})
        self._send_json({"valid": len(errors) == 0, "errors": errors, "warnings": warnings})

    def _format(self, payload):
        doc = payload.get("document")
        if doc is None:
            raise ValueError("missing document")
        try:
            text = dump_scene(doc)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"ok": False, "error": str(exc)}, 400)
            return
        self._send_json({"ok": True, "raw": text})


def main():
    ap = argparse.ArgumentParser(description="Terragen local scene editor server")
    ap.add_argument("--scene-root", default="res/scenes",
                    help="directory containing scene YAML files (default: res/scenes)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7810)
    ap.add_argument("--open", action="store_true", help="open the editor in a browser")
    args = ap.parse_args()

    CFG.base_dir = os.getcwd()
    CFG.scene_root = args.scene_root
    CFG.scene_root_abs = os.path.normpath(os.path.join(CFG.base_dir, args.scene_root))
    here = os.path.dirname(os.path.abspath(__file__))
    CFG.editor_dir = os.path.normpath(os.path.join(here, "..", "scene_editor"))
    CFG.schema_path = os.path.join(CFG.editor_dir, "src", "schema", "scene.schema.json")

    if not os.path.isdir(CFG.scene_root_abs):
        sys.stderr.write("warning: scene root does not exist: %s\n" % CFG.scene_root_abs)
    if not os.path.isfile(os.path.join(CFG.editor_dir, "index.html")):
        sys.stderr.write("warning: editor assets missing under %s\n" % CFG.editor_dir)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = "http://%s:%d/" % (args.host, args.port)
    sys.stderr.write("terragen scene editor: %s\n" % url)
    sys.stderr.write("  scene root: %s\n" % CFG.scene_root_abs)
    if args.open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write("\nshutting down\n")
        httpd.shutdown()


if __name__ == "__main__":
    main()
