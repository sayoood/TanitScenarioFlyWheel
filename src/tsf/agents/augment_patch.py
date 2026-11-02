"""Restricted JSON-Patch (RFC 6902: add / replace / remove) over a TSIR document, the per-dimension whitelists
the editor may touch, the structural diff vs the parent and the isolation check.

The scenario augmenter (``tsf.agents.augment``) varies a verified parent along three dimensions:

    weather      D1  /environment/*  (weather, time of day, precipitation, fog, cloud cover, road surface)
    behavior     D2  actor speeds / positions / routes, actions (timed behaviour), additional NPCs
    environment  D3  road geometry and lanes, static objects and signs, environment.setting

Isolation = the structural diff between parent and variant may only touch the whitelist of the requested
dimension(s) (union in manual combined mode). ``/meta`` and ``/parameters`` are code-owned housekeeping and are
exempt (the editor can never write them: its path grammar does not allow it).
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field

DIMS = ("weather", "behavior", "environment")          # D1, D2, D3
DIM_ID = {"weather": "D1", "behavior": "D2", "environment": "D3"}
DIM_TITLE = {"weather": "weather & time of day", "behavior": "behaviour", "environment": "environment"}
_TOK = r"[A-Za-z0-9_~-]+"
_ENV_WEATHER = r"/environment/(?:time_of_day|weather|precipitation|fog_visibility_m|cloud_cover|road_surface|sun_azimuth_deg)"
_ACTOR_INIT = r"/actors/[0-9]+/initial(?:/(?:road|direction|lane|s|t_offset))?"
_ASSERT_TUNE = r"/assertions/[0-9]+/(?:value|value2)"

# Pointer grammar the editor may WRITE (op level; "-" appends). Full match, no anchors needed.
WRITE: dict[str, dict[str, list[str]]] = {
    "weather": {"primary": [_ENV_WEATHER], "secondary": []},
    "behavior": {
        "primary": [r"/actors/[0-9]+/(?:initial_speed|route|headway)", _ACTOR_INIT, r"/actors/-",
                    rf"/actions(?:/{_TOK})*"],
        # assertion thresholds may be retuned when the new behaviour changes what is critical (never removed);
        # the run duration may grow when the new behaviour needs more time
        "secondary": [_ASSERT_TUNE, r"/duration"]},
    "environment": {
        "primary": [rf"/road(?:/{_TOK})*", rf"/static_objects(?:/{_TOK})*", r"/environment/setting"],
        "secondary": [_ACTOR_INIT, r"/duration"]},
}

# Pointer paths of the structural DIFF (what actually changed); an appended actor shows up as /actors/<n>.
ALLOW: dict[str, dict[str, list[str]]] = {
    "weather": WRITE["weather"],
    "behavior": {"primary": [r"/actors/[0-9]+/(?:initial_speed|route|headway)", _ACTOR_INIT,
                             rf"/actions(?:/{_TOK})*"],
                 "secondary": WRITE["behavior"]["secondary"]},
    "environment": WRITE["environment"],
}
HOUSEKEEPING = [rf"/meta(?:/{_TOK})*", rf"/parameters(?:/{_TOK})*"]


class PatchError(ValueError):
    """The patch could not be applied (bad pointer, index out of range, missing target)."""


# ----------------------------------------------------------------------------- JSON pointer + patch
def tokens(ptr: str) -> list[str]:
    if ptr == "":
        return []
    if not ptr.startswith("/"):
        raise PatchError(f"pointer {ptr!r} must start with '/'")
    return [t.replace("~1", "/").replace("~0", "~") for t in ptr[1:].split("/")]


def pointer(toks: list) -> str:
    return "".join("/" + str(t).replace("~", "~0").replace("/", "~1") for t in toks)


def _walk(doc, toks: list[str], ptr: str):
    cur = doc
    for i, t in enumerate(toks):
        try:
            cur = cur[int(t)] if isinstance(cur, list) else cur[t]
        except (KeyError, IndexError, ValueError, TypeError):
            keys = (f"; keys there: {', '.join(map(str, list(cur)[:14]))}" if isinstance(cur, dict) else
                    f"; the list has {len(cur)} items" if isinstance(cur, list) else "")
            raise PatchError(f"path {pointer(toks[:i + 1])} does not exist (while resolving {ptr}){keys}") from None
    return cur


def apply_patch(doc: dict, ops: list[dict]) -> dict:
    """Apply RFC 6902 ``add`` / ``replace`` / ``remove`` operations in order to a deep copy of ``doc``."""
    out = copy.deepcopy(doc)
    for n, op in enumerate(ops):
        kind, ptr = op.get("op"), op.get("path", "")
        if kind not in ("add", "replace", "remove"):
            raise PatchError(f"op {n}: unsupported op {kind!r} (only add, replace, remove)")
        toks = tokens(ptr)
        if not toks:
            raise PatchError(f"op {n}: the document root cannot be patched")
        if kind != "remove" and "value" not in op:
            raise PatchError(f"op {n}: {kind} {ptr} needs a value")
        val = copy.deepcopy(op.get("value"))
        parent, last = _walk(out, toks[:-1], ptr), toks[-1]
        if isinstance(parent, list):
            if kind == "add":
                if last == "-":
                    parent.append(val)
                    continue
                idx = _index(last, len(parent) + 1, n, ptr)
                parent.insert(idx, val)
            else:
                if last == "-":
                    raise PatchError(f"op {n}: '-' is only valid for add")
                idx = _index(last, len(parent), n, ptr)
                if kind == "replace":
                    parent[idx] = val
                else:
                    del parent[idx]
        elif isinstance(parent, dict):
            if kind == "add":
                parent[last] = val
            elif last not in parent:
                where = pointer(toks[:-1]) or "/"
                raise PatchError(f"op {n}: {kind} {ptr}: key {last!r} does not exist at {where}; keys there: "
                                 f"{', '.join(map(str, list(parent)[:14]))}")
            elif kind == "replace":
                parent[last] = val
            else:
                del parent[last]
        else:
            raise PatchError(f"op {n}: {ptr}: parent is a scalar")
    return out


def _index(tok: str, bound: int, n: int, ptr: str) -> int:
    try:
        i = int(tok)
    except ValueError:
        raise PatchError(f"op {n}: {ptr}: {tok!r} is not an array index") from None
    if not 0 <= i < bound:
        raise PatchError(f"op {n}: {ptr}: index {i} out of range (array has {bound - (1 if bound else 0)} entries "
                         f"at that point; ops are applied in order, earlier add/remove ops shift indices)")
    return i


# ----------------------------------------------------------------------------- structural diff
@dataclass
class Change:
    path: str
    op: str                     # add | replace | remove
    before: object = None
    after: object = None

    def to_dict(self) -> dict:
        d = {"path": self.path, "op": self.op}
        if self.op != "add":
            d["before"] = _short(self.before)
        if self.op != "remove":
            d["after"] = _short(self.after)
        return d


def _short(v, n: int = 300):
    s = repr(v) if not isinstance(v, (int, float, str, bool, type(None))) else v
    return s if not isinstance(s, str) or len(s) <= n else s[:n] + "..."


def _id_list(xs) -> bool:
    return bool(xs) and all(isinstance(x, dict) and isinstance(x.get("id"), str) for x in xs)


def diff(a, b, base: tuple = ()) -> list[Change]:
    """Leaf-level differences between two JSON documents. Lists whose items all carry an ``id`` (actors, actions,
    static objects) are aligned by id, so removing an action in the middle is one ``remove`` and not a cascade."""
    out: list[Change] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a.keys() | b.keys():
            p = base + (k,)
            if k not in b:
                out.append(Change(pointer(p), "remove", a[k]))
            elif k not in a:
                out.append(Change(pointer(p), "add", None, b[k]))
            else:
                out += diff(a[k], b[k], p)
    elif isinstance(a, list) and isinstance(b, list):
        if _id_list(a) and _id_list(b):
            ia = {x["id"]: i for i, x in enumerate(a)}
            ib = {x["id"]: i for i, x in enumerate(b)}
            for k, i in ia.items():
                if k not in ib:
                    out.append(Change(pointer(base + (i,)), "remove", a[i]))
            for k, j in ib.items():
                if k not in ia:
                    out.append(Change(pointer(base + (j,)), "add", None, b[j]))
                else:
                    out += diff(a[ia[k]], b[j], base + (j,))
        else:
            for i in range(min(len(a), len(b))):
                out += diff(a[i], b[i], base + (i,))
            for i in range(len(b), len(a)):
                out.append(Change(pointer(base + (i,)), "remove", a[i]))
            for i in range(len(a), len(b)):
                out.append(Change(pointer(base + (i,)), "add", None, b[i]))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) \
            and not isinstance(b, bool):
        if abs(a - b) > 1e-9:
            out.append(Change(pointer(base), "replace", a, b))
    elif a != b:
        out.append(Change(pointer(base), "replace", a, b))
    return sorted(out, key=lambda c: c.path)


# ----------------------------------------------------------------------------- whitelist grammar
def _alt(frags: list[str]) -> str:
    return "(?:" + "|".join(frags) + ")" if frags else "(?!)"


def write_pattern(dims) -> str:
    """Anchored regex for the ``path`` of an editor op (embedded into the JSON schema as a decoding constraint)."""
    frags = [f for d in dims for part in ("primary", "secondary") for f in WRITE[d][part]]
    return "^" + _alt(frags) + "$"


def _matches(frags: list[str], path: str) -> bool:
    return any(re.fullmatch(f, path) for f in frags)


def is_housekeeping(path: str) -> bool:
    """/meta and /parameters are code-owned (ids, lineage, tags, derived parameter ranges)."""
    return _matches(HOUSEKEEPING, path)


def whitelist_text(dims) -> list[str]:
    """Human-readable whitelist for the prompt."""
    t = {
        "weather": ["/environment/{time_of_day,weather,precipitation,fog_visibility_m,cloud_cover,road_surface}"],
        "behavior": ["/actors/<i>/{initial_speed,route,headway}", "/actors/<i>/initial/*",
                     "/actions/** (add / replace / remove actions; append with /actions/-)",
                     "/actors/- (append ONE OR MORE new NPC actors, with their actions)",
                     "/assertions/<i>/{value,value2} (retune a threshold only when the new behaviour needs it)",
                     "/duration"],
        "environment": ["/road/** (lanes, lane width, turn lanes, speed limit, median, shoulder, segments, curvature, "
                        "junction arms, signal plan, corner radius)", "/static_objects/** (append with "
                        "/static_objects/-)", "/environment/setting", "/actors/<i>/initial/* (only to keep actors on "
                        "the changed road)", "/duration"]}
    return [x for d in dims for x in t[d]]


@dataclass
class Isolation:
    ok: bool
    violations: list[dict] = field(default_factory=list)
    by_dim: dict = field(default_factory=dict)       # dim -> number of changes that fall into its primary whitelist
    untouched: list[str] = field(default_factory=list)   # requested dims with no change at all
    housekeeping: list[str] = field(default_factory=list)
    changes: list[dict] = field(default_factory=list)

    def messages(self) -> list[str]:
        out = [f"{v['op']} {v['path']}: {v['why']}" for v in self.violations]
        out += [f"dimension '{d}' was requested but nothing in its whitelist changed" for d in self.untouched]
        return out

    def to_dict(self) -> dict:
        return {"ok": self.ok, "violations": self.violations, "by_dim": self.by_dim, "untouched": self.untouched,
                "housekeeping": self.housekeeping, "n_changes": len(self.changes)}


def isolation(parent: dict, child: dict, dims, require_all: bool = True) -> Isolation:
    """The diff vs the parent may only touch the whitelist of ``dims``. Appended actors must be NPCs."""
    dims = list(dims)
    changes = diff(parent, child)
    n_par_actors = len(parent.get("actors", []))
    allow = [f for d in dims for part in ("primary", "secondary") for f in ALLOW[d][part]]
    iso = Isolation(True, changes=[c.to_dict() for c in changes])
    by_dim = {d: 0 for d in dims}
    for c in changes:
        if _matches(HOUSEKEEPING, c.path):
            iso.housekeeping.append(c.path)
            continue
        m = re.fullmatch(r"/actors/([0-9]+)", c.path)
        if m and "behavior" in dims and c.op == "add" and int(m.group(1)) >= n_par_actors:
            if (c.after or {}).get("role", "npc") != "npc":
                iso.violations.append({"path": c.path, "op": c.op, "why": "an added actor must have role 'npc'"})
            else:
                by_dim["behavior"] += 1
            continue
        if not _matches(allow, c.path):
            iso.violations.append({"path": c.path, "op": c.op, "why": "outside the whitelist of "
                                   + "+".join(DIM_TITLE[d] for d in dims) + " (" + "; ".join(whitelist_text(dims))[:300] + ")"})
            continue
        for d in dims:
            if _matches(ALLOW[d]["primary"], c.path):
                by_dim[d] += 1
    iso.by_dim = by_dim
    iso.untouched = [d for d, n in by_dim.items() if n == 0] if require_all else []
    if require_all and not any(by_dim.values()):
        iso.untouched = list(dims)
    iso.ok = not iso.violations and not iso.untouched
    return iso


def index_table(doc: dict) -> str:
    """Array indices for the prompt (the model addresses actors / actions / static objects by index)."""
    rows = [f"actors: " + ", ".join(f"{i}={a['id']}({a.get('role', 'npc')})" for i, a in enumerate(doc["actors"]))]
    if doc.get("actions"):
        rows.append("actions: " + ", ".join(f"{i}={a['id']}" for i, a in enumerate(doc["actions"])))
    if doc.get("assertions"):
        rows.append("assertions: " + ", ".join(f"{i}={a['kind']}" for i, a in enumerate(doc["assertions"])))
    if doc.get("static_objects"):
        rows.append("static_objects: " + ", ".join(f"{i}={a['id']}" for i, a in enumerate(doc["static_objects"])))
    road = doc["road"]
    if road.get("type") == "junction":
        rows.append("road.approaches: " + ", ".join(f"{i}=heading {a['heading_deg']:.0f}deg"
                                                    for i, a in enumerate(road["approaches"])))
    elif road.get("type") == "linear":
        rows.append("road.segments: " + ", ".join(f"{i}={s['kind']} {s['length']:.0f}m"
                                                  for i, s in enumerate(road["segments"])))
    return "\n".join(rows)
