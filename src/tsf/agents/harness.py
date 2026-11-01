"""Harness² runtime: versioned, editable harness components + verifier-backed experience library.

Layout (repo `harness/`):
  <agent>/system_prompt.md, guardrails.md, tool_guidance.md   editable text components
  <agent>/skills/*.json                                         verified few-shot exemplars
  <agent>/knobs.json                                            decoding/loop parameters
  EXPERIENCE.jsonl                                              lessons with applicability + evidence
  revisions.jsonl                                               Propose-Probe-Compose decisions
Only entries whose evidence shows a verifier-score gain are `active` (Harness² library level).
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

# live (evolving) harness state; defaults to the versioned copy in the repo
ROOT = Path(os.environ.get("TSF_HARNESS", Path(__file__).resolve().parents[3] / "harness"))

_SYN = {"intersection": "junction", "crossing": "junction", "crossroad": "junction", "traffic": "signal",
        "light": "signal", "lights": "signal", "signalized": "signal", "motorway": "highway", "freeway": "highway",
        "autobahn": "highway", "expressway": "highway", "linear": "highway", "merge": "cut", "cuts": "cut"}


def _terms(text: str) -> set[str]:
    return {_SYN.get(w, w) for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2}


DEFAULT_KNOBS = {"thinking": False, "temperature": 0.3, "max_repairs": 3, "n_skills": 1, "n_lessons": 6}


@dataclass
class Lesson:
    id: str
    scope: str                 # agent name or "*"
    applies_to: list[str]      # keywords/tags; empty = always
    lesson: str
    kind: str = "DOMAIN"       # INVARIANT | DOMAIN
    status: str = "candidate"  # candidate | active | retired
    evidence: dict = field(default_factory=dict)   # {"delta_score":..., "n":..., "runs":[...]}
    created: float = field(default_factory=time.time)


class Harness:
    def __init__(self, agent: str, root: Path = ROOT):
        self.agent, self.root = agent, root
        self.dir = root / agent

    def part(self, name: str) -> str:
        p = self.dir / name
        return p.read_text(encoding="utf-8").strip() if p.exists() else ""

    def knobs(self) -> dict:
        p = self.dir / "knobs.json"
        return {**DEFAULT_KNOBS, **(json.loads(p.read_text()) if p.exists() else {})}

    def skills(self, n: int, text: str = "") -> list[dict]:
        """Verified exemplars, most relevant to `text` first (term overlap incl. road-type synonyms)."""
        docs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((self.dir / "skills").glob("*.json"))]
        if text:
            q = _terms(text)
            docs.sort(key=lambda d: -len(q & _terms(" ".join([d["meta"]["title"], d["meta"]["description"],
                                                              " ".join(d["meta"].get("tags", [])),
                                                              d["road"]["type"]]))))
        return docs[:n]

    # ---------------------------------------------------------------- experience library
    def _lessons_path(self) -> Path:
        return self.root / "EXPERIENCE.jsonl"

    def lessons(self, text: str = "", n: int = 6, include_candidates: bool = False) -> list[Lesson]:
        p = self._lessons_path()
        if not p.exists():
            return []
        words = set(re.findall(r"[a-z_]+", text.lower()))
        rows = [Lesson(**json.loads(ln)) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        ok = {"active"} | ({"candidate"} if include_candidates else set())
        rows = [r for r in rows if r.scope in (self.agent, "*") and r.status in ok]
        rows = [r for r in rows if not r.applies_to or words & set(r.applies_to)]
        rows.sort(key=lambda r: -r.evidence.get("delta_score", 0.0))
        return rows[:n]

    def add_lesson(self, lesson: Lesson):
        with self._lessons_path().open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(lesson)) + "\n")

    def set_lesson_status(self, lesson_id: str, status: str, evidence: dict | None = None):
        p = self._lessons_path()
        rows = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for r in rows:
            if r["id"] == lesson_id:
                r["status"] = status
                if evidence:
                    r["evidence"] = {**r.get("evidence", {}), **evidence}
        p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def log_revision(self, editor: str, target: str, proposal: str, verdict: str, evidence: dict) -> str:
        rid = f"rev_{uuid.uuid4().hex[:10]}"
        with (self.root / "revisions.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"revision_id": rid, "agent": self.agent, "editor_type": editor, "target": target,
                                "proposal": proposal, "verdict": verdict, "evidence": evidence,
                                "t": time.time()}) + "\n")
        return rid

    # ---------------------------------------------------------------- prompt assembly
    def system(self, text: str = "", extra_lessons: list[Lesson] | None = None) -> str:
        k = self.knobs()
        lessons = (extra_lessons or []) + self.lessons(text, k["n_lessons"])
        parts = [self.part("system_prompt.md"), "## Authoring guide\n" + self.part("tool_guidance.md"),
                 "## Guardrails\n" + self.part("guardrails.md")]
        if lessons:
            parts.append("## Verified lessons from previous runs\n" + "\n".join(f"- {l.lesson}" for l in lessons))
        return "\n\n".join(p for p in parts if p.strip())
