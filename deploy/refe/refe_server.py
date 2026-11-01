#!/usr/bin/env python3
"""Persistent REFe SFT-4 inference server (the policy under test of the TSF closed loop).

Runs inside a torch container on thor3 (needs torch + numpy + fastapi/uvicorn; see deploy/refe/README.md), loads the
TanitAD model code from a checkout (``--refe-src``, the ``refe/`` directory of TanitAD ``2026-09-20-refe-plan``;
the model code is NOT vendored into this public repo) and the FULL checkpoint ``model_sft4_B_u400.pt``.

Wire protocol (``tsf.closedloop.refe.RefeClient`` is the only intended client)
  POST /infer   body = ``np.savez`` archive with
                  img   uint8   [4, 512, 960, 3]  RGB, camera order CAM_F0, CAM_L0, CAM_R0, CAM_B0
                  ego   float32 [7]               vx vy ax ay yaw_rate tire_steering_angle speed (rear axle, body frame)
                  goal  float32 [4]               2 route goal points (x1 y1 x2 y2) in the ego frame (goal-mode checkpoints)
                  calib float64 [4, 16]           optional per-camera rig (calib_table.FIELDS); default = the model's baked rig
                response JSON: traj (20 x [x, y, yaw] @ 5 Hz, ego frame), k, agg, latency_ms, ...
  GET  /health  GET /info

Selection follows TanitAD ``refe/planner.py``: proposal k = argmax of the benchmark aggregate over the 64 scored
proposals; a 7-output checkpoint (SFT-4) uses the rule ``navsim_v1_lane`` = NC*DAC*(5 EP + 5 TTC + 2 C)/12 * p(lane).
"""
import argparse
import io
import sys
import threading
import time

import numpy as np
import torch

CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")
V1_W = (5.0, 5.0, 2.0)          # progress, ttc, comfort   (planner.REFePlanner.V1_W)


def aggregate(score: torch.Tensor, rule: str = "auto") -> torch.Tensor:
    """[.., M, K] logits -> [.., M] aggregate (planner.REFePlanner.aggregate, rules navsim_v1 / navsim_v1_lane)."""
    p = score.float().sigmoid()
    if rule == "auto":
        rule = "navsim_v1_lane" if p.shape[-1] >= 7 else "navsim_v1"
    w = torch.tensor(V1_W, device=p.device, dtype=p.dtype)
    base = p[..., 0] * p[..., 1] * (p[..., 2] * w[0] + p[..., 3] * w[1] + p[..., 4] * w[2]) / w.sum()
    if rule == "navsim_v1_lane":
        if p.shape[-1] < 7:
            raise ValueError("rule navsim_v1_lane needs a 7-output scorer (an SFT-4 checkpoint)")
        return base * p[..., 6]
    if rule == "navsim_v1":
        return base
    raise ValueError(f"unknown rule {rule!r}")


class RefeEngine:
    def __init__(self, ckpt: str, refe_src: str, device: str = "cuda", dtype: str = "bf16", rule: str = "auto"):
        sys.path.insert(0, refe_src)
        import ckpt_io                                  # TanitAD refe/ckpt_io.py
        from model import REFe, REFeConfig              # TanitAD refe/model.py
        self.ckpt_path, self.rule = ckpt, rule
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = dtype
        cfg = REFeConfig.for_backbone("vitl16")
        cfg = ckpt_io.config_for_checkpoint(cfg, ckpt)  # 7-output scorer + nav_mode from the checkpoint meta
        self.cfg = cfg
        t0 = time.time()
        self.model = REFe(cfg).eval()
        meta: dict = {}
        self.ckpt_format = ckpt_io.load_for_inference(self.model, ckpt, map_location="cpu", meta_out=meta)
        self.meta = meta
        self.per_sample_calib = bool(meta.get("per_sample_calib", False))
        self.model.to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.load_s = time.time() - t0
        self.baked_calib = torch.tensor([self.model.baked_calib()], dtype=torch.float64)   # [1, 4, 16]
        self.lock = threading.Lock()
        self.n_calls = 0

    # ------------------------------------------------------------------ one forward
    @torch.no_grad()
    def infer(self, img: np.ndarray, ego, goal, calib=None, return_all: bool = False) -> dict:
        c = self.cfg
        assert img.shape == (c.n_cameras, c.img_h, c.img_w, 3) and img.dtype == np.uint8, (img.shape, img.dtype)
        t0 = time.perf_counter()
        x = torch.from_numpy(np.ascontiguousarray(img)).to(self.device, non_blocking=True)
        x = x.permute(0, 3, 1, 2).float().div_(255.0).unsqueeze(0)                    # [1, N, 3, H, W]
        ego_t = torch.as_tensor(np.asarray(ego, np.float32).reshape(1, -1), device=self.device)
        goal_t = torch.as_tensor(np.asarray(goal, np.float32).reshape(1, -1), device=self.device)
        cal = None
        if self.per_sample_calib:       # a model trained on per-sample rigs must be FED a rig (planner.py _calib_for)
            cal = self.baked_calib if calib is None else torch.as_tensor(np.asarray(calib, np.float64)).reshape(1, c.n_cameras, 16)
        elif calib is not None:
            raise ValueError("this checkpoint was not trained with per-sample calibration; do not send `calib`")
        if self.device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        with self.lock:
            use_amp = self.dtype == "bf16" and self.device == "cuda"
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                traj, score = self.model(x, ego_t, goal_t, calib=cal)
            if self.device == "cuda":
                torch.cuda.synchronize()
        t2 = time.perf_counter()
        agg = aggregate(score, self.rule)[0]
        k = int(agg.argmax())
        self.n_calls += 1
        out = {
            "traj": traj[0, k].float().cpu().numpy().tolist(),          # [20, 3] x y yaw, ego frame, 5 Hz, t = 0.2 (i+1)
            "k": k, "agg": float(agg[k]), "agg_top5": [float(v) for v in agg.topk(5).values],
            "score_sigmoid": [float(v) for v in score[0, k].float().sigmoid()],
            "n_proposals": int(traj.shape[1]),
            "latency_ms": {"h2d": (t1 - t0) * 1e3, "forward": (t2 - t1) * 1e3, "total": (time.perf_counter() - t0) * 1e3},
        }
        if return_all:
            out["proposals"] = traj[0].float().cpu().numpy().tolist()
            out["agg_all"] = agg.float().cpu().numpy().tolist()
        return out

    def info(self) -> dict:
        n = sum(p.numel() for p in self.model.parameters())
        return {"ckpt": self.ckpt_path, "ckpt_format": self.ckpt_format, "params": n, "device": self.device,
                "dtype": self.dtype, "rule": self.rule, "n_score_components": self.cfg.n_score_components,
                "nav_mode": self.cfg.nav_mode, "per_sample_calib": self.per_sample_calib,
                "cameras": list(self.cfg.cameras), "img_hw": [self.cfg.img_h, self.cfg.img_w],
                "horizon_steps": self.cfg.horizon_steps, "ego_dim": self.cfg.ego_dim,
                "n_goal_points": self.cfg.n_goal_points, "load_s": round(self.load_s, 1),
                "torch": torch.__version__,
                "gpu_alloc_mb": (torch.cuda.memory_allocated() / 2**20 if self.device == "cuda" else 0)}


# ====================================================================== http
def make_app(engine: RefeEngine):
    # NB: no `from __future__ import annotations` in this module: FastAPI resolves the endpoint annotations (Request) at runtime
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    app = FastAPI(title="tsf-refe")

    @app.get("/health")
    def health():
        return {"ok": True, "calls": engine.n_calls}

    @app.get("/info")
    def info():
        return engine.info()

    @app.post("/infer")
    async def infer(request: Request):
        body = await request.body()
        z = np.load(io.BytesIO(body), allow_pickle=False)
        try:
            out = engine.infer(z["img"], z["ego"], z["goal"], z["calib"] if "calib" in z.files else None,
                               return_all=bool(z["return_all"]) if "return_all" in z.files else False)
        except Exception as e:                                     # surface the reason to the client, never a silent hold
            return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=400)
        return out

    return app


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="FULL checkpoint, e.g. model_sft4_B_u400.pt")
    ap.add_argument("--refe-src", required=True, help="TanitAD .../2026-09-20-refe-plan/refe directory")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp32"))
    ap.add_argument("--rule", default="auto", choices=("auto", "navsim_v1", "navsim_v1_lane"))
    ap.add_argument("--selftest", action="store_true", help="random-input forward passes, print latency and memory, exit")
    a = ap.parse_args()
    eng = RefeEngine(a.ckpt, a.refe_src, dtype=a.dtype, rule=a.rule)
    print("INFO", eng.info(), flush=True)
    if a.selftest:
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, (4, 512, 960, 3), dtype=np.uint8)
        ego = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0]
        goal = [60.0, 0.0, 120.0, 0.0]
        lat = []
        for i in range(12):
            r = eng.infer(img, ego, goal)
            lat.append(r["latency_ms"]["forward"])
            if i == 0:
                print("first call", r["latency_ms"], "k", r["k"], "traj_end", r["traj"][-1], flush=True)
        print("SELFTEST forward_ms median %.1f min %.1f max %.1f" % (np.median(lat[2:]), min(lat[2:]), max(lat[2:])), flush=True)
        if eng.device == "cuda":
            print("SELFTEST gpu peak MB", torch.cuda.max_memory_allocated() / 2**20,
                  "reserved MB", torch.cuda.memory_reserved() / 2**20)
        return 0
    import uvicorn
    uvicorn.run(make_app(eng), host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
