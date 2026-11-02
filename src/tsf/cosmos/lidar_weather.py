"""Weather for generated LiDAR: how much the captions change it, and an optional physics-based degradation.

The joint Sim-Auto model attends the camera captions, so a rain / fog / snow caption may or may not reach the LiDAR.
``compare_variants`` measures that (same control, same seed, different caption: points, max range, near-range clutter).
``degrade`` is the deterministic post-process used when the effect is too small: it works on the generated range maps
(``frames`` ``[3, T, 128, 1800]``: range m, intensity, validity), keeps the input untouched and is the identity in
clear air.

Model (engineering estimate, every constant is a field of :class:`WeatherParams`)
  extinction ``alpha`` [1/m] = fog (Koschmieder, ``3.912 / V``, V = TSIR ``fog_visibility_m``) + rain
  (``k_rain * rate^g_rain``, rate = 60 mm/h * ``precipitation``) + snow (``k_snow * rate^g_snow``, 30 mm/h w.e. *
  ``precipitation``); the constants follow the usual 905 nm geometric-optics fits (heavy rain 25 mm/h ~ 9 dB/km).
  * dropout: a return at range R with reflectivity rho (the intensity channel, floored at ``rho_floor``) has the
    link margin ``M = rho / rho_min * (r_ref / R)^2 * exp(-2 alpha R)`` (two-way Beer-Lambert, 1/R^2 spreading);
    it is kept with probability ``Phi(10 log10 M / sigma_dB) / Phi(10 log10 M_clear / sigma_dB)`` (clear-air margin as
    the reference, so alpha = 0 keeps everything) and the intensity is scaled by ``exp(-2 alpha R)``.
  * rain / snow also drop returns with probability ``0.25 * precipitation * R / 50 m`` (capped at 0.9): drop interference
    and wet-surface specular loss, anchored on the 10-35 % point-count loss reported for heavy rain at 100 m;
  * range noise (rain / snow): Gaussian, ``0.12 m * precipitation * sqrt(1 + R / 30)``.
  * clutter (new returns, nearest wins, intensity 0.005..0.08): fog backscatter on a fraction ``1 - exp(-2.5 alpha_fog)``
    of the rays at ``5 m + Exp(4 m)``; rain on ``0.03 * precipitation`` of the rays at ``5 m + Exp(6 m)``; snow on
    ``0.06 * precipitation`` at ``5 m + Exp(5 m)`` (all clipped at 30 m and in front of the existing return).
All randomness comes from ``default_rng([seed, sweep, 0x57])``: the same (frames, environment, seed) gives the same result.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from tsf.cosmos import lidar as L

PRECIP_DEFAULT = {"rain": 0.35, "heavy_rain": 0.75, "snow": 0.5}      # when the weather label says more than the numbers
FOG_DEFAULT_VIS_M = 120.0
CLEAR_VIS_M = 20000.0                                                  # above this the air counts as clear


@dataclass(frozen=True)
class WeatherParams:
    rho_min: float = 0.05
    r_ref: float = 100.0
    rho_floor: float = 0.10
    snr_sigma_db: float = 1.5
    rain_rate_max: float = 60.0
    k_rain: float = 2.48e-4
    g_rain: float = 0.67
    snow_rate_max: float = 30.0
    k_snow: float = 7.4e-4
    g_snow: float = 0.75
    precip_dropout: float = 0.25
    noise_m: float = 0.12
    fog_bs_gain: float = 2.5
    rain_clutter: float = 0.03
    snow_clutter: float = 0.06
    clutter_max_m: float = 30.0


def effective_environment(env: dict | None) -> dict:
    """TSIR ``environment`` (+ appearance overrides) with the numeric fields filled in from the weather label:
    ``precipitation`` (rain 0.35, heavy_rain 0.75, snow 0.5 when it is 0) and ``fog_visibility_m`` (fog 120 m when the
    visibility is the clear-air default); a missing ``time_of_day`` / ``road_surface`` gets 12:00 / dry."""
    e = dict(env or {})
    e.setdefault("time_of_day", "12:00")          # lifted scenarios may leave the environment partly unset
    e.setdefault("road_surface", "dry")
    w = str(e.get("weather", "clear"))
    prec = float(e.get("precipitation", 0.0) or 0.0)
    vis = float(e.get("fog_visibility_m", 100000.0) or 100000.0)
    if w in PRECIP_DEFAULT and prec <= 0.0:
        prec = PRECIP_DEFAULT[w]
    if w == "fog" and vis >= CLEAR_VIS_M:
        vis = FOG_DEFAULT_VIS_M
    e.update(weather=w, precipitation=prec, fog_visibility_m=vis)
    return e


def is_adverse(env: dict | None) -> bool:
    e = effective_environment(env)
    return e["fog_visibility_m"] < CLEAR_VIS_M or (e["precipitation"] > 0.0 and e["weather"] in (*PRECIP_DEFAULT, "fog"))


def extinction(env: dict | None, p: WeatherParams = WeatherParams()) -> dict:
    """``{fog, rain, snow, total}`` extinction coefficients in 1/m for an environment."""
    e = effective_environment(env)
    fog = 3.912 / e["fog_visibility_m"] if e["fog_visibility_m"] < CLEAR_VIS_M else 0.0
    prec = e["precipitation"] if e["weather"] in (*PRECIP_DEFAULT, "fog") else 0.0
    rain = snow = 0.0
    if e["weather"] in ("rain", "heavy_rain") or (e["weather"] == "fog" and prec > 0):
        rain = p.k_rain * (p.rain_rate_max * prec) ** p.g_rain if prec > 0 else 0.0
    if e["weather"] == "snow":
        snow = p.k_snow * (p.snow_rate_max * prec) ** p.g_snow if prec > 0 else 0.0
    return {"fog": fog, "rain": rain, "snow": snow, "total": fog + rain + snow,
            "precipitation": prec if (rain or snow) else 0.0}


def _phi(x):
    from scipy.special import erf
    return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))


def degrade(frames, env: dict | None, seed: int = 0, params: WeatherParams = WeatherParams(), *, thr: float = L.VALID_THRESHOLD):
    """Weather-degraded copy of generated range maps. Returns ``(frames_out, report)``; ``frames`` is not modified."""
    a = extinction(env, params)
    x = np.array(frames, dtype=np.float32)                              # copy
    rep = {"environment": effective_environment(env), "alpha_per_m": {k: round(v, 6) for k, v in a.items()},
           "params": asdict(params), "seed": seed, "sweeps": []}
    if a["total"] <= 0.0:
        rep["note"] = "clear air: unchanged"
        return x, rep
    rays_z = L.ray_directions()[..., 2]
    prec = a["precipitation"]
    for i in range(x.shape[1]):
        rng = np.random.default_rng([int(seed), i, 0x57])
        valid = L.valid_mask(x, i, thr)
        r = x[0, i]
        n0 = int(valid.sum())
        # --- dropout (link margin) and intensity loss
        two_way = np.exp(-2.0 * a["total"] * r)
        rho = np.maximum(x[1, i], params.rho_floor)
        m_clear = rho / params.rho_min * (params.r_ref / np.maximum(r, 1.0)) ** 2
        db = lambda m: 10.0 * np.log10(np.maximum(m, 1e-9))      # noqa: E731
        p_keep = np.clip(_phi(db(m_clear * two_way) / params.snr_sigma_db) /
                         np.maximum(_phi(db(m_clear) / params.snr_sigma_db), 1e-9), 0.0, 1.0)
        keep = valid & (rng.random(r.shape) < p_keep)
        if prec > 0:                                          # drop interference / wet-surface loss, grows with range
            keep &= rng.random(r.shape) >= np.clip(params.precip_dropout * prec * (r / 50.0), 0.0, 0.9)
        x[1, i] = np.where(keep, np.clip(x[1, i] * two_way, 0.0, 1.0), 0.0)
        # --- range noise (precipitation)
        if prec > 0:
            sig = params.noise_m * prec * np.sqrt(1.0 + r / 30.0)
            r = np.where(keep, np.clip(r + rng.normal(0.0, 1.0, r.shape) * sig, L.MIN_RANGE_M, L.MAX_RANGE_M), 0.0)
        else:
            r = np.where(keep, r, 0.0)
        # --- clutter: backscatter / drops / flakes, nearest return wins
        p_cl, scale = 0.0, 4.0
        if a["fog"] > 0:
            p_cl = 1.0 - math.exp(-params.fog_bs_gain * a["fog"])
        if a["rain"] > 0:
            p_cl, scale = max(p_cl, params.rain_clutter * prec), 6.0
        if a["snow"] > 0:
            p_cl, scale = max(p_cl, params.snow_clutter * prec), 5.0
        hit = rng.random(r.shape) < p_cl
        rc = np.minimum(L.MIN_RANGE_M + rng.exponential(scale, r.shape), params.clutter_max_m)
        height = rc * rays_z + L.LIDAR_Z_M                                         # keep clutter above the ground
        put = hit & (height > 0.1) & ((r <= 0) | (rc < r - 0.5))
        ic = rng.uniform(0.005, 0.08, r.shape).astype(np.float32)
        r = np.where(put, rc, r)
        x[1, i] = np.where(put, ic, x[1, i])
        x[0, i] = r.astype(np.float32)
        x[2, i] = (r > 0).astype(np.float32)
        x[1, i] = np.where(r > 0, x[1, i], 0.0)
        rep["sweeps"].append({"points_in": n0, "dropped": int(valid.sum() - keep.sum()), "clutter_added": int(put.sum()),
                              "points_out": int((r > 0).sum())})
    rep["totals"] = {k: int(sum(s[k] for s in rep["sweeps"])) for k in ("points_in", "dropped", "clutter_added", "points_out")}
    return x, rep


def compare_variants(variants: dict[str, object], control, *, reference: str = "clear", noise_reference: str | None = None) -> dict:
    """Weather-vs-clear table. ``variants`` = ``{name: frames or safetensors path}`` of generations with the same control
    and seed; ``reference`` is the clear caption, ``noise_reference`` (optional) a second clear run with another seed:
    its difference to the reference is the seed noise floor the weather effect has to exceed. Metrics are the per-sweep
    means of :func:`tsf.cosmos.lidar.cloud_metrics`; ``effect_over_noise`` = |delta| / |seed delta|."""
    mets = {}
    for name, v in variants.items():
        fr = L.read_safetensors(v)[0]["frames"] if isinstance(v, (str, bytes)) or hasattr(v, "__fspath__") else v
        mets[name] = L.cloud_metrics(fr, control)["mean"]
    ref = mets[reference]
    out = {"metrics": mets, "delta_vs_" + reference: {}, "ratio_vs_" + reference: {}}
    keys = [k for k in ref if k not in ("valid_fraction",)]
    for name, m in mets.items():
        if name == reference:
            continue
        out["delta_vs_" + reference][name] = {k: round(m[k] - ref[k], 4) for k in keys}
        out["ratio_vs_" + reference][name] = {k: round(m[k] / ref[k], 4) if ref[k] else None for k in keys}
    if noise_reference and noise_reference in mets:
        nz = {k: abs(mets[noise_reference][k] - ref[k]) for k in keys}
        out["seed_noise"] = {k: round(v, 4) for k, v in nz.items()}
        out["effect_over_noise"] = {name: {k: round(abs(m[k] - ref[k]) / nz[k], 2) if nz[k] > 0 else None for k in keys}
                                    for name, m in mets.items() if name not in (reference, noise_reference)}
    return out
