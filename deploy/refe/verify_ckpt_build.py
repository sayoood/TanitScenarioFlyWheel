#!/usr/bin/env python3
"""Checkpoint lineage check (runs in the refe container, paths are the container's: /work = ~/tsf/dev/refe).

Rebuild the SFT-4 B u400 FULL checkpoint with TanitAD's own eval/build_sft_ckpt.py from the base model_final.pt (HF repo
Sayood/tanitad-refe-vitl16-navtrain10-grow-final) and the scorer states, and compare with Sayood/tanitad-refe-rls1/base/model_sft4_B_u400.pt.
The repos do not publish `scorer_states_u400.pt`; its content is the score_q_mlp / score_dec / score_head tensors of the published
SFT-4 checkpoint (that is what build_sft_ckpt copies into the base), so we cut it from there and check that EVERYTHING ELSE already equals the base."""
import hashlib
import subprocess
import sys
import time
import torch
from huggingface_hub import hf_hub_download
t0 = time.time()
base = hf_hub_download("Sayood/tanitad-refe-vitl16-navtrain10-grow-final", "model_final.pt", local_dir="/work/ckpt/base")
def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()
print("base md5", md5(base), "(expected b54773f8c302b2ec8ff4c5971d1e1370)", f"{time.time()-t0:.0f}s", flush=True)
sft = torch.load("/work/ckpt/rls1/base/model_sft4_B_u400.pt", map_location="cpu", weights_only=False)
print("sft4 keys", list(sft), "meta", {k: v for k, v in sft.get("meta", {}).items() if k != "sft_intermediate"}, flush=True)
st = sft["model"]
mods = ("score_q_mlp", "score_dec", "score_head")
B = {m: {k[len(m) + 1:]: v for k, v in st.items() if k.startswith(m + ".")} for m in mods}
torch.save({"B": B, "A": B, "update": 400}, "/work/ckpt/scorer_states_u400_cut.pt")
out = "/work/ckpt/rebuilt_sft4_B_u400.pt"
src = "/work/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/eval/build_sft_ckpt.py"
r = subprocess.run([sys.executable, src, "--base", base, "--states", "/work/ckpt/scorer_states_u400_cut.pt", "--arm", "B", "--out", out],
                   capture_output=True, text=True)
print(r.stdout.strip()[-400:], r.stderr.strip()[-600:], flush=True)
re = torch.load(out, map_location="cpu", weights_only=False)["model"]
bs = torch.load(base, map_location="cpu", weights_only=False)["model"]
diff_scorer = diff_other = n = 0
for k, v in st.items():
    n += 1
    same = k in re and re[k].shape == v.shape and torch.equal(re[k], v)
    if not same:
        diff_scorer += 1
print(f"rebuilt vs published SFT-4: {n} tensors, {diff_scorer} differ (must be 0); extra keys {set(re)-set(st)}; missing {set(st)-set(re)}")
# lineage: tensors OUTSIDE the three scorer modules are identical between the published SFT-4 checkpoint and the base model_final
other = [k for k in st if not any(k.startswith(m + ".") for m in mods)]
bad = [k for k in other if k not in bs or not torch.equal(bs[k], st[k])]
print(f"non-scorer tensors: {len(other)}, differing from base model_final.pt: {len(bad)} {bad[:3]}")
chg = [k for k in st if any(k.startswith(m + ".") for m in mods) and (k not in bs or bs[k].shape != st[k].shape or not torch.equal(bs[k], st[k]))]
print(f"scorer tensors changed by SFT-4 vs base: {len(chg)} of {len(st)-len(other)}")
print("sha/md5 rebuilt", md5(out), "published md5 (HF rls1/base) 8e382eebb35b3e155862453617e5786e", f"total {time.time()-t0:.0f}s")
