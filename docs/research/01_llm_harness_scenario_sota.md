# Research 01: Local LLM, Harness², scenario-generation SOTA (as of 2026-11-01)

Source: research agent, primary pages fetched. "unverified" = could not be opened at the primary source.

## 1. Local LLM on Jetson Thor
- **Qwen3.8-27B** exists: https://huggingface.co/Qwen/Qwen3.8-27B (Aug 14 2026), Apache-2.0, dense 27B **VLM** (image+video), 64 layers alternating Gated DeltaNet / Gated Attention, 262K ctx (1M YaRN), MTP head, thinking on by default. No open Qwen3.7 weights. Siblings: Qwen3.8-Flash-Next (125B-A6B VLM, qwen-community-1.0). Fallback: Qwen3.6-27B (Apr 2026).
- Quants: `unsloth/Qwen3.8-27B-GGUF` (UD-Q4_K_XL 17.6 GB, UD-Q8_K_XL 31.5 GB, MTP Q4_0 1.37 GB); `unsloth/Qwen3.8-27B-NVFP4` (Dynamic v3 preview, 92-97 % recovery, vLLM >= 0.25); `nvidia/Qwen3.8-27B-NVFP4` (ModelOpt, NVFP4 MLP + FP8 attn + FP8 KV); `Qwen/Qwen3.8-27B-FP8`.
- Thor: 273 GB/s LPDDR5X, so dense 27B @4 bit is bandwidth-bound to ~17 tok/s without speculation.
  - TensorRT Edge-LLM 0.10: Qwen3.5-27B NVFP4 14.7 tok/s b1, 95 tok/s b8, 35.0 with MTP; Qwen3.6-35B-A3B 84.5 tok/s b1.
  - Jetson AI Lab: llama.cpp Qwen3.8-27B Q4_K_M + MTP(3) ~13 to ~35 tok/s (https://www.jetson-ai-lab.com/tutorials/speculative-decoding/).
  - Community: FlashRT 42.8-52.9 tok/s, vLLM DFlash 42.3 tok/s (Qwen3.6-27B NVFP4 + MTP).
- Containers: `vllm/vllm-openai:v0.26.0-aarch64-ubuntu2404` (modelopt_fp4), `nvcr.io/nvidia/vllm:26.01-py3`. SGLang has no official Thor container.
- vLLM flags (https://recipes.vllm.ai/Qwen/Qwen3.8-27B): `--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder`, MTP `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`. Structured outputs via xgrammar (JSON schema / regex / EBNF); `enable_in_reasoning` is needed with a reasoning parser.

## 2. "Google Harness 2" = Harness²
- **Harness²: Recursive Agent Harnessing for an Open World**, arXiv 2610.14224 (Oct 9 2026), Google (Chen-Yu Lee, Tomas Pfister et al.), code https://github.com/google-research/harness2 (Apache-2.0). Predecessor: Meta-Harness, arXiv 2603.28052.
- Frozen model weights; the **harness** is edited. Task level: **Propose → Probe → Compose**.
  - Propose: a reference run, then an invariant candidate and a domain-specific candidate.
  - Probe: matched runs produce a behavioural contrast.
  - Compose: merge / revise / adopt / keep.
  - Parallel-k or sequential, with an agentic selector over anonymized outputs.
- Harnessing level: an **improvement library** where each entry is an executable harness plus EXPERIENCE.md (applicability, lessons, evidence). Lessons are kept only if execution contrasts support them.
- Eight editable components: system prompt, guardrails, memory, skills, tool guidance, scripts, subagents, plugin middleware.
- Results: GLM-5 WorkBuddy-Bench 64.2 to 75.7; LAB +11.8; Gemini 3.5 Flash JobBench 31.5 to 36.0. Ablations: library, scoping and composition each matter.
- Tanit twist: Harness² is verifier-free (contrastive). **We add hard verifiers** (schema, ASAM QC, semantic, sim).

## 3. Evidence base
- **NL to OSC/OpenDRIVE:**
  - ChatScene 2405.14062 (retrieval of description→code pairs).
  - ScenicNL 2405.03709 (compiler-in-the-loop, crash reports).
  - Text2Scenario 2503.02911 (.xosc, 80.3 %).
  - Txt2Sce 2509.02150 (template + mutation, 33 to 4,373 files).
  - LD-Scene 2505.11247 (LLM guidance for latent diffusion).
  - OmniTester 2409.06450.
  - OSC2: compilers only, no LLM (2604.16452 ANTLR4 + semantic pass; 2606.26533 OSC2Runner).
  - MeMoDecode 2610.29749: decode-time consistency, +23 % vs unconstrained, +8 % vs grammar-only.
- **Crash to scenario / counterfactual:**
  - SoVAR 2409.08081 (LLM + constraint solver).
  - CrashAgent 2505.18341 (multi-agent + counterfactuals).
  - CIREN→Scenic IR 2602.20644 (97-98 % trajectory/oracle match).
  - TRACE 2604.22068.
  - Negative-data retrieval 2509.18626 (recall 24 to 53 %).
  - No paper does "positive twin" generation, so this is a **gap and an opportunity**.
- **Safety-critical / coverage / mining:**
  - AdvScene 2609.20103 (initial scene as lever, diffusion + RL).
  - LAMBDA 2412.00517 (coverage criterion).
  - Teach-to-Crash 2609.27296 (LLM teacher only on stagnation, 90.79 % hit rate).
  - RefAV 2505.20981 (NL to programs over logs).
  - Agentic scenario mining with VLM verifier 2606.09180.
- **Closed loop without 3DGS:**
  - NAVSIM pseudo-simulation 2506.04218 (R²≈0.8, but uses 3DGS).
  - EPDMS (https://github.com/autonomousvision/navsim/blob/main/docs/metrics.md): multipliers NC, DAC, DDC, TLC; weighted EP 5, TTC 5, LK 2, HC 2, EC 2.
  - **NVIDIA OmniDreams** 2606.03159: real-time action-conditioned world model with AlpaSim, github.com/nv-tlabs/omni-dreams. Thor feasibility not stated.
  - DreamStream 2609.26792 (FDπ metric, Navhard-CL).
- **Self-improving harnesses:**
  - GEPA 2507.19457 (reflective Pareto prompt evolution, beats GRPO by +6 points avg, up to 35× fewer rollouts).
  - ACE 2510.04618 (evolving playbooks, +10.6 %).
  - Darwin Gödel Machine 2505.22954.
- **Tooling:**
  - esmini 3.10.2 (Oct 22 2026; OpenDRIVE 1.9 lanes since 3.10.0, no OSC2).
  - scenariogeneration 0.16.7 (OSC 1.1-1.3.1, OpenDRIVE 1.7.1).
  - py-osc2 0.1.0 (syntax only, OSC DSL 2.1.0 draft).
  - ASAM qc-framework / qc-opendrive (pip `asam-qc-opendrive`) / qc-openscenarioxml.
  - libOpenDRIVE (C++, Apache-2.0).
  - Standards: OSC XML 1.3.1, OSC DSL 2.1.0, OpenDRIVE 1.8.0 (unverified).

## 4. Ranked moat ideas (agent proposal)
1. Hard-verifier stack (schema → ASAM QC → semantic/intent → sim dry-run with TTC/collision asserts) used as reward.
2. Harness²-style two-level self-improvement, plus GEPA Pareto search, ACE playbooks and a DGM archive, scored by idea 1. Validators stay human-gated.
3. Incident → typed IR → compiled OSC1/OSC2/OpenDRIVE with constrained decoding (xgrammar) + MeMoDecode-style consistency, never raw XML. Build a regression corpus.
4. Counterfactual "positive twin" generator (avoided + near-miss variants labelled with TTC/EPDMS).
5. Coverage-guided criticality search (LAMBDA), with an LLM teacher only on stagnation and our sim as the cheap inner loop.
6. Initial-scene-first adversarial seeds (AdvScene/LD-Scene) behind a realism filter.
7. Fleet/video mining by program synthesis + VLM verification (RefAV-style).
8. Tiered closed loop: cheap kinematic/EPDMS tier, then the Cosmos world-model tier with ranking-preservation tests.
