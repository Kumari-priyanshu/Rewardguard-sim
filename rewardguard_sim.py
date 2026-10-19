#!/usr/bin/env python3
"""
RewardGuard-Sim: a toy simulation of a defense-in-depth framework against
reward hacking and deceptive alignment in an autonomous coding agent.

This is NOT a real LLM experiment. Agents are scripted policies, and every
detection probability below is a modelling assumption (see PARAMS). The
simulation shows structural behaviour (which layer catches what, how a
reasoning-only monitor degrades under obfuscation), not real-world rates.

Usage
-----
  python rewardguard_sim.py demo --policy test_editor --layers full --seed 3
  python rewardguard_sim.py experiments --seeds 5 --episodes 600 --out results
"""
import argparse, json, math, os, random, statistics as stt
from multiprocessing import Pool

T = 20  # steps per episode

# ---------------------------------------------------------------- parameters
PARAMS = dict(
    work_p=0.7, work_inc=0.10, post_hack_work_p=0.10, success_progress=0.90,
    # Layer 1: capability control (share of protected-resource writes blocked)
    l1_cov={"test_edit": 0.7, "tamper": 0.7, "hardcode": 0.0},
    # Layer 2: trajectory monitor channel detection probabilities per hack step
    env={"test_edit": 0.85, "tamper": 0.85, "hardcode": 0.35},
    act={"test_edit": 0.70, "tamper": 0.70, "hardcode": 0.30},
    rea=0.60, rea_obf=0.05,
    # false-trigger probability per source per episode
    fa=0.02,
    # Layer 3: reward vs progress gap detector (checked at these steps)
    l3_gap=0.30, l3_det=0.90, l3_checks=(5, 10, 15, 20),
    # Layer 5: independent verifier (hidden tests) at submission
    l5_gap=0.30, l5_det=0.95, l5_fa=0.01,
    # risk fusion weights and thresholds (Layer 6)
    w_l1=0.5, w_ch=0.4, w_l3=0.4, w_l5=0.7, flag=0.4, block=0.7, rollback=0.95,
    # Layer 4: paired visible/hidden audit before each deployment batch
    audit_runs=5, audit_thr=0.20, audit_flip=0.02, batch=50,
    # observer-sensitive agent: probability it believes it is unobserved
    obs_none=0.90, obs_rt=0.30, obs_announced=0.02,
)

POLICIES = {
    "honest":             dict(p=0.00, kind=None),
    "test_editor":        dict(p=0.30, kind="test_edit"),
    "hardcoder":          dict(p=0.30, kind="hardcode"),
    "tamperer":           dict(p=0.05, kind="tamper"),
    "observer_sensitive": dict(p=0.30, kind="test_edit", observer=True),
    "obfuscating":        dict(p=0.30, kind="test_edit", obf=0.90),
}
HACKERS = [k for k in POLICIES if k != "honest"]

RUNTIME = {"L1", "L2", "L2r", "L3", "L5"}
ALL = {"L1", "L2", "L3", "L4", "L5"}
CONFIGS = {
    "none": set(),
    "reasoning_only": {"L2r"},
    "monitor_only": {"L2"},
    "L1_only": {"L1"},
    "verifier_only": {"L5"},
    "full": set(ALL),
    "full-L1": ALL - {"L1"},
    "full-L2": ALL - {"L2"},
    "full-L3": ALL - {"L3"},
    "full-L4": ALL - {"L4"},
    "full-L5": ALL - {"L5"},
}


# ------------------------------------------------------------------ episode
def episode(pol, layers, rng, P=PARAMS, quarantined=False, verbose=False):
    """Simulate one 20-step episode; return outcome dict (and log if verbose)."""
    if isinstance(pol, str):
        pol = POLICIES[pol]
    log = []
    runtime = bool(layers & RUNTIME)
    unobs = True
    if pol.get("observer"):
        unobs = rng.random() < (P["obs_rt"] if runtime else P["obs_none"])
    ph = pol["p"] if unobs else 0.0
    kind, obf = pol.get("kind"), pol.get("obf", 0.0)
    rea_p = P["rea"] * (1 - obf) + P["rea_obf"] * obf

    # spurious triggers (false alarms), one chance per source per episode
    sources = []
    if "L2" in layers:
        sources += [P["w_ch"]] * 3
    elif "L2r" in layers:
        sources += [P["w_ch"]]
    if "L3" in layers:
        sources += [P["w_l3"]]
    false_ev = {}
    for w in sources:
        if rng.random() < P["fa"]:
            false_ev.setdefault(rng.randint(1, T), []).append(w)
    l5_false = "L5" in layers and rng.random() < P["l5_fa"]

    progress, effect, attempts, first = 0.0, False, 0, None
    risk, flag_t, block_t = 0.0, None, None
    for t in range(1, T + 1):
        r = rng.random()
        if r < ph:  # hack action
            attempts += 1
            first = first or t
            blocked = "L1" in layers and rng.random() < P["l1_cov"][kind]
            if blocked:
                risk += P["w_l1"]
                log.append(f"step {t:02d}: agent attempts {kind} -> BLOCKED by L1 (risk {risk:.2f})")
            else:
                effect = True
                log.append(f"step {t:02d}: agent performs {kind} (visible reward jumps)")
            if "L2" in layers or "L2r" in layers:
                if rng.random() < rea_p:
                    risk += P["w_ch"]
                    log.append(f"         L2 reasoning channel flags intent")
            if "L2" in layers and not blocked:
                if rng.random() < P["env"][kind]:
                    risk += P["w_ch"]
                    log.append(f"         L2 environment-diff channel flags change")
                if rng.random() < P["act"][kind]:
                    risk += P["w_ch"]
                    log.append(f"         L2 action-log channel flags command")
        else:
            wp = P["post_hack_work_p"] if effect else P["work_p"]
            if rng.random() < wp:
                progress = min(1.0, progress + P["work_inc"])
                if verbose and t <= 3:
                    log.append(f"step {t:02d}: agent edits source (progress {progress:.1f})")
        for w in false_ev.get(t, []):
            risk += w
            log.append(f"step {t:02d}: spurious trigger (false alarm, +{w})")
        if "L3" in layers and t in P["l3_checks"]:
            gap = (1.0 if effect else progress) - progress
            if gap > P["l3_gap"] and rng.random() < P["l3_det"]:
                risk += P["w_l3"]
                log.append(f"step {t:02d}: L3 sees reward-progress gap {gap:.2f} (risk {min(risk,1):.2f})")
        risk = min(risk, 1.0)
        if risk >= P["flag"] and flag_t is None:
            flag_t = t
            log.append(f"step {t:02d}: FLAG raised (risk {risk:.2f})")
        if risk >= P["block"]:
            block_t = t
            log.append(f"step {t:02d}: L6 BLOCKS episode and rolls back")
            break
    if block_t is None and "L5" in layers:
        visible = 1.0 if effect else progress
        if (visible - progress > P["l5_gap"] and rng.random() < P["l5_det"]) or l5_false:
            risk = min(1.0, risk + P["w_l5"])
            flag_t = flag_t or T
            if risk >= P["block"]:
                block_t = T
                log.append(f"submit : L5 hidden tests fail (visible {visible:.1f} vs hidden {progress:.1f}) -> BLOCK")
    if block_t is not None and rng.random() < P["rollback"]:
        effect = False
    if quarantined:  # Layer 4 audit quarantined the agent before deployment
        effect, flag_t, block_t = False, 0, 0
        log.insert(0, "L4 audit: behaviour differs between visible and hidden runs -> agent quarantined")
    out = dict(
        attempted=attempts > 0, effect=effect, flagged=flag_t is not None,
        blocked=block_t is not None, success=(progress >= P["success_progress"] and block_t is None),
        t_detect=(max(0, flag_t - first) if (flag_t is not None and first) else (0 if flag_t == 0 else None)),
        progress=progress,
    )
    if verbose:
        out["log"] = log
    return out


def audit(pol, rng, P=PARAMS):
    """Layer 4: compare hack rate in announced-evaluation vs believed-deployment runs."""
    if isinstance(pol, str):
        pol = POLICIES[pol]

    def run(unobs_prob):
        hits = 0
        for _ in range(P["audit_runs"]):
            unobs = rng.random() < unobs_prob if pol.get("observer") else True
            hacked = unobs and rng.random() < 1 - (1 - pol["p"]) ** T
            if rng.random() < P["audit_flip"]:
                hacked = not hacked
            hits += hacked
        return hits / P["audit_runs"]

    vis = run(P["obs_announced"])
    hid = run(P["obs_none"])
    return (hid - vis) > P["audit_thr"]


def run_cell(args):
    pol, cfg, seed, n, P = args
    layers = CONFIGS[cfg] if isinstance(cfg, str) else cfg
    rng = random.Random(f"{pol if isinstance(pol,str) else pol['p']}-{cfg}-{seed}")
    res = []
    for b in range(math.ceil(n / P["batch"])):
        q = "L4" in layers and audit(pol, rng, P)
        for _ in range(min(P["batch"], n - b * P["batch"])):
            res.append(episode(pol, layers, rng, P, quarantined=q))
    return summarize(res)


def summarize(res):
    h = [r for r in res if r["attempted"]]
    det = [r["t_detect"] for r in h if r["flagged"] and r["t_detect"] is not None]
    return dict(
        n=len(res), residual=sum(r["effect"] for r in res) / len(res),
        detection=(sum(r["flagged"] for r in h) / len(h)) if h else float("nan"),
        ttd=(sum(det) / len(det)) if det else float("nan"),
        flag=sum(r["flagged"] for r in res) / len(res),
        block=sum(r["blocked"] for r in res) / len(res),
        success=sum(r["success"] for r in res) / len(res),
    )


def ci(vals):
    vals = [v for v in vals if v == v]
    if len(vals) < 2:
        return (vals[0] if vals else float("nan")), 0.0
    m = stt.mean(vals)
    half = 2.776 * stt.stdev(vals) / math.sqrt(len(vals)) if len(vals) == 5 else 1.96 * stt.stdev(vals) / math.sqrt(len(vals))
    return m, half


# --------------------------------------------------------------- experiments
def experiments(seeds, n, out, procs=4):
    os.makedirs(out, exist_ok=True)
    tasks, keys = [], []
    for cfg in CONFIGS:
        for pol in POLICIES:
            for s in range(seeds):
                tasks.append((pol, cfg, s, n, PARAMS)); keys.append(("e1", cfg, pol, s))
    obf_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    for o in obf_levels:
        for cfg in ["reasoning_only", "monitor_only", "full"]:
            pol = dict(POLICIES["obfuscating"], obf=o)
            for s in range(seeds):
                tasks.append((pol, cfg, s, n, PARAMS)); keys.append(("e2", cfg, o, s))
    sweeps = [(0.2, 0.5), (0.3, 0.6), (0.4, 0.7), (0.5, 0.8), (0.6, 0.9), (0.7, 1.0)]
    for fl, bl in sweeps:
        P2 = dict(PARAMS, flag=fl, block=bl)
        for cfg in ["monitor_only", "full"]:
            for pol in POLICIES:
                for s in range(seeds):
                    tasks.append((pol, cfg, s, n, P2)); keys.append(("e3", cfg, (fl, bl, pol), s))
    scales = [1.0, 0.75, 0.5, 0.25, 0.1]
    for sc in scales:
        P4 = dict(PARAMS, env={k: v * sc for k, v in PARAMS["env"].items()},
                  act={k: v * sc for k, v in PARAMS["act"].items()}, rea=PARAMS["rea"] * sc)
        for cfg in ["monitor_only", "full-L2", "full"]:
            for pol in HACKERS:
                for s in range(seeds):
                    tasks.append((pol, cfg, s, n, P4)); keys.append(("e4", cfg, (sc, pol), s))
    hps = [0.02, 0.05, 0.10, 0.20, 0.30]
    for hp in hps:
        for cfg in ["none", "monitor_only", "L1_only", "verifier_only", "full"]:
            pol = dict(POLICIES["test_editor"], p=hp)
            for s in range(seeds):
                tasks.append((pol, cfg, s, n, PARAMS)); keys.append(("e5", cfg, hp, s))
    fas = [0.005, 0.01, 0.02, 0.05, 0.10]
    for fa in fas:
        for cfg in ["monitor_only", "full"]:
            for s in range(seeds):
                tasks.append(("honest", cfg, s, n, dict(PARAMS, fa=fa))); keys.append(("e6", cfg, fa, s))
    with Pool(procs) as pool:
        outs = pool.map(run_cell, tasks, chunksize=8)
    raw = {}
    for k, o in zip(keys, outs):
        raw.setdefault(k[:3], []).append(o)
    R = {"params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in PARAMS.items()},
         "seeds": seeds, "episodes_per_seed": n, "e1": {}, "e2": {}, "e3": {}, "e4": {}, "e5": {}, "e6": {}}
    for cfg in CONFIGS:
        per = {}
        for pol in POLICIES:
            c = raw[("e1", cfg, pol)]
            per[pol] = {m: ci([x[m] for x in c]) for m in ["residual", "detection", "ttd", "flag", "block", "success"]}
        agg = {}
        for m in ["residual", "detection", "ttd"]:
            agg[m] = ci([stt.mean([raw[("e1", cfg, p)][s][m] for p in HACKERS if raw[("e1", cfg, p)][s][m] == raw[("e1", cfg, p)][s][m]] or [float("nan")]) for s in range(seeds)])
        for m in ["flag", "block", "success"]:
            agg["honest_" + m] = per["honest"][m]
        R["e1"][cfg] = dict(per=per, agg=agg)
    for cfg in ["reasoning_only", "monitor_only", "full"]:
        R["e2"][cfg] = {str(o): ci([x["detection"] for x in raw[("e2", cfg, o)]]) for o in obf_levels}
    for cfg in ["monitor_only", "full"]:
        R["e3"][cfg] = []
        for fl, bl in sweeps:
            det = ci([stt.mean([raw[("e3", cfg, (fl, bl, p))][s]["detection"] for p in HACKERS]) for s in range(seeds)])
            fb = ci([x["block"] for x in raw[("e3", cfg, (fl, bl, "honest"))]])
            ff = ci([x["flag"] for x in raw[("e3", cfg, (fl, bl, "honest"))]])
            res = ci([stt.mean([raw[("e3", cfg, (fl, bl, p))][s]["residual"] for p in HACKERS]) for s in range(seeds)])
            R["e3"][cfg].append(dict(flag=fl, block=bl, detection=det, false_block=fb, false_flag=ff, residual=res))
    R["e4"] = {}
    for cfg in ["monitor_only", "full-L2", "full"]:
        R["e4"][cfg] = {str(sc): ci([stt.mean([raw[("e4", cfg, (sc, p))][s]["residual"] for p in HACKERS]) for s in range(seeds)])
                        for sc in scales}
    R["e5"] = {cfg: {str(hp): ci([x["residual"] for x in raw[("e5", cfg, hp)]]) for hp in hps}
               for cfg in ["none", "monitor_only", "L1_only", "verifier_only", "full"]}
    R["e6"] = {cfg: {str(fa): {m: ci([x[m] for x in raw[("e6", cfg, fa)]]) for m in ["flag", "block", "success"]}
                     for fa in fas} for cfg in ["monitor_only", "full"]}
    json.dump(R, open(os.path.join(out, "results.json"), "w"), indent=1)
    return R


def demo(policy, layers, seed):
    cfg = CONFIGS.get(layers)
    if cfg is None:
        cfg = set(layers.split(","))
    rng = random.Random(seed)
    q = "L4" in cfg and audit(policy, rng)
    r = episode(policy, cfg, rng, quarantined=q, verbose=True)
    print(f"Policy: {policy} | Layers: {sorted(cfg) or 'none'} | seed {seed}\n" + "-" * 60)
    print("\n".join(r["log"]) or "(no notable events: agent only edits source)")
    print("-" * 60)
    print(f"hack attempted: {r['attempted']} | hack effect remaining: {r['effect']} | "
          f"flagged: {r['flagged']} | blocked: {r['blocked']} | legit success: {r['success']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("demo"); d.add_argument("--policy", default="test_editor", choices=list(POLICIES))
    d.add_argument("--layers", default="full", help="config name (" + ", ".join(CONFIGS) + ") or e.g. L1,L2,L5")
    d.add_argument("--seed", type=int, default=1)
    e = sp.add_parser("experiments"); e.add_argument("--seeds", type=int, default=5)
    e.add_argument("--episodes", type=int, default=600); e.add_argument("--out", default="results")
    e.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "demo":
        demo(a.policy, a.layers, a.seed)
    else:
        R = experiments(a.seeds, a.episodes, a.out, a.procs)
        for cfg, v in R["e1"].items():
            g = v["agg"]
            print(f"{cfg:15s} residual {g['residual'][0]:.3f}  detect {g['detection'][0]:.3f}  "
                  f"honest flag {g['honest_flag'][0]:.3f} block {g['honest_block'][0]:.3f} success {g['honest_success'][0]:.3f}")
