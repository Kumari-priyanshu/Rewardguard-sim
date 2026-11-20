"""RewardGuard-Sim: interactive Streamlit front end for rewardguard_sim.py."""
import math
import random

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import rewardguard_sim as rg

AUDIT_BATCH = 50  # L4 audits once per batch of 50 episodes (as in the paper)

# Two-sided 95% Student-t critical values by degrees of freedom
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
       6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}

# label -> (key, colormap, higher_is_better)
METRICS = {
    "Residual hack rate": ("residual", "Reds", False),
    "Detection rate": ("detection", "Greens", True),
    "Flag rate": ("flag", "Oranges", False),
    "Block rate": ("block", "Purples", False),
    "Legitimate success": ("success", "Blues", True),
}

ALL_POLICIES = list(rg.POLICIES.keys())
ALL_CONFIGS = list(rg.CONFIGS.keys())

st.set_page_config(page_title="RewardGuard-Sim", page_icon="🛡️", layout="wide")


# ---------------------------------------------------------------- helpers
def simulate(pol, cfg, seed, episodes, verbose=False):
    """Run `episodes` episodes; L4 audit runs once per batch of AUDIT_BATCH."""
    rng = random.Random(f"{pol}-{cfg}-{seed}")
    layers = rg.CONFIGS[cfg]
    outcomes, quarantined = [], False
    for i in range(episodes):
        if i % AUDIT_BATCH == 0:
            quarantined = "L4" in layers and bool(rg.audit(pol, rng))
        outcomes.append(
            rg.episode(pol, layers, rng, quarantined=quarantined, verbose=verbose)
        )
    return outcomes


@st.cache_data(show_spinner=False)
def cell_metrics(pol, cfg, seed, episodes):
    outcomes = simulate(pol, cfg, seed, episodes)
    s = rg.summarize(outcomes)
    flag = sum(bool(o["flagged"]) for o in outcomes) / len(outcomes)
    return {
        "residual": s.get("residual"),
        "detection": s.get("detection"),
        "block": s.get("block"),
        "success": s.get("success"),
        "flag": flag,
    }


def aggregate(values):
    """Mean and 95% CI half-width across seeds (None values ignored)."""
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return float("nan"), 0.0
    n = len(vals)
    mean = sum(vals) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    t = T95.get(n - 1, 2.0 if n - 1 < 30 else 1.96)
    return mean, t * math.sqrt(var / n)


def fmt(m, ci):
    if math.isnan(m):
        return "–"
    return f"{m * 100:.1f}" + (f" ± {ci * 100:.1f}" if ci else "")


def heatmap(pivot, title, cmap):
    fig, ax = plt.subplots(
        figsize=(1.5 * len(pivot.columns) + 3, 0.55 * len(pivot) + 2)
    )
    im = ax.imshow(pivot.values.astype(float), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not math.isnan(v):
                ax.text(j, i, f"{v * 100:.1f}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- header
st.title("🛡️ RewardGuard-Sim")
st.subheader("Interactive Reward Hacking Defense Simulation")
st.warning(
    "Educational toy simulation. Agent behaviours are scripted, "
    "and detection probabilities are assumptions."
)

with st.sidebar:
    st.header("Global settings")
    seed = st.number_input("Base random seed", min_value=0, value=3)
    episodes = st.slider("Episodes per cell", 50, 1000, 200, 50)
    n_seeds = st.slider("Seeds (for 95% confidence intervals)", 1, 10, 3)
    st.caption(
        "With more than one seed, results are means over seeds with "
        "95% CIs (Student's t)."
    )

tab_single, tab_compare, tab_explore, tab_about = st.tabs(
    ["Single run", "Compare defenses", "Episode explorer", "About"]
)

# ---------------------------------------------------------------- single run
with tab_single:
    c1, c2 = st.columns(2)
    default_pol = ALL_POLICIES.index("test_editor") if "test_editor" in ALL_POLICIES else 0
    default_cfg = ALL_CONFIGS.index("full") if "full" in ALL_CONFIGS else 0
    s_pol = c1.selectbox("Agent policy", ALL_POLICIES, index=default_pol, key="s_pol")
    s_cfg = c2.selectbox("Defense configuration", ALL_CONFIGS, index=default_cfg, key="s_cfg")
    st.caption("Active layers: " + (", ".join(sorted(rg.CONFIGS[s_cfg])) or "none"))

    if st.button("Run single simulation"):
        rng = random.Random(int(seed))
        layers = rg.CONFIGS[s_cfg]
        quarantined = "L4" in layers and bool(rg.audit(s_pol, rng))
        result = rg.episode(s_pol, layers, rng, quarantined=quarantined, verbose=True)
        st.session_state["single"] = (s_pol, s_cfg, quarantined, result)

    if "single" in st.session_state:
        pol, cfg, quarantined, result = st.session_state["single"]
        st.markdown(f"**Result for `{pol}` under `{cfg}`**")
        if quarantined:
            st.error("L4 quarantined this agent before the episode ran.")

        m = st.columns(4)
        m[0].metric("Hack attempted", str(result["attempted"]))
        m[1].metric("Flagged", str(result["flagged"]))
        m[2].metric("Blocked", str(result["blocked"]))
        m[3].metric("Legitimate success", str(result["success"]))
        st.progress(min(max(float(result["progress"]), 0.0), 1.0), text="True progress")
        st.write("Hack effect remaining:", result["effect"])

        st.subheader("Event log")
        if result.get("log"):
            st.code("\n".join(result["log"]))
            st.download_button("Download log", "\n".join(result["log"]), "episode_log.txt")
        else:
            st.info("No notable events occurred.")

# ---------------------------------------------------------------- compare
with tab_compare:
    default_pols = [p for p in ["honest", "test_editor", "hardcoder", "tamperer"] if p in ALL_POLICIES]
    default_cfgs = [c for c in ["none", "monitor_only", "L1_only", "verifier_only", "full"] if c in ALL_CONFIGS]
    pols = st.multiselect("Agent policies", ALL_POLICIES, default=default_pols)
    cfgs = st.multiselect("Defense configurations", ALL_CONFIGS, default=default_cfgs)

    if st.button("Run comparison", disabled=not (pols and cfgs)):
        rows, total, done = [], len(pols) * len(cfgs) * n_seeds, 0
        bar = st.progress(0, text="Simulating...")
        for cfg in cfgs:
            for pol in pols:
                per_seed = []
                for k in range(n_seeds):
                    per_seed.append(cell_metrics(pol, cfg, int(seed) + k, episodes))
                    done += 1
                    bar.progress(done / total, text=f"Simulating... {done}/{total}")
                row = {"Defense": cfg, "Agent Policy": pol}
                for _, (key, _, _) in METRICS.items():
                    mean, ci = aggregate([d[key] for d in per_seed])
                    row[key], row[key + "_ci"] = mean, ci
                rows.append(row)
        bar.empty()
        st.session_state["compare"] = (pd.DataFrame(rows), episodes, n_seeds)

    if "compare" in st.session_state:
        df, ep_used, seeds_used = st.session_state["compare"]
        st.caption(
            f"{ep_used} episodes x {seeds_used} seed(s) per cell. "
            "Values in %, mean ± 95% CI."
        )

        shown = df[["Defense", "Agent Policy"]].copy()
        for label, (key, _, _) in METRICS.items():
            shown[label] = [fmt(m, c) for m, c in zip(df[key], df[key + "_ci"])]
        st.dataframe(shown, use_container_width=True)

        label = st.selectbox("Metric to chart", list(METRICS.keys()))
        key, cmap, _ = METRICS[label]
        pivot = df.pivot(index="Defense", columns="Agent Policy", values=key)
        ci_pivot = df.pivot(index="Defense", columns="Agent Policy", values=key + "_ci")

        view = st.radio("Chart type", ["Heatmap", "Grouped bars"], horizontal=True)
        if view == "Heatmap":
            fig = heatmap(pivot, label, cmap)
        else:
            hackers = [c for c in pivot.columns if c != "honest"] or list(pivot.columns)
            fig, ax = plt.subplots(figsize=(10, 5))
            pivot[hackers].plot(kind="bar", ax=ax, yerr=ci_pivot[hackers], capsize=3)
            ax.set_ylabel(label)
            ax.set_ylim(0, 1)
            ax.set_title(f"{label} by defense")
            plt.xticks(rotation=0)
            fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        if "honest" in set(df["Agent Policy"]):
            st.subheader("Cost to the honest agent")
            hon = df[df["Agent Policy"] == "honest"].set_index("Defense")
            cost = pd.DataFrame({
                "Flag rate %": hon["flag"] * 100,
                "False-block rate %": hon["block"] * 100,
                "Success retained %": hon["success"] * 100,
            }).round(1)
            st.dataframe(cost, use_container_width=True)
            st.caption("More layers add more chances for a false alarm; calibrate each detector.")

        st.download_button("Download results CSV", df.to_csv(index=False), "rewardguard_results.csv", "text/csv")

# ---------------------------------------------------------------- explorer
with tab_explore:
    c1, c2, c3 = st.columns(3)
    e_pol = c1.selectbox("Agent policy", ALL_POLICIES, key="e_pol")
    e_cfg = c2.selectbox("Defense configuration", ALL_CONFIGS, index=default_cfg, key="e_cfg")
    e_n = c3.slider("Episodes", 20, 200, 100, 20, key="e_n")

    if st.button("Run episodes"):
        st.session_state["explore"] = (e_pol, e_cfg, simulate(e_pol, e_cfg, int(seed), e_n, verbose=True))

    if "explore" in st.session_state:
        pol, cfg, outs = st.session_state["explore"]
        table = pd.DataFrame([
            {"episode": i, **{k: o.get(k) for k in ("attempted", "flagged", "blocked", "success", "progress", "effect")}}
            for i, o in enumerate(outs)
        ])
        st.markdown(f"**{len(outs)} episodes of `{pol}` under `{cfg}`**")
        st.dataframe(table, use_container_width=True, height=250)

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(table["progress"].astype(float), bins=10, range=(0, 1))
        ax.set_xlabel("Final true progress")
        ax.set_ylabel("Episodes")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        idx = st.number_input("Inspect episode", 0, len(outs) - 1, 0)
        log = outs[int(idx)].get("log")
        if log:
            st.code("\n".join(log))
        else:
            st.info("No notable events in this episode.")
        st.download_button("Download episode table", table.to_csv(index=False), "episodes.csv", "text/csv")

# ---------------------------------------------------------------- about
with tab_about:
    st.markdown("""
### Defense layers

| Layer | Role |
| --- | --- |
| **L1** | Capability control (write-protects test and reward files) |
| **L2** | Trajectory monitoring (reasoning, action log, environment diffs) |
| **L3** | Reward-progress gap detection |
| **L4** | Behavioural audit (observed vs believed-unobserved runs) |
| **L5** | Independent hidden-test verification |
| **L6** | Risk-based flagging, blocking and rollback |
""")
    st.markdown("**Configurations available in this build:**")
    st.dataframe(
        pd.DataFrame(
            [{"Configuration": c, "Layers": ", ".join(sorted(rg.CONFIGS[c])) or "none"} for c in ALL_CONFIGS]
        ),
        use_container_width=True,
    )

st.divider()
st.caption(
    "This toy simulation uses scripted policies and assumed probabilities. "
    "It is not a real-world LLM experiment."
)
