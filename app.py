
import random
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import rewardguard_sim as rg

st.set_page_config(
    page_title="RewardGuard-Sim",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ RewardGuard-Sim")
st.subheader("Interactive Reward Hacking Defense Simulation")

st.warning(
    "Educational toy simulation. Agent behaviours are scripted, "
    "and detection probabilities are assumptions."
)

with st.sidebar:
    st.header("Simulation Controls")

    policy = st.selectbox(
        "Agent Policy", list(rg.POLICIES.keys()),
        index=1
    )

    config = st.selectbox(
        "Defense Configuration", list(rg.CONFIGS.keys()),
        index=list(rg.CONFIGS.keys()).index("full")
    )

    seed = st.number_input(
        "Random Seed", min_value=0, value=3
    )

    episodes = st.slider(
        "Episodes per configuration", 50, 500, 200, 50
    )

    single = st.button("Run Single Simulation")
    compare = st.button("Compare Defense Layers")

st.markdown("""
### Defense Layers

- **L1:** Capability control
- **L2:** Trajectory monitoring
- **L3:** Reward-progress gap detection
- **L4:** Behavioural audit
- **L5:** Independent hidden-test verification
- **L6:** Risk-based blocking and rollback
""")

if single:
    rng = random.Random(int(seed))
    layers = rg.CONFIGS[config]

    quarantined = (
        "L4" in layers and rg.audit(policy, rng)
    )

    result = rg.episode(
        policy,
        layers,
        rng,
        quarantined=quarantined,
        verbose=True
    )

    st.subheader("Simulation Event Log")

    if result.get("log"):
        st.code("\n".join(result["log"]))
    else:
        st.info("No notable events occurred.")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Hack Attempted", str(result["attempted"]))
    c2.metric("Flagged", str(result["flagged"]))
    c3.metric("Blocked", str(result["blocked"]))
    c4.metric("Legitimate Success", str(result["success"]))

    st.write("Progress:", round(result["progress"], 2))
    st.write("Hack Effect Remaining:", result["effect"])

if compare:
    rows = []
    policies = [
        "honest", "test_editor", "hardcoder", "tamperer"
    ]
    configs = [
        "none", "monitor_only", "L1_only",
        "verifier_only", "full"
    ]

    progress = st.progress(0)
    total = len(policies) * len(configs)
    completed = 0

    for cfg in configs:
        for pol in policies:
            rng = random.Random(
                f"{pol}-{cfg}-{int(seed)}"
            )
            layers = rg.CONFIGS[cfg]
            outcomes = []

            for _ in range(episodes):
                quarantined = (
                    "L4" in layers and rg.audit(pol, rng)
                )
                outcomes.append(
                    rg.episode(
                        pol, layers, rng,
                        quarantined=quarantined
                    )
                )

            summary = rg.summarize(outcomes)

            rows.append({
                "Defense": cfg,
                "Agent Policy": pol,
                "Residual Hack Rate": summary["residual"],
                "Detection Rate": summary["detection"],
                "Blocked Rate": summary["block"],
                "Legitimate Success": summary["success"]
            })

            completed += 1
            progress.progress(completed / total)

    df = pd.DataFrame(rows)

    st.subheader("Comparison Results")
    st.dataframe(df.round(3), use_container_width=True)

    hackers = df[
        df["Agent Policy"].isin(
            ["test_editor", "hardcoder", "tamperer"]
        )
    ]

    pivot = hackers.pivot(
        index="Defense",
        columns="Agent Policy",
        values="Residual Hack Rate"
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax)

    ax.set_title("Residual Hacking Effects by Defense")
    ax.set_ylabel("Residual Hack Rate")
    ax.set_xlabel("Defense Configuration")
    ax.set_ylim(0, 1)

    plt.xticks(rotation=0)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.download_button(
        "Download Results CSV",
        df.to_csv(index=False),
        "rewardguard_results.csv",
        "text/csv"
    )

st.divider()

st.caption(
    "This toy simulation uses scripted policies and assumed "
    "probabilities. It is not a real-world LLM experiment."
)
