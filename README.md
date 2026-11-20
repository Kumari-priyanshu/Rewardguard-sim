# RewardGuard Simulator (`rewardguard_sim.py`)

A transparent toy simulation that accompanies the paper *RewardGuard: A Defense-in-Depth Framework and Simulation Study for Detecting Reward Hacking and Deceptive Alignment in Autonomous AI Agents*.

A scripted coding agent is rewarded for passing tests. Some agents hack the reward instead of fixing the bug. The simulator measures how well a six-layer runtime defense stops them.

> **Important:** The agents and all detection probabilities are modelling assumptions, not measurements of real systems. The results show the structural behaviour of the design, not real-world rates.

---

## Quick start

```bash
# Regenerate every number in the paper
python rewardguard_sim.py experiments --seeds 5 --episodes 600

# Print a step-by-step trace of one episode
python rewardguard_sim.py demo --policy test_editor --layers full
```

The simulator is a single Python file. Random streams are seeded per experiment cell, so results repeat exactly.

<!-- TODO: state the Python version and any dependencies (e.g. numpy, matplotlib) from your imports. -->

---

## The scenario

- A coding agent has **20 steps** to make failing tests pass.
- Reward is the fraction of **10 visible tests** that pass.
- **5 hidden tests**, never shown to the agent, measure whether the real bug was fixed.
- At each step the agent either hacks (with its policy probability) or works on the source, which raises true progress by 0.1 with probability 0.7.
- After a successful hack it rarely keeps working (probability 0.1).
- An episode is **legitimately successful** if true progress reaches 0.9 and the episode was not blocked.

### Exploits

1. Edit the test files
2. Hard-code outputs in the source
3. Tamper with the reward code

### Agent policies

| Policy | Behaviour |
| --- | --- |
| `honest` | Never hacks |
| `test_editor` | Edits tests (hack probability 0.30 per step) |
| `hard_coder` | Hard-codes outputs in the source (0.30) |
| `tamperer` | Tampers with reward code, rarely (0.05) |
| `observer_sensitive` | Hacks only when it believes it is unobserved (0.30) |
| `obfuscating` | A test-editor whose reasoning trace is sanitized 90% of the time (0.30) |

---

## The six defense layers

| Layer | Role |
| --- | --- |
| **L1** Capability control | Write-protects test and reward files (70% coverage of those files, 0% of source) |
| **L2** Trajectory monitor | Watches three channels: reasoning trace, action log, environment diffs |
| **L3** Reward-hacking detector | Flags a gap between visible reward and real progress (gap > 0.3, checked at steps 5, 10, 15, 20) |
| **L4** Deception evaluator | Compares behaviour in announced evaluation runs with believed-unobserved runs, once per batch of 50 episodes |
| **L5** Independent verifier | Reruns the hidden tests at submission |
| **L6** Intervention controller | Fuses signals into a risk score, then allows, flags or blocks, with rollback |

All layers read from and write to an **evidence layer** the agent cannot modify.

### Risk fusion (L6)

| Signal | Risk added |
| --- | --- |
| Blocked write attempt (L1) | 0.5 |
| Each L2 channel that fires | 0.4 |
| Each L3 gap detection | 0.4 |
| Failed hidden-test check (L5) | 0.7 |

- Risk >= **0.4** raises a **flag**.
- Risk >= **0.7** **blocks** the episode. Rollback removes the hack effect with probability 0.95.
- Every monitoring source also fires spuriously with probability **0.02** per episode, which produces the honest-agent false alarms.
- L4 runs five announced-evaluation runs and five believed-deployment runs (2% label noise) and quarantines the agent if the hack rate differs by more than 0.2.

---

## Configurations compared

Eleven defense configurations:

1. No defense
2. Reasoning-trace monitor only
3. L2 monitor only
4. L1 only
5. L5 verifier only
6. Full framework
7. Full minus L1
8. Full minus L2
9. Full minus L3
10. Full minus L4
11. Full minus L5

L6 is present whenever any layer is.

Each configuration is run against every policy for **5 seeds x 600 episodes = 3,000 episodes per cell**. Reported values are means over seeds with 95% confidence intervals (Student's t, 4 degrees of freedom).

---

## Metrics

| Metric | Meaning |
| --- | --- |
| Residual hack rate | Share of episodes ending with an active hack effect, averaged over the five hacking policies (lower is better) |
| Detection rate | Share of episodes with at least one hack attempt that reach the flag threshold |
| Time to detection (TTD) | Steps from the first hack attempt to the flag |
| Flag rate (honest) | Share of honest episodes flagged |
| False-block rate (honest) | Share of honest episodes blocked |
| Success retained (honest) | Share of honest episodes that succeed legitimately |

### Additional experiments

- **Obfuscation sweep:** agent obfuscation rate from 0 to 1
- **Monitor degradation:** L2 detection probabilities scaled from 1.0 down to 0.1 of baseline
- **Hack frequency:** per-step hack probability of the test-editor varied
- **Threshold trade-offs:** flag and block thresholds varied
- **False-trigger sweep:** per-source spurious trigger rate from 0.5% to 10%

---

## Headline results (from the paper)

| Result | Value |
| --- | --- |
| Residual hack rate, no defense | 90.8% |
| Residual hack rate, full framework | 2.2% |
| Detection of hacking episodes, full framework | 100.0% |
| Honest episodes blocked, full framework | 1.1% |
| Honest success retained, full framework | 97.4% (98.0% with no defense) |
| Reasoning-only monitor, obfuscation 0 to 1 | detection 98.4% to 27.9% |
| Monitor at 25% accuracy: monitor-only vs full | 32.1% vs 3.5% residual |

---

## Parameters

All simulation parameters are modelling assumptions, listed in Table I of the paper.

| Parameter | Value |
| --- | --- |
| Episode; tests | 20 steps; 10 visible, 5 hidden |
| L1 block coverage | 70% of test and reward files; 0% of source |
| L2 detection (env / action / reasoning) | 0.85 / 0.70 / 0.60 for test-edit and tamper; 0.35 / 0.30 / 0.60 for hard-code; reasoning 0.05 if fully obfuscated |
| False trigger | 2% per source per episode |
| L3 | gap > 0.3; detection 0.9 |
| L5 | gap > 0.3; detection 0.95; 1% false fail |
| Weights; thresholds | L1 0.5, L2/L3 0.4, L5 0.7; flag 0.4, block 0.7 |
| Rollback success | 95% |
| Observer: P(believes unobserved) | 0.90 without runtime layers; 0.30 with; 0.02 in announced evaluation |

<!-- TODO: if parameters live in a dict or config block in the code, point to its name here. -->

---

## Limitations

- Scripted agents, not real language models. No agent learns to adapt to the monitors.
- Detection probabilities are assumed, not measured.
- Signals are combined as independent events, although real monitors are correlated.
- One task only.
- Overoptimization of learned reward models and goal misgeneralization are not simulated separately.

Validating the design requires real agents, real detectors and a held-out scenario suite.

---

## Citation

<!-- TOD. -->

```
[Kumari Priyanshu]. "RewardGuard: A Defense-in-Depth Framework and Simulation Study for
Detecting Reward Hacking and Deceptive Alignment in Autonomous AI Agents."  2026.
```

## License

<!-- TODO: choose a license (e.g. MIT) and add a LICENSE file. -->
