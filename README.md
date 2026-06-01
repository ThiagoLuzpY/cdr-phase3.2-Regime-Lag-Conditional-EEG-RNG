# Core Distinguishability Relativity (CDR)


## Status

- **Phase I** ✅ **COMPLETE** (7/7 gates PASS)
- **Phase II.1A** (Empirical validation – energy systems) ✅ **COMPLETE**
- **Phase II.1B** (Empirical validation – neurodynamics) ✅ **COMPLETE**
- **Phase II.2** (Human mobility systems) ✅ **COMPLETE**
- **Phase II.3** (Ecological population dynamics) ✅ **COMPLETE**
- **Phase II.4** (Protein dynamics) ✅ **COMPLETE**
- **Phase III.0.1** (EEG-only experimental neural validation) ✅ **COMPLETE**
- **Phase III.0.2** (RNG-only experimental stochastic baseline validation) ✅ **COMPLETE**
- **Phase III.0.3** (Joint EEG + RNG validation) ✅ **COMPLETE**
- **Phase III.1** (Advanced multi-subject EEG + RNG redesign with latent and quantum-aware proxies) ✅ **COMPLETE — CLEAN NULL RESULT**
- **Phase III.2** (Regime-aware, lagged and conditional EEG–RNG validation) 📋 **PLANNED**

---

## What is CDR?

**Core Distinguishability Relativity (CDR)** is a pre-registered framework designed to detect information-driven selection bias in observed dynamics without falling into common statistical pitfalls such as p-hacking, circular reasoning, or model over-flexibility.

The framework focuses on answering a fundamental scientific question:

> *When we observe a dynamic system, how do we know whether a detected pattern is a real causal effect or merely an artifact of noise or modeling assumptions?*

CDR addresses this through:

- Pre-registered hypotheses  
- Mandatory validation gates  
- Adversarial and structural controls  
- Out-of-sample generalization tests  

Instead of relying solely on p-values, CDR requires multiple orthogonal validation gates to pass before any claim can be considered detectable.

---


## Current empirical status

The framework has now been tested across macroscopic, biological, neural, stochastic and joint experimental domains. 

The current empirical status is: 

- CDR successfully detects injected structure when the representation is adequate 
- controls collapse cleanly across all validated phases 
- neural domains show low but non-zero residual structure 
- RNG-only behaves as a clean stochastic null baseline 
- simple and advanced EEG + RNG joint models have not yet produced robust cross-domain coupling 
- Phase III.1 produced a clean null_result under stricter multi-subject, latent and quantum-aware validation 


The next planned stage, Phase III.2, will test whether any possible EEG–RNG relationship is regime-dependent, lag-dependent, conditional, or better captured by enriched EEG features.

---

## Project Roadmap

The CDR validation program is divided into empirical phases.

| Phase                    | Objective | Status |
|--------------------------|-----------|--------|
| **Phase I**              | Toy-model validation (controlled system) | ✅ Complete |
| **Phase II.1A**          | Real-world validation on energy infrastructure | ✅ Complete |
| **Phase II.1B**          | Real-world validation on neural dynamics (fMRI) | ✅ Complete |
| **Phase II.2**           | Human mobility systems | ✅ Complete |
| **Phase II.3**           | Ecological population dynamics | ✅ Complete |
| **Phase II.4**           | Protein dynamics | ✅ Complete |
| **Phase III.0.1**        | EEG-only experimental neural validation | ✅ Complete |
| **Phase III.0.2**        | RNG-only baseline validation | ✅ Complete |
| **Phase III.0.3**        | Joint EEG + RNG validation | ✅ Complete |
| **Phase III.1** | Advanced multi-subject EEG + RNG redesign with informational, latent and quantum-aware proxy layers | ✅ Complete — clean null result | 
| **Phase III.2** | Regime-aware, lagged and conditional EEG–RNG validation | 📋 Planned |



---

## Phase I — Toy Model Validation (Completed)

Phase I validates the CDR framework in a fully controlled environment using a small enumerated system.

### Model used

- 2-component Ising conditional kernel  
- Binary state variables  
- Known ground-truth coupling parameter `ε`  

### State space
```
2 components × binary states → 4 states
```

---

### Phase I Gates

| Gate | Meaning |
|------|---------|
| **G1** | H₀ recovery |
| **G2** | H₁ recovery |
| **G3** | Control collapse |
| **G4** | Parameter identifiability |
| **G5** | Stability |
| **G6** | Adversarial robustness |
| **G7** | Out-of-sample generalization |

---

### Result
```
CDR Phase I+
────────────────
G1_H0_recovery: PASS
G2_H1_recovery: PASS
G3_controls_collapse: PASS
G4_identifiability: PASS
G5_stability: PASS
G6_adversarial: PASS
G7_out_of_sample: PASS
────────────────
FINAL: PASS
```

---

## Phase II — Empirical Validation

Phase II tests the framework on real observational systems.

The objective is to verify that the estimator:

- Detects reweighting when present
- Does not produce false positives
- Generalizes across unseen data
- Remains stable under discretization changes

---

## Phase II.1A — Energy Infrastructure Validation (Completed)

**Dataset:**
```
Open Power System Data (OPSD)
```

**Variables used:**
```
(load, wind, solar, price)
```

**State definition:**
```
state = (load_bin, wind_bin, solar_bin, price_bin)
```

**Discretization:**
```
3 bins per variable
3⁴ = 81 states
```

**Observations:**
```
8740 hourly transitions
Germany/Luxembourg grid
```

---

### Results
```
CDR Phase II.1A
────────────────
F1_injection_recovery: PASS
F2_controls_collapse: PASS
F3_holdout_generalization: PASS
F5_sensitivity: PASS
────────────────
FINAL: PASS
```

---

### Key Finding

The German electrical grid shows:
```
ε ≈ 0
```

consistent with a highly regulated infrastructure system.

---

## Phase II.1B — Neural Dynamics Validation (Completed)

**Dataset:**
```
OpenNeuro
ds002938
task: effort
subject: sub-01
```

**State construction:**
```
state = (ROI₁, ROI₂, ROI₃, ROI₄, ROI₅)
```

**Discretization:**
```
2 bins per ROI
2⁵ = 32 states
```

**Observations:**
```
661 transitions
```

---

### Phase II.1B Results
```
CDR Phase II.1B (fMRI)
────────────────────────────────

F1_injection_recovery: PASS
eps_hat: 0.0
eps_true: 0.05
abs_err: 0.05

F2_controls_collapse: PASS
median_eps_controls: 0.0
fraction_below_tol: 1.0
max_eps_controls: 0.0
n_controls: 20

F3_holdout_generalization: PASS
eps_train: 0.08
eps_test: 0.00
abs_delta: 0.08

F5_sensitivity: PASS
eps_binsA: 0.08
eps_binsB: 0.06
abs_delta: 0.02

────────────────────────────────
FINAL: PASS
```

---

## Phase II.2 — Human Mobility Validation (Completed)

This phase applies the CDR framework to **large-scale human mobility trajectories**.

---

### Dataset
```
Microsoft GeoLife GPS Trajectories
```

**Characteristics:**
```
182 users
17,000+ trajectories
~1.2 million GPS points
Sampling interval ≈ 1–5 seconds
```

**After preprocessing:**
```
user-level state trajectories
discretized spatial bins
temporal transition sequences
```

---

### System Representation

Human mobility was represented as a discrete dynamical system:
```
state = (spatial_cell_t)
```

**Transitions:**
```
s(t) → s(t+1)
```

Kernel estimated via empirical transition frequencies.

---

### Computational Complexity

This phase required substantially heavier computation than previous domains.

**Pipeline runtime:**
```
~48 hours
```

**Reasons:**

- Large trajectory dataset
- Multiple adversarial controls
- Likelihood surface estimation
- Holdout generalization checks

Control experiments alone required several hours due to repeated recomputation of transition kernels.

---

### Phase II.2 Gates

| Gate | Meaning |
|------|---------|
| **F1** | Injection recovery |
| **F2** | Control collapse |
| **F3** | Train/test generalization |
| **F5** | Discretization sensitivity |

---

### Phase II.2 Results
```
CDR Phase II.2 (Human Mobility)
────────────────────────────────

F1_injection_recovery: PASS
eps_hat: 0.30
eps_true: 0.30
abs_err: 0.00

F2_controls_collapse: PASS
median_eps_controls: 0.00
fraction_below_tol: 1.0
max_eps_controls: 0.00
n_controls: 10

F3_holdout_generalization: PASS
eps_train: 0.00
eps_test: 0.00
abs_delta: 0.00

F5_sensitivity: PASS
eps_binsA: 0.00
eps_binsB: 0.00
abs_delta: 0.00

────────────────────────────────
FINAL: PASS
```

---

### Interpretation

The mobility dynamics in the GeoLife dataset appear consistent with a Markovian mobility kernel:
```
ε ≈ 0
```

**Meaning:**

Human spatial transitions in this dataset do not require additional structural mixture beyond the empirical transition kernel.

Importantly, the estimator correctly recovered injected structure:
```
ε_true = 0.30
ε_hat = 0.30
```

confirming estimator sensitivity.

---

## Phase II Conclusions

Across **three independent empirical domains**:

| Domain | Result |
|--------|--------|
| Energy infrastructure | `ε ≈ 0` |
| Neural dynamics (fMRI) | `ε ≈ 0.06–0.08` |
| Human mobility | `ε ≈ 0` |

The CDR estimator demonstrated:

- ✅ Successful injection recovery
- ✅ Collapse under adversarial controls
- ✅ Stable holdout generalization
- ✅ Robustness to discretization

These results support the **cross-domain stability of the CDR estimation framework**.

---

# Phase II.3 — Ecological Dynamics (Completed)

🔗 **GitHub repository:**
https://github.com/ThiagoLuzpY/cdr-phase2.3-ecology

---

## Objective

Apply the CDR framework to **biological dynamical systems**, specifically:


predator-prey population dynamics


This phase introduces systems with:

- Non-linear feedback loops  
- Cyclical dynamics  
- Strong endogenous structure  

---

## Dataset

```
Hudson Bay Company Lynx–Hare dataset
```

---

## System Representation

Final state definition:

```
state = (hare_log_return, lynx_log_return)
```

---

## Discretization

```
3 bins per variable
3² = 9 states
```

---

## Key Methodological Adjustments

### 1. Dimensionality Correction

Removed:

```
year_norm
exogenous variables
```

Reason:

```
avoid sparsity
preserve endogenous structure
```

---

### 2. Temporal Strategy

Used:

```
interleaved train/test split
```

Reason:

```
ecological systems are cyclical
chronological split breaks phase consistency
```

---

### 3. Control System (Final)

```
shuffle_time
block_shuffle
species_swap
transition_randomization
```

---

## Phase II.3 Results


CDR Phase II.3 (Ecology)
────────────────────────────────
```
F1_injection_recovery: PASS
eps_hat: 0.275
eps_true: 0.25
abs_err: 0.025

F2_controls_collapse: PASS
median_eps_controls: 0.00
fraction_below_tol: 1.0

F3_holdout_generalization: PASS
eps_train: 0.00
eps_test: 0.00
abs_delta: 0.00

F5_sensitivity: PASS
eps_binsA: 0.00
eps_binsB: 0.00
abs_delta: 0.00

────────────────────────────────
FINAL: PASS
```

---

## Interpretation

```
ε ≈ 0
```

Meaning:

- System is fully explained by internal dynamics  
- No external reweighting required  
- Strong structural determinism  

---

## Scientific Insight

This confirms that:


CDR correctly identifies endogenous structure in biological systems


---

# Phase II.4 — Protein Dynamics (Completed)

🔗 **GitHub repository:**
https://github.com/ThiagoLuzpY/cdr-phase2.4-protein


## Objective

Apply the CDR framework to microscopic biological dynamical systems, specifically:

- molecular dynamics simulations
- protein folding trajectories
- conformational state transitions


This phase introduces systems with:

- Energy-landscape-governed transitions

- Strong physical constraints

- Low-dimensional conformational coordinates

- Multiple independent simulation trajectories

---

## Dataset
```
Alanine dipeptide molecular dynamics trajectories
```

Files used:
```
alanine-dipeptide-nowater.pdb
alanine-dipeptide-0-250ns-nowater.xtc
alanine-dipeptide-1-250ns-nowater.xtc
alanine-dipeptide-2-250ns-nowater.xtc
```
---
## System Representation


Protein dynamics were represented through backbone dihedral structure:
```
state = (phi_bin, psi_bin)
```
These variables capture the dominant conformational geometry of alanine dipeptide and provide a compact state-space representation of the molecular dynamics.
---

## Discretization
```
3 bins per variable
3² = 9 states
```
---

## Sampling / Preprocessing

Frame thinning:
```
frame_stride = 10
```

Effective observations loaded:
```
75,000 frames
```
The loader extracted dihedral trajectories from independent molecular dynamics simulations and assembled the conformational variables used in the CDR pipeline.

---

## Key Methodological Adjustment
```
Trajectory-Level Holdout for F3
```
The initial sequential holdout failed because independent .xtc simulations should not be treated as one single continuous trajectory.

---

### Final F3 strategy:
```
leave-one-trajectory-out holdout
```

Implemented as:

- Train on all trajectories except one held-out simulation

- Test on the held-out trajectory only

- Build transitions only within each trajectory

- Prevent artificial transitions across .xtc file boundaries


This adjustment preserved the falsifiability of the framework while aligning the holdout protocol with the physical structure of the dataset.
```
Control System
shuffle_next
circular_shift
block_shuffle
```

These controls were designed to break correct frame-to-frame pairing while preserving partial marginal or short-range structure.

---

## Phase II.4 Results
CDR Phase II.4 (Protein)
────────────────────────────────
```
F1_injection_recovery: PASS
eps_hat: 0.27
eps_true: 0.25
abs_err: 0.02

F2_controls_collapse: PASS
median_eps_controls: 0.00
fraction_below_tol: 1.0
max_eps_controls: 0.00
n_controls: 10

F3_holdout_generalization: PASS
eps_train: 0.00
eps_test: 0.10
abs_delta: 0.10

F5_sensitivity: PASS
eps_binsA: 0.00
eps_binsB: 0.00
abs_delta: 0.00

────────────────────────────────
FINAL: PASS
Interpretation
```

---

The alanine dipeptide dynamics appear consistent with a structurally sufficient physical kernel:
```
ε ≈ 0
```

Meaning:

- Molecular transitions are largely explained by the endogenous conformational dynamics

- No additional structural reweighting is required at the tested resolution

- The estimator remains sensitive, as shown by successful injected-structure recovery

- Importantly, the final result is consistent with the expectation that strongly constrained molecular systems should behave closer to energy-governed physical dynamics than to partially latent neural systems.

---

## Scientific Insight

This phase confirms that:

- CDR remains stable at the molecular scale

- and that trajectory-aware holdout design is essential when independent simulations are used as empirical test domains.

---

## Updated Cross-Domain Conclusions (Phase II + Phase III)

Across **nine empirical domains / experimental configurations**:

| Domain | Result |
|--------|--------|
| Energy infrastructure | ε ≈ 0 |
| Neural dynamics (fMRI) | ε ≈ 0.06–0.08 |
| Human mobility | ε ≈ 0 |
| Ecological systems | ε ≈ 0 |
| Protein dynamics | ε ≈ 0 |
| EEG-only | ε ≈ 0.04 |
| RNG-only | ε ≈ 0 |
| Joint EEG + RNG | ε ≈ 0.00–0.03 |
| Advanced multi-subject EEG + RNG redesign | ε_joint ≈ 0.00; clean null_result |


---

The cumulative empirical picture now shows that CDR can distinguish at least **three broad operational regimes**:

### 1. Structurally sufficient systems

```
ε ≈ 0
```

Observed in:

- Energy infrastructure
- Human mobility
- Ecological population dynamics
- Protein dynamics
- RNG-only baseline

These systems are adequately explained by the empirical reference kernel at the tested resolution.

---

### 2. Low but non-zero residual-structure systems

```
ε > 0 (low)
```

Observed in:

- fMRI
- EEG-only

These results suggest that some neural systems may retain low residual structure not fully captured by the chosen baseline kernel, while remaining stable under controls, holdout, and sensitivity tests.

---

### 3. Weak joint-structure systems

```
ε_joint ≈ 0.00–0.03
```

Observed in:

- Joint EEG + RNG validation

The joint domain did **not** show strong cross-domain structure under the present observational setup. The final compact joint representations stabilized the pipeline and eliminated major sparsity artifacts, but the observed joint epsilon remained low.

---

### 4. Advanced joint null-result systems


```
ε_joint ≈ 0
```

Observed in:

- Phase III.1 advanced multi-subject EEG + RNG redesign

The advanced joint redesign introduced:

- multi-subject EEG expansion
- informational proxy variables `I_t`
- latent inferred states `Z_t`
- quantum-aware proxy variables `Q_t`
- subject-level diagnostics
- ablation-corrected proxy controls
- BIC / complexity-penalty model selection

Despite this stronger design, the final advanced joint model did **not** reveal residual joint structure beyond the EEG-only or RNG-only baselines.

The result is therefore interpreted as a **clean null_result**, not as a technical failure.

This indicates that, under the present public observational EEG + RNG setup, the hypothesized joint EEG–RNG structure is not detectable through a global joint-state model, even after adding latent and quantum-aware proxy layers.

---

## Emerging Methodological Insight

The cumulative results also suggest that **F1 injection recovery is domain-dependent**.

A contrast emerged between:

- **high-injectability domains**, where injected structure is recovered robustly  
  (e.g. mobility, ecology, protein, engineered systems)

and

- **low-injectability domains**, where injected structure is recovered only weakly or at the threshold of tolerance  
  (e.g. fMRI, EEG, RNG, and the joint EEG+RNG domain)

This suggests that F1 may measure not only estimator validity, but also a domain-sensitive property related to how easily imposed artificial structure can be absorbed and recovered by the system representation.

---

### Additional methodological findings from Phase III.1

Phase III.1 added several methodological constraints and insights to the CDR framework.

#### 1. Clean null results are scientifically meaningful

The advanced joint model did not pass the stronger Phase III.1 gates, but this outcome is methodologically important.

The pipeline showed:

- successful injection recovery
- clean control collapse
- stable holdout behavior
- stable sensitivity behavior
- rejection of the advanced joint model under complexity and lift criteria

This demonstrates that CDR can produce **clean null_results** without inflating false positives through model complexity, proxy layers, or post-hoc interpretation.

---

#### 2. Joint-state modeling can dilute individual structure

Phase III.1 showed that individual-domain residual structure may not survive joint-state construction.

In the 10-subject diagnostic run:

- EEG-only reached `ε_test ≈ 0.10`
- RNG-only remained near `ε_test ≈ 0.00` globally
- the final joint model remained at `ε_joint_test = 0.00`

This suggests that combining domains into a single joint state can dilute or suppress signals that are visible in single-domain baselines.

This does **not** imply that the individual signals are false; rather, it indicates that joint-state representation may be insufficient when two domains contain structures that are orthogonal, weakly coupled, or conditionally related rather than directly coupled.

---

#### 3. BIC / complexity penalty is essential in CDR

The advanced Phase III.1 model was theoretically richer but statistically unjustified.

Representative comparison:

| Model | State space | Parameters | Result |
|------|-------------|------------|--------|
| `M0_observed_compact` | 8 states | 57 parameters | lower complexity |
| `M5_augmented_final` | 27 states | 703 parameters | strongly penalized by BIC |

The final model `M5_augmented_final` did not improve test likelihood or joint epsilon enough to justify its additional complexity.

This supports the use of BIC / MDL-style penalties as a necessary protection against over-flexible CDR models.

---

#### 4. Subject-shift sensitivity requires internal validation

Phase III.1 diagnostics showed that some subject-level LOSO signals appeared initially but disappeared under within-subject chronological diagnostics.

Examples:

- `SC4002 / M3_informational_joint`
- `SC4032 / M2_rng_quantum_proxy`
- `SC4022 / B1_rng_only`

These signals were therefore interpreted as likely subject-shift or split-dependent effects rather than robust EEG–RNG coupling.

This establishes a new methodological requirement:

```
LOSO leads must be checked against within-subject chronological diagnostics before being treated as candidate CDR effects.
```

---

#### 5. State-space density remains a critical design constraint

The Phase III experimental sequence confirmed that CDR is highly sensitive to the ratio between transition count and state-space size.

Empirically, sparse state spaces produced unstable behavior, while compact representations were more reproducible.

A practical rule emerging from the project is:

```
n_states should remain well below the effective number of observed transitions,
with transitions_per_state treated as a core stability diagnostic.
```

A provisional practical heuristic is:


```
n_states ≤ √n_transitions
``` 

but this should be treated only as a conservative design guideline, not as a universal mathematical law.

The final decision must still depend on:

- observed transition density
- control collapse
- injection recovery
- holdout behavior
- sensitivity stability
- BIC / complexity penalty

---

## Interpretation

Taken together, these results support the following view:

- **CDR does not act as a generic detector of randomness**
- **CDR behaves as a domain-sensitive detector of deviation from structural sufficiency**
- neural and joint experimental domains appear to require more careful state construction than macroscopic or strongly endogenous systems
- the present EEG + RNG implementation provides a stable falsifiable baseline, but does **not** yet supply strong evidence for robust cross-domain coupling

---

# Phase III — Experimental Validation (Completed)

🔗 **GitHub repository:**
https://github.com/ThiagoLuzpY/cdr-phase3.0-eeg-rng

Phase III extended the CDR framework from observational domains into **experimental neural and stochastic domains**.

Instead of testing a single joint system immediately, the experimental program was intentionally divided into **three internal stages**:

1. **Phase III.0.1 — EEG-only validation**
2. **Phase III.0.2 — RNG-only baseline validation**
3. **Phase III.0.3 — Joint EEG + RNG validation**

This modular strategy allowed the project to separate:

- neural residual structure
- stochastic baseline behavior
- and cross-domain coupling structure

before attempting stronger theoretical interpretations.

---

## Phase III.0.1 — EEG-Only Validation (Completed)

### Objective

Test whether direct neural activity measured from EEG exhibits low but detectable residual structure under the CDR framework.

This phase was designed as the experimental neural counterpart of the earlier fMRI validation, but with:

- higher temporal resolution
- direct electrophysiological measurement
- and explicit sleep-stage structure

---

### Dataset
```
Sleep-EDF Expanded
subject: SC4001
files:
SC4001E0-PSG.edf
SC4001EC-Hypnogram.edf
```
---

### Initial modeling strategy

The first EEG attempts used a richer state representation combining multiple spectral variables, including configurations such as:

```
(delta_power, theta_power, alpha_power, stage_code)
```

and later expanded neural feature sets.

These early runs showed that the EEG domain was highly sensitive to state-space construction. Some richer configurations produced unstable or weakly generalizable F1 behavior.

---

### Final state representation

The final validated EEG representation used a compact neural state:

```
state = (delta_power_bin, alpha_power_bin)
```

This reduced the state-space dimensionality while preserving physiologically meaningful contrast between lower-frequency and higher-frequency neural regimes.

---

### Final discretization

```
3 bins per variable
3² = 9 states
```

---

### EEG-only results


CDR Phase III.1 (EEG)
────────────────────────────────
```
F1_injection_recovery: PASS
eps_hat: 0.00
eps_true: 0.05
abs_err: 0.05

F2_controls_collapse: PASS
median_eps_controls: 0.00
fraction_below_tol: 1.0
max_eps_controls: 0.00

F3_holdout_generalization: PASS
eps_train: 0.04
eps_test: 0.00
abs_delta: 0.04

F5_sensitivity: PASS
eps_binsA: 0.04
eps_binsB: 0.04
abs_delta: 0.00

────────────────────────────────
FINAL: PASS
```

---

### EEG interpretation

The final EEG-only result indicates:

```
ε ≈ 0.04
```

Meaning:

- neural dynamics exhibit **low but non-zero residual structure**
- the result is stronger than a pure stochastic baseline
- but weaker than a strongly structured macroscopic domain

This is consistent with the interpretation that direct neural data may retain subtle residual organization not fully absorbed by the empirical kernel.

---

## Phase III.0.2 — RNG-Only Validation (Completed)

### Objective

Establish a strong experimental null baseline using a random source expected to approximate minimal structural organization.

This phase had two scientific roles:

- verify that CDR does not hallucinate structure in a stochastic domain
- provide a direct baseline against which joint EEG + RNG analysis could later be compared

---

### Dataset

```
ANU Quantum Random Number Generator
JSON sample file:
anu_sample.json
```

Because the free API limited the raw number count, the final validated representation used:

```
uint8 values converted into binary bits
```

This preserved falsifiability while increasing usable sequence density from a single legitimate sample.

---

### Initial modeling strategy

The first RNG attempts used the raw `uint8` values with larger state spaces, including configurations analogous to:

```
state = (x0_bin, x1_bin) with 3 bins
```

These early versions failed because the control system became unstable and created spurious structure under low-sample, high-sparsity settings.

---

### Final state representation

The final validated RNG representation used a binary-bit state:

```
state = (bit_t, bit_t+1)
```

---

### Final discretization

```
2 bins per variable
2² = 4 states
```

---

### RNG-only results


CDR Phase III.2 (RNG)
────────────────────────────────
```
F1_injection_recovery: PASS
eps_hat: 0.00
eps_true: 0.05
abs_err: 0.05

F2_controls_collapse: PASS
median_eps_controls: 0.00
fraction_below_tol: 1.0
max_eps_controls: 0.00

F3_holdout_generalization: PASS
eps_train: 0.00
eps_test: 0.00
abs_delta: 0.00

F5_sensitivity: PASS
eps_binsA: 0.00
eps_binsB: 0.00
abs_delta: 0.00

────────────────────────────────
FINAL: PASS
```

---

### RNG interpretation

The final RNG-only result indicates:

```
ε ≈ 0
```

Meaning:

- no residual structure was detected
- controls collapsed exactly as expected
- the stochastic baseline behaved as an experimentally clean null domain

This is one of the strongest confirmations in the project that CDR does **not** generate spurious structure under a domain intended to behave as near-pure randomness.

---

## Phase III.0.3 — Joint EEG + RNG Validation (Completed)

### Objective

Test whether the joint system formed by EEG and RNG exhibits cross-domain structure stronger than either component alone.

This was the most theoretically ambitious phase of the project, because it aimed to test whether neural activity and a stochastic baseline could form a jointly structured domain under the CDR framework.

---

### Joint strategy

The joint phase was deliberately exploratory and went through **multiple state-space redesigns**.

The original idea was to combine richer EEG and RNG components simultaneously, including joint representations of the form:

```
(delta_power, alpha_power, x0, x1)
```

However, these initial versions produced high sparsity and unstable injection recovery.

---

### Evolution of joint state-space tests

Three major joint configurations were explored:

#### 1. High-dimensional joint state
```
36 states
```

This representation was too sparse and produced unstable F1 behavior, including strong overshoot of injected epsilon.

#### 2. Compact joint state
```
6 states
state = (delta_power_bin, x0)
with 3 EEG bins × 2 RNG bins
```

This version stabilized the joint domain substantially and passed all gates, but only with very low injection values.

#### 3. Intermediate joint states
```
8 states
10 states
```

These were tested as robustness checks by increasing EEG granularity while keeping the same compact joint logic.

Both intermediate variants remained stable and showed results similar to the 6-state model:

- low joint epsilon
- clean controls
- stable holdout
- threshold-level F1 recovery

This demonstrated that the joint conclusion did not depend on a single minimal state-space choice.

---

### Final validated joint conclusion

Across the stabilized joint runs, the pattern remained:

```
eps_train ≈ 0.01–0.03
eps_test ≈ 0.00
```

and no robust high joint epsilon emerged.

---

### Representative joint result


CDR Phase III.3 (Joint EEG + RNG)
────────────────────────────────
```
F1_injection_recovery: PASS
eps_hat: 0.00
eps_true: 0.05
abs_err: 0.05

F2_controls_collapse: PASS
median_eps_controls: 0.00
fraction_below_tol: 1.0
max_eps_controls: low

F3_holdout_generalization: PASS
eps_train: 0.01–0.03
eps_test: 0.00
abs_delta: low

F5_sensitivity: PASS
eps_binsA: low
eps_binsB: low
abs_delta: low

────────────────────────────────
FINAL: PASS
```

---

### Joint interpretation

The final joint experimental result indicates:

```
ε_joint ≈ 0.00–0.03
```

Meaning:

- the joint EEG + RNG domain did **not** exhibit strong cross-domain structure
- the compact state-space redesign successfully stabilized the method
- but the observed joint epsilon remained weak

This is best interpreted as:

- a successful falsifiable implementation of the first EEG–RNG joint experiment
- but **not** as strong evidence of robust neural–stochastic coupling under the current observational setup


---

## Phase III.1 — Advanced Multi-Subject EEG + RNG Redesign (Completed)

🔗 **GitHub repository:**

https://github.com/ThiagoLuzpY/cdr-phase3.1-advanced-eeg-rng


---

### Objective

Phase III.1 was designed as a more advanced redesign of the previous joint EEG + RNG experiment.

The earlier Phase III.0.3 joint experiment showed that compact EEG + RNG state spaces could be stabilized, but the observed joint epsilon remained weak:


```
ε_joint ≈ 0.00–0.03
```

Phase III.1 therefore tested whether stronger joint structure could emerge after adding:

- multiple EEG subjects
- informational proxy variables
- latent inferred states
- quantum-aware proxy variables
- stricter model-comparison gates
- subject-level diagnostics
- corrected proxy ablations

The central question was:

> Can a deeper EEG–RNG joint structure be detected only after introducing informational, latent, or quantum-aware proxy layers rather than relying on raw EEG–RNG state pairing?

---

### Dataset

**EEG source:**

Sleep-EDF Expanded
PhysioNet
subset: sleep-cassette


Subjects used:


```
SC4001
SC4002
SC4011
SC4012
SC4021
SC4022
SC4031
SC4032
SC4041
SC4042
```

Recordings used:


```
SC4001E0
SC4002E0
SC4011E0
SC4012E0
SC4021E0
SC4022E0
SC4031E0
SC4032E0
SC4041E0
SC4042E0
```

EEG channel used:


```
EEG Fpz-Cz
```

Effective dataset size:


```
10 subjects
27,597 EEG epochs
30 seconds per epoch
```

---

### RNG source

**RNG source:**

ANU Quantum Random Number Generator


Raw RNG sample:


```
1024 uint8 values
```

Final RNG representation:


```
1024 uint8 × 8 bits = 8192 binary bits
```

RNG alignment mode:


```
resample_rng_metrics_to_eeg_epochs
```

RNG window size:


```
64 bits
```

---

### Advanced state construction

Phase III.1 expanded the previous joint representation from:


```
state_t = (EEG_t, RNG_t)
```

toward an augmented structure:


```
state_t = (EEG_t, RNG_t, I_t, Z_t, Q_t)
```

where:

- `I_t` = informational proxy layer
- `Z_t` = latent inferred state
- `Q_t` = quantum-aware proxy layer

The conceptual effective structural term was represented as:

```
Δχ* = Δχ_observed + λ₁ I_t + λ₂ Z_t + λ₃ Q_t
```

This preserved the falsifiable structure of CDR while allowing the model to test deeper informational and quantum-aware proxy hypotheses.

Model family

Phase III.1 evaluated the following models:

Model	Description
B0_eeg_only	EEG-only baseline
B1_rng_only	RNG-only baseline
M0_observed_compact	compact observed EEG + RNG model
M1_eeg_informational	EEG + informational proxy model
M2_rng_quantum_proxy	RNG + quantum-aware proxy model
M3_informational_joint	EEG informational proxy × RNG informational proxy
M4_latent_joint	latent-state joint model
M5_augmented_final	final augmented model with informational, latent, and quantum-aware proxy structure

The final registered advanced model was:
```
M5_augmented_final
Phase III.1 gates
```

Phase III.1 retained the original empirical gates:
```
Gate	Meaning
F1	Injection recovery
F2	Controls collapse
F3	Holdout generalization
F5	Sensitivity stability
```
and added advanced joint-model gates:

```
Gate	Meaning
F6	Incremental joint lift beyond EEG-only and RNG-only baselines
F7	Subject-level generalization
F8	Complexity penalty / BIC justification
F9	Proxy ablation stability
Main Phase III.1 results
```


CDR Phase III.1 (Advanced EEG + RNG)
────────────────────────────────

```
F1_injection_recovery: PASS
eps_hat: 0.03
eps_true: 0.05
abs_err: 0.02
tol_abs: 0.05

F2_controls_collapse: PASS
n_controls: 12
median_eps_controls: 0.00
mean_eps_controls: 0.00
max_eps_controls: 0.00
fraction_below_tol: 1.00

F3_holdout_generalization: PASS
eps_train: 0.00
eps_test: 0.00
abs_delta: 0.00
max_delta: 0.10

F5_sensitivity: PASS
eps_primary: 0.00
eps_sensitivity: 0.00
abs_delta: 0.00
max_delta: 0.12

F6_incremental_joint_lift: FAIL
eps_joint_test: 0.00
eps_eeg_test: 0.10
eps_rng_test: 0.00
baseline_max: 0.10
joint_lift: -0.10
joint_lift_min: 0.03
strong_joint_eps_min: 0.07

F7_subject_generalization: FAIL
n_subjects: 10
n_valid_subjects: 10
fraction_positive_lift: 0.00
median_joint_eps: 0.00
median_eeg_eps: 0.00
median_rng_eps: 0.00

F8_complexity_penalty: FAIL
baseline_model: M0_observed_compact
advanced_model: M5_augmented_final
baseline_BIC: 9178.45
advanced_BIC: 16540.56
baseline_n_states: 8
advanced_n_states: 27
baseline_n_params: 57
advanced_n_params: 703

F9_proxy_ablation_stability: FAIL
full_eps_joint: 0.00
remove_I_t: 0.00
remove_Z_t: 0.00
remove_Q_t: 0.00

────────────────────────────────
FINAL: NULL_RESULT
```


Diagnostic model ranking

The diagnostic ranking showed that the best global model was not the advanced joint model.

Best diagnostic model:

B0_eeg_only

Diagnostic values:
```
eps_test: 0.10
BIC_test: 6987.06
LL_test: -3203.99
n_states: 9
```
This means that the EEG-only model retained detectable residual structure, while the advanced EEG + RNG joint model did not improve beyond the single-domain baseline.

Subject-level diagnostics

The 10-subject LOSO diagnostic initially produced isolated leads:
```
SC4002 / M3_informational_joint:
eps_test: 0.08
lift_over_baseline: 0.08

SC4032 / M2_rng_quantum_proxy:
eps_test: 0.08
lift_over_baseline: 0.08

SC4022 / B1_rng_only:
eps_test: 0.80
```

However, these leads did not survive internal chronological diagnostics.

Internal all-subject diagnostics showed:

```
internal_positive_lift_rows: none
```

The only internal positive epsilon row was:


SC4002 / B1_rng_only:

```
eps_test: 0.15
lift_over_baseline: 0.00
```

This indicates that the apparent LOSO leads were more consistent with:

```
subject-shift effects
split-dependent structure
baseline-specific variation
```

rather than robust EEG–RNG coupling.

RNG alignment audit

The RNG alignment audit did not reveal a gross alignment anomaly.

Summary:
```
n_subjects: 10

rng_bit_mean:
min: 0.4824
median: 0.5005
max: 0.5110

rng_bit_transition_rate_subject:
min: 0.4841
median: 0.4922
max: 0.5055
```

rng_window_reuse_ratio:

```
1.00 across subjects
```

This supports the interpretation that the Phase III.1 null result was not caused by an obvious RNG bit-balance or window-reuse artifact.

Sleep-stage distribution

The Sleep-EDF subjects were strongly dominated by Wake epochs.

Examples:

Subject	W epochs	N2 epochs	N3 epochs	REM epochs
SC4001	1997	250	220	125
SC4002	1885	373	297	215
SC4022	1871	402	119	179
SC4032	1957	400	131	199
SC4042	1773	514	94	270

This observation motivates the next stage of analysis:

### Phase III.2 — regime-aware and sleep-stage-filtered EEG–RNG validation

because a global model mixing Wake, N1, N2, N3 and REM may dilute regime-specific structure.

### Phase III.1 interpretation

Phase III.1 produced a clean null result.

The advanced model succeeded technically in the sense that:
```
injection recovery passed
controls collapsed
holdout generalization passed
sensitivity stability passed
ablation artifacts were corrected
subject-level diagnostics were performed
RNG alignment was audited
```

However, the advanced joint model failed to demonstrate:
```
positive joint lift
subject-level generalization
complexity justification
proxy-dependent signal reduction
```

Therefore, the final conclusion is:

Phase III.1 did not reveal robust EEG–RNG joint structure beyond the single-domain baselines under the current design.

This does not invalidate the CDR framework.

Instead, it strengthens the falsifiability of the framework by showing that CDR can reject an advanced theoretical model when the evidence does not survive the required gates.

Scientific meaning of the null result

The Phase III.1 null result should be interpreted as a boundary condition:

```
EEG-only structure remains detectable
RNG-only behaves as a clean stochastic baseline globally
the advanced joint EEG + RNG model does not show robust coupling
LOSO leads require internal validation before interpretation
model complexity must be justified by BIC / likelihood improvement
global joint-state modeling may be too coarse for subtle EEG–RNG hypotheses
```

The next step is therefore not to repeat the same joint model.

The next step is to test whether any possible effect is:

```
regime-dependent
sleep-stage-dependent
lag-dependent
conditional rather than directly joint
better captured through enriched multi-channel EEG features
```

---


## Phase III Scientific Interpretation

Phase III achieved three important results:

1. **EEG-only** validation confirmed low but non-zero residual neural structure  
2. **RNG-only** validation established a strong null stochastic baseline  
3. **Joint EEG + RNG** validation showed that, in the present design, the cross-domain joint epsilon remains weak
4. **Advanced multi-subject EEG + RNG redesign** produced a clean null result under stricter latent, informational, quantum-aware and subject-level validation gates


The main conclusion of Phase III is therefore:

- direct neural domains may exhibit low residual structure under CDR
- stochastic RNG baselines remain structurally sufficient at the tested resolution
- simple joint EEG + RNG state construction does not show robust coupling
- advanced joint modeling with `I_t`, `Z_t` and `Q_t` also does not reveal robust coupling under the current public observational setup
- subject-level LOSO leads must be validated internally before interpretation
- CDR can produce clean null results without inflating false positives


The main conclusion of Phase III is: 

- the EEG-only domain remains the strongest neural signal found so far 
- the RNG-only domain remains a clean stochastic null 
- the EEG + RNG joint domain remains unresolved 
- a deeper test requires regime-aware, lagged and conditional modeling rather than a single global joint-state model

## Future Validation Domains

### Phase III.2 — Regime-Aware, Lagged and Conditional EEG–RNG Validation

The next stage of the project is **not** to repeat the Phase III.1 global joint-state model.

Phase III.1 showed that:

- EEG-only residual structure remains detectable
- RNG-only remains globally consistent with a stochastic baseline
- the advanced multi-subject joint model did not produce robust joint lift
- isolated LOSO leads disappeared under internal subject diagnostics
- Wake epochs dominate the Sleep-EDF records
- global EEG + RNG state construction may be too coarse for subtle coupling hypotheses

This motivates a new stage:



Phase III.2 — Regime-Aware, Lagged and Conditional EEG–RNG Validation


---

### Objective

Test whether the absence of robust EEG–RNG joint structure in Phase III.1 is due to limitations of global state construction.

Phase III.2 will investigate whether any possible EEG–RNG effect is:

- regime-dependent
- sleep-stage-dependent
- lag-dependent
- conditional rather than directly joint
- better captured through enriched EEG features

---

### Planned submodules

#### III.2A — Regime-aware / sleep-stage analysis

Phase III.2A will test whether the EEG–RNG relationship changes when the data are separated by sleep regime.

Planned regimes:


```
full
no_wake
stable_sleep = N2 + N3
deep_sleep = N3
REM_only
transition_epochs
```

Purpose:

- test whether Wake dominates or dilutes the joint signal
- evaluate whether sleep-only states show stronger residual structure
- separate global null results from regime-specific behavior

---

#### III.2B — Lagged EEG–RNG alignment

Phase III.2B will test whether the possible relationship depends on temporal offset.

Pre-registered lags:


```
-5, -3, -1, 0, +1, +3, +5 epochs
```

With 30-second EEG epochs, these correspond to:


```
-150 s, -90 s, -30 s, 0 s, +30 s, +90 s, +150 s
```

Purpose:

- test temporal alignment sensitivity
- avoid assuming that any possible relationship must occur at lag 0
- preserve falsifiability by limiting lags to a fixed pre-defined set

---

#### III.2C — Conditional CDR

Phase III.2C will test conditional relationships rather than direct joint-state coupling.

Main comparisons:


```
C0: P(EEG_{t+1} | EEG_t)
C1: P(EEG_{t+1} | EEG_t, RNG_t)

C2: P(RNG_{t+1} | RNG_t)
C3: P(RNG_{t+1} | RNG_t, EEG_t)
```

Purpose:

- test whether RNG_t adds information about EEG_{t+1}
- test whether EEG_t adds information about RNG_{t+1}
- avoid diluting signal in a large global joint state
- distinguish conditional influence from simple joint-state structure

This is expected to be the most important Phase III.2 module.

---

#### III.2D — Multi-channel EEG enrichment

Phase III.2D will evaluate whether a richer EEG representation improves detection.

The current validated EEG channel is:


```
EEG Fpz-Cz
```

Future enrichment may include:


```
EEG Pz-Oz
```

and derived cross-channel features such as:

- power differences
- spectral balance differences
- simple epoch-level channel correlation
- cross-channel spectral ratios

Purpose:

- test whether a single EEG channel is too limited
- evaluate whether spatial EEG structure improves CDR sensitivity
- prepare for future synchronized EEG + QRNG experiments

---

#### III.2E — Future synchronized EEG + QRNG protocol

The current EEG and RNG data were not collected simultaneously.

A stronger future experiment would require:


```
real-time EEG
+
real-time QRNG
+
shared timestamps
+
same subject
+
same session
+
controlled sleep, rest, meditation or cognitive conditions
```

This future design would test a much stronger version of the EEG–QRNG hypothesis than the current public-data observational setup.

---

### Scientific meaning of Phase III.2

Phase III.2 will answer a more refined question than Phase III.1:

> If a deeper EEG–RNG relationship exists, does it appear only under specific neural regimes, temporal lags, or conditional transition structures rather than in a global joint-state model?

The goal is not to force a positive result.

The goal is to test whether the Phase III.1 null result is:

- a true null under public EEG + RNG data
- a limitation of global joint-state representation
- or a sign that more precise regime-aware and conditional modeling is required

---

## Project Structure
```
cdr-phase1-validation/
│
├── config/
│   ├── __init__.py
│   ├── phase1_config.py
│   ├── phase2_config.py
│   ├── phase2_config_ecology.py
│   ├── phase2_config_fmri.py
│   ├── phase2_config_mobility.py
│   └── phase2_config_protein.py
│   └── phase3_config_eeg.py
│   └── phase3_config_joint.py
│   └── phase3_config_rng.py
│   └── phase3_1_config.py
│
├── data/
│   ├── interim/
│       ├── interion/
│       │   ├── phase3_1/
│   ├── processed/
│       ├── phase3_1/
│   └── raw/
│       ├── ecology/
│       ├── fmri/
│       ├── geolife/
│       ├── eeg/
│       ├── rng/
│       ├── opsp/
│       └── protein/
│
├── results/
│   ├── golden_run_phase1_plus_v1/
│   ├── phase2_opsp/
│   ├── phase2_ecology/
│   ├── phase2_fmri/
│   ├── phase2_mobility/
│   ├── phase2_protein/
│   ├── phase3_eeg/
│   ├── phase3_joint/
│   ├── phase3_rng/
│   ├── phase3_1/
│   └── .gitkeep
│
├── scripts/
│   ├── __init__.py
│   ├── make_audit_bundle.py
│   ├── download_qrng.py
│   └── run_phase1_plus_full.py
│
├── src/
│   ├── kernels/
│   │   ├── empirical_kernel.py
│   │   └── reweighted_kernel.py
│   ├── __init__.py
│   ├── adversarial_kernel.py
│   ├── artifacts.py
│   ├── build_states.py
│   ├── controls.py
│   ├── controls_phase2.py
│   ├── controls_phase2_ecology.py
│   ├── controls_phase2_fmri.py
│   ├── controls_phase2_mobility.py
│   ├── controls_phase2_protein.py
│   ├── controls_phase3_eeg.py
│   ├── controls_phase3_joint.py
│   ├── controls_phase3_rng.py
│   ├── controls_phase3_1.py
│   ├── discretize.py
│   ├── ecology_loader.py
│   ├── eeg_loader.py
│   ├── estimators.py
│   ├── fmri_loader.py
│   ├── geolife_loader.py
│   ├── ising_kernel.py
│   ├── joint_loader.py
│   ├── model_selection.py
│   ├── opsp_loader.py
│   ├── phase1_plus_runner.py
│   ├── phase1_runner.py
│   ├── phase2_runner.py
│   ├── phase2_runner_ecology.py
│   ├── phase2_runner_fmri.py
│   ├── phase2_runner_mobility.py
│   ├── phase2_runner_protein.py
│   ├── phase3_runner_eeg.py
│   ├── phase3_runner_joint.py
│   ├── phase3_runner_rng.py
│   ├── phase3_1_diagnostics.py
│   ├── phase3_1_features.py
│   ├── phase3_1_latent.py
│   ├── phase3_1_loader.py
│   ├── phase3_1_metrics.py
│   ├── phase3_1_runner.py
│   ├── protein_loader.py
│   ├── rng_loader.py
│   ├── statistics.py
│   ├── validators.py
│   └── validators_phase2.py
│
├── tests/
│   ├── __init__.py
│   ├── test_controls.py
│   ├── test_estimators.py
│   ├── test_ising.py
│   ├── test_phase1_plus_runner.py
│   ├── test_statistics.py
│   └── test_validators.py
│
├── venv/
│
├── .gitignore
├── main.py
├── README.md
└── requirements.txt
```

---

## Running Phase II

**Energy system validation:**
```bash
python -m src.phase2_runner
```

**fMRI validation:**
```bash
python -m src.phase2_runner_fmri
```

**Mobility validation:**
```bash
python -m src.phase2_runner_mobility
```

**Ecology validation:**
```bash
python -m src.phase2_runner_ecology
```

**Protein validation:**
```bash
python -m src.phase2_runner_protein
```

**EEG-only validation:**
```bash
python -m src.phase3_runner_eeg
```

**RNG-only validation:**
```bash
python -m src.phase3_runner_rng
```

**Joint EEG + RNG validation:**
```bash
python -m src.phase3_runner_joint
```

**Advanced Phase III.1 EEG + RNG redesign:**
```bash
 python -m src.phase3_1_runner 
 python -m src.phase3_1_diagnostics
 ```

**Results saved in:**
```
results/phase2_opsp/
results/phase2_fmri/
results/phase2_mobility/
results/phase2_ecology/
results/phase2_protein/
results/phase3_eeg/
results/phase3_rng/
results/phase3_joint/
results/phase3_1/
results/phase3_1/diagnostics/
```

---

## Reproducibility

The pipeline ensures reproducibility via:

- ✅ Fixed random seeds
- ✅ Serialized configuration files
- ✅ Saved discretization bins
- ✅ Stored likelihood curves
- ✅ Checkpoint files
- ✅ Domain-specific holdout protocols when required by data structure
- ✅ Explicit separation between observational domains, neural domains, stochastic baselines, and joint domains
- ✅ Stable compact state-space redesign when sparsity becomes dominant

Additional reproducibility safeguards introduced in Phase III.1: 

- ✅ Multi-subject EEG loading with explicit subject identifiers 
- ✅ Prevention of artificial transitions across subject / recording boundaries 
- ✅ RNG alignment audit 
- ✅ Corrected proxy ablations using within-subject shuffle rather than constant replacement 
- ✅ Internal chronological diagnostics for all subjects 
- ✅ LOSO leads separated from within-subject persistent effects 
- ✅ Explicit BIC / complexity-penalty evaluation 
- ✅ Separation between registered model results and post-run diagnostics

All experiments are deterministic under identical configurations.

---
### Experimental data sources used in Phase III

**EEG source:**

```
Sleep-EDF Expanded
PhysioNet
subset: sleep-cassette
```

Files used in the first validated Phase III.0.1 EEG-only run:

```
SC4001E0-PSG.edf
SC4001EC-Hypnogram.edf
```

Files used in Phase III.1 advanced multi-subject redesign:


```
SC4001E0-PSG.edf
SC4001EC-Hypnogram.edf

SC4002E0-PSG.edf
SC4002EC-Hypnogram.edf

SC4011E0-PSG.edf
SC4011EH-Hypnogram.edf

SC4012E0-PSG.edf
SC4012EC-Hypnogram.edf

SC4021E0-PSG.edf
SC4021EH-Hypnogram.edf

SC4022E0-PSG.edf
SC4022??-Hypnogram.edf

SC4031E0-PSG.edf
SC4031??-Hypnogram.edf

SC4032E0-PSG.edf
SC4032??-Hypnogram.edf

SC4041E0-PSG.edf
SC4041??-Hypnogram.edf

SC4042E0-PSG.edf
SC4042??-Hypnogram.edf
```


**RNG source:**

```
ANU Quantum Random Number Generator
```

Sample file used:
```
anu_sample.json
```

Raw RNG sample:
```
1024 uint8 values
```

Final RNG representation:
```
1024 uint8 × 8 bits = 8192 binary bits
```

Final validated RNG alignment mode:
```
resample_rng_metrics_to_eeg_epochs
```

This allowed the stochastic baseline to remain falsifiable while enabling EEG-aligned RNG proxy construction for Phase III.1.


---
## References

- Popper, K.R. (1959). *The Logic of Scientific Discovery.*
- Lakatos, I. (1978). *The Methodology of Scientific Research Programmes.*
- Rosen, R. (1991). *Life Itself.*
- Open Power System Data (2020) https://open-power-system-data.org/
- GeoLife GPS Trajectory Dataset (Microsoft Research)
- OpenNeuro dataset ds002938
- Alanine dipeptide molecular dynamics trajectories
- Sleep-EDF Expanded dataset (PhysioNet)
- ANU Quantum Random Number Generator
- MNE-Python 
- scikit-learn

---

## Citation
```bibtex
@software{luz2026cdr,
  title={Core Distinguishability Relativity: Empirical Validation Framework},
  author={Luz, Thiago},
  year={2026},
  url={https://github.com/ThiagoLuzpY/cdr-phase1-validation}
}
```

---

## License

**CC0 1.0 Universal (Public Domain)**

---

## Author

**Thiago Luz**  
Independent Researcher  
Rio de Janeiro, Brazil

- **GitHub:** https://github.com/ThiagoLuzpY/
- **ORCID:** https://orcid.org/0009-0008-9732-324X

---

## Acknowledgments

Thanks to the open scientific ecosystem:

- NumPy
- SciPy
- pandas
- nilearn
- mdtraj
- OpenNeuro
- Open Power System Data
- Microsoft GeoLife Dataset
- Sleep-EDF Expanded dataset (PhysioNet)
- ANU Quantum Random Number Generator
- MNE-Python 
- scikit-learn

---

**Last updated:** June 2026  
**Status:** Phase I complete ✅ | Phase II.1A complete ✅ | Phase II.1B complete ✅ | Phase II.2 complete ✅ | Phase II.3 complete ✅ | Phase II.4 complete ✅ | Phase III.0.1 complete ✅ | Phase III.0.2 complete ✅ | Phase III.0.3 complete ✅ | Phase III.1 complete ✅ | Phase III.2 planned 📋
