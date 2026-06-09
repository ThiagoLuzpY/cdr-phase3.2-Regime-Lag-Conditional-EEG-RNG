# Phase III.2E — Synchronized EEG–QRNG Protocol, Surrogate Validation, and Prospective F8 Revision

## 1. Document Control

**Project:** CDR — Core Distinguishability Relativity  
**Phase:** III.2E  
**Title:** Synchronized EEG–QRNG Protocol and Gate-Sensitivity Validation  
**Protocol version documented here:** III.2E v2, with prospective III.2E v3 amendments  
**Current operational status:** `pipeline_completed`  
**Current scientific classification:** `surrogate_primary_localized_predictive_lead_bic_limited`  
**Current empirical claim status:** `not permitted from surrogate data`  
**Next planned execution:** III.2E v3 with real synchronized EEG–QRNG acquisition  

This document replaces the original protocol-only description of Phase III.2E.

Phase III.2E has now completed a full synchronized surrogate replication, including:

- schema validation;
- synchronization validation;
- leakage-safe loading;
- multi-resolution window construction;
- feature generation;
- registered lag alignment;
- conditional CDR estimation;
- synchronized negative controls;
- official metrics and gates;
- F7/F8/F12 gate-sensitivity analysis;
- final diagnostics;
- safe-resume runner validation;
- audit-bundle generation.

The current result is not an empirical EEG–QRNG claim because the synchronized input is surrogate. It is a completed end-to-end validation of the Phase III.2E analytical architecture.

---

## 2. Scientific Position of Phase III.2E

Phase III.2E is the synchronized continuation of Phase III.2A/B/C/D.

The earlier subphases used real EEG and QRNG sources that were not co-acquired under a common physical clock. Those analyses were useful for testing regime-aware, lagged, conditional, and multichannel CDR machinery, but they could not establish a synchronized cross-domain relation.

Phase III.2E changes the experimental question from post hoc alignment to synchronized acquisition:

> Does one domain provide predictive information about the future state of the other domain when EEG and QRNG are recorded under a shared or traceable temporal reference?

The Phase III.2E v2 surrogate execution was designed to stress-test the complete synchronized pipeline before real acquisition. The surrogate is not treated as biological or quantum evidence. It is treated as a protocol-replication and computational-validation instrument.

---

## 3. Current v2 Surrogate Result

The completed surrogate execution produced the following consolidated classification:

```text
operational_status:
pipeline_completed

scientific_status:
primary_lead_bic_limited

scientific_classification:
surrogate_primary_localized_predictive_lead_bic_limited

candidate_detected:
True

surrogate:
True

can_claim_empirical_cdr:
False
```

### 3.1 Consolidated data volumes

```text
loaded rows:                    27,597
window rows:                    50,590
feature rows:                   50,590
alignment inventory rows:       28
conditional model rows:         147
conditional pair rows:          84
primary positive rows:          1
control runs:                    2,268
control pair rows:              9,072
metrics table rows:             11
F7 sensitivity rows:            10
F8 sensitivity rows:            9
F12 sensitivity rows:           70
```

### 3.2 Selected primary lead

The selected primary lead was found in the registered primary region:

```text
window_seconds:                 30
lag_windows:                    +5
lag_seconds:                    150
baseline_model:                 E2_QRNG_next_given_QRNG
augmented_model:                E6_QRNG_next_given_QRNG_MCEEG
primary:                        True
multichannel:                   True
conditional_lift:               1.0
baseline_eps_test:              0.0
augmented_eps_test:             1.0
held-out LL improvement:        +6.068517...
AIC improvement:                -19.862965...
BIC improvement:                -132.193071...
positive subjects:              1/10
fraction positive subjects:     0.10
```

### 3.3 Gate outcome in the v2 implementation

```text
PASS:
F2, F3, F6, F7, F9, F10, F11, F12

FAIL in the current v2 code:
F8

NOT EVALUATED:
F1, F5
```

All required synchronized negative controls passed after the corrected control-specific classification.

The result therefore supports the existence of a registered localized predictive lead in the surrogate execution. It does not establish empirical EEG–QRNG coupling.

---

## 4. Central Research Questions

### 4.1 Forward direction

Does QRNGₜ add predictive information about EEGₜ₊τ beyond the autoregressive structure of EEG?

```text
P(EEGₜ₊τ | EEGₜ, QRNGₜ)
versus
P(EEGₜ₊τ | EEGₜ)
```

### 4.2 Reverse direction

Does EEGₜ add predictive information about QRNGₜ₊τ beyond the autoregressive structure of QRNG?

```text
P(QRNGₜ₊τ | QRNGₜ, EEGₜ)
versus
P(QRNGₜ₊τ | QRNGₜ)
```

### 4.3 Multichannel direction

Does multichannel EEG state information improve prediction of future QRNG state, or does QRNG state improve prediction of future multichannel EEG state?

The reverse and multichannel directions are required to examine:

- directional asymmetry;
- bidirectional artifacts;
- temporal leakage;
- model instability;
- state-space sparsity;
- synchronization dependence;
- subject-specific or subgroup-specific effects.

---

## 5. Hypotheses

### 5.1 Null hypothesis — H0

In synchronized EEG–QRNG data, neither domain adds reproducible held-out predictive information about the future state of the other domain beyond its own autoregressive transition structure.

```text
P(EEGₜ₊τ | EEGₜ, QRNGₜ)
≈
P(EEGₜ₊τ | EEGₜ)

P(QRNGₜ₊τ | QRNGₜ, EEGₜ)
≈
P(QRNGₜ₊τ | QRNGₜ)
```

### 5.2 Localized predictive alternative — H1-L

At least one preregistered subject, window, lag, direction, and model pair exhibits:

- positive conditional lift;
- positive held-out likelihood improvement;
- collapse under required negative controls;
- valid synchronization;
- no critical epsilon saturation;
- protection against multiple comparisons.

H1-L permits a heterogeneous effect. It does not require the effect to occur in most subjects.

### 5.3 Global parsimonious alternative — H1-G

The augmented model is not only predictively better in held-out data but is also favored after a defensible complexity correction appropriate to dependent, repeated-measures, sparse-state time-series data.

H1-G is stronger than H1-L.

A result may support H1-L without supporting H1-G.

---

## 6. Required Data Contract

The minimum accepted analytical unit is a synchronized subject/session/window row.

### 6.1 Required identity columns

```text
subject_id
session_id
window_id
```

### 6.2 Required timing and synchronization columns

```text
window_start_utc
window_end_utc
eeg_timestamp_utc
qrng_timestamp_utc
sync_quality
clock_drift_ms
```

### 6.3 Required EEG columns

```text
eeg_channel
eeg_delta_power
eeg_theta_power
eeg_alpha_power
eeg_beta_power
eeg_entropy
```

Optional but recommended:

```text
eeg_gamma_power
eeg_hjorth_mobility
eeg_hjorth_complexity
eeg_line_length
eeg_signal_std
eeg_artifact_score
eeg_quality_score
```

### 6.4 Required multichannel EEG columns

```text
eeg_primary_channel
eeg_secondary_channel
eeg_primary_delta_power
eeg_primary_alpha_power
eeg_secondary_delta_power
eeg_secondary_alpha_power
interchannel_delta_ratio
interchannel_alpha_ratio
fronto_parietal_delta_shift
fronto_parietal_alpha_shift
cross_channel_corr
```

### 6.5 Required QRNG columns

```text
qrng_state
qrng_bit_balance
qrng_transition_rate
qrng_entropy
```

Optional but recommended:

```text
qrng_run_length_mean
qrng_run_length_std
qrng_run_length_max
qrng_compressibility_proxy
qrng_surprise_index
qrng_transition_asymmetry
qrng_quality_score
```

---

## 7. Synchronization Requirements

### 7.1 Timestamp requirements

The following fields must be present and parseable:

```text
window_start_utc
window_end_utc
eeg_timestamp_utc
qrng_timestamp_utc
```

Within each subject/session:

- timestamps must be monotonic;
- no timestamp may cross subject or session boundaries;
- synchronization metadata must be preserved;
- no alignment may be selected by maximizing the observed signal.

### 7.2 Default synchronization thresholds

```text
sync_quality >= 0.95
abs(clock_drift_ms) <= 50
```

Rows failing either threshold are invalid for confirmatory analysis.

### 7.3 Accepted synchronization modes

```text
hardware_shared_clock
software_timestamp_aligned
posthoc_clock_drift_corrected
external_sync_marker
```

### 7.4 Unaccepted synchronization modes

```text
unknown_clock_relation
manual_alignment_without_log
random_pairing
posthoc_alignment_to_maximize_signal
```

### 7.5 Synchronization failure status

```text
status: sync_invalid
```

A sync-invalid dataset cannot proceed to an empirical cross-domain interpretation.

---

## 8. Registered Window Strategy

The revised III.2E v2 registered grid is:

```text
all registered windows:
5s, 30s, 60s, 90s

primary windows:
30s, 60s

exploratory windows:
5s, 90s
```

### 8.1 Rationale

The original 5–10 second confirmatory range was considered excessively restrictive for a subtle, heterogeneous structure.

The revised primary range of 30–60 seconds allows:

- more stable EEG band-power estimates;
- longer cross-domain transition accumulation;
- reduced sensitivity to single-window noise;
- a more plausible observation interval for subject-specific state selection.

The 5-second and 90-second windows remain available for exploratory sensitivity analysis.

### 8.2 Window validity

A window is valid only if:

- EEG features are available;
- QRNG features are available;
- synchronization thresholds pass;
- no critical timestamp is missing;
- subject/session boundaries are preserved;
- aggregation follows the registered non-overlapping or native derivation rule.

---

## 9. Registered Lag Strategy

The revised registered lag grid is:

```text
-1, 0, +1, +2, +3, +4, +5 windows
```

These lags are evaluated across the registered windows.

Examples:

```text
window = 30s, lag = +5
target displacement = +150s

window = 60s, lag = -1
target displacement = -60s
```

Interpretation:

- negative lag: target precedes the predictor window;
- zero lag: concurrent relation;
- positive lag: predictor precedes the target;
- positive lag is central to directional predictive interpretation.

The entire registered lag range must remain fixed before real-data analysis.

---

## 10. Conditional CDR Models

### 10.1 Single-channel EEG models

```text
E0:
P(EEGₜ₊τ | EEGₜ)

E1:
P(EEGₜ₊τ | EEGₜ, QRNGₜ)
```

```text
lift_EEG = ε(E1) - ε(E0)
```

### 10.2 QRNG models with single-channel EEG

```text
E2:
P(QRNGₜ₊τ | QRNGₜ)

E3:
P(QRNGₜ₊τ | QRNGₜ, EEGₜ)
```

```text
lift_QRNG = ε(E3) - ε(E2)
```

### 10.3 Multichannel EEG models

```text
E4:
P(MC_EEGₜ₊τ | MC_EEGₜ)

E5:
P(MC_EEGₜ₊τ | MC_EEGₜ, QRNGₜ)

E6:
P(QRNGₜ₊τ | QRNGₜ, MC_EEGₜ)
```

```text
lift_MC_EEG = ε(E5) - ε(E4)

lift_QRNG_MC = ε(E6) - ε(E2)
```

### 10.4 Registered primary model pairs

The current primary test family includes:

```text
E0 versus E1
E2 versus E3
E2 versus E6
```

The E4 versus E5 multichannel-EEG target direction remains available as an active exploratory pair unless prospectively promoted in the v3 configuration.

---

## 11. Leakage-Safe Estimator

The estimator must preserve the split:

```text
reference_train
calibration
test_holdout
```

Rules:

1. P₀ is fitted only on `reference_train`.
2. Δχ is estimated only on `calibration`.
3. ε is selected only on `calibration`.
4. Confirmation is performed only on `test_holdout`.
5. No confirmatory threshold may be selected using the test set.
6. If the selected ε does not improve held-out likelihood relative to ε=0, the confirmed test epsilon is set to 0.0.
7. Subject, session, window, and temporal boundaries must prevent leakage.

---

## 12. Negative Controls

### 12.1 Registered control families

The III.2E control layer contains:

```text
within_subject_qrng_shuffle
within_subject_eeg_shuffle
circular_qrng_shift
subject_mismatch
timestamp_jitter_control
phase_randomized_eeg_control
window_permutation_control
conditional_target_shuffle
sync_break_control
```

### 12.2 Required controls for gate F2

The current required control set is:

```text
within_subject_qrng_shuffle
within_subject_eeg_shuffle
circular_qrng_shift
subject_mismatch
conditional_target_shuffle
sync_break_control
```

The remaining controls are retained as diagnostic or supplementary controls.

### 12.3 Replication design

```text
control replicates per registered combination: 12
completed control runs: 2,268
completed control pair rows: 9,072
```

### 12.4 Control-specific subject threshold

The control-collapse threshold is distinct from the discovery threshold.

```text
control subject-fraction threshold: 0.60
discovery subject-effect fraction: 0.10
```

A subtle effect may be accepted as a discovery lead in one of ten subjects, while a negative control must collapse broadly enough to demonstrate that the observed structure is not a generic artifact.

### 12.5 Sync-break control

The sync-break control is mandatory.

It breaks the temporal relation between EEG and QRNG while preserving marginal structure.

Expected behavior:

- a synchronization-dependent effect must collapse;
- survival under sync break blocks synchronized interpretation.

---

## 13. Official Gates in the III.2E v2 Implementation

### 13.1 F2 — Negative controls

Pass condition:

All required negative-control families satisfy the registered control-specific collapse criterion.

Current v2 surrogate result:

```text
PASS
```

### 13.2 F3 — Held-out generalization

Pass condition:

The selected augmented configuration improves held-out test likelihood relative to its baseline.

Current v2 surrogate result:

```text
PASS
```

### 13.3 F6 — Conditional lift

Pass condition:

The augmented model produces positive conditional lift above the registered minimum.

Current v2 surrogate result:

```text
PASS
```

### 13.4 F7 — Adaptive subject consistency

The revised discovery rule accepts heterogeneous and localized effects.

Current registered rule:

```text
subject_effect_fraction >= 0.10
minimum positive subjects >= 1
```

For a cohort of ten subjects:

```text
required positive subjects = 1
```

Current v2 surrogate result:

```text
1/10 subjects
PASS
```

For larger real-data cohorts, the required proportion may be prospectively reduced to approximately 1–3%, provided the absolute minimum subject count and replication requirements are frozen before analysis.

### 13.5 F8 — v2 global BIC criterion

The v2 code implemented:

```text
BIC improvement >= 0
```

with:

```text
BIC improvement =
BIC_baseline - BIC_augmented
```

Current v2 surrogate result:

```text
BIC improvement: -132.193071...
v2 code status: FAIL
```

This stored status must remain unchanged in the v2 audit trail.

However, the scientific meaning is restricted:

> The conventional global BIC did not favor the augmented model under the nominal complexity penalty.

It does not mean:

- no predictive structure was found;
- the registered lead is invalid;
- held-out predictive improvement was absent;
- the entire F8 concept has been disproven.

The current result is therefore classified as:

```text
primary_lead_bic_limited
```

### 13.6 F9 — Registered localization or ablation support

Pass condition:

The lead is located inside the registered window, lag, and model-pair region and is not dependent on an unregistered post hoc selection.

Current v2 surrogate result:

```text
PASS
```

### 13.7 F10 — Registered window support and state-density validity

Pass condition:

The selected lead occurs in a registered primary window and retains sufficient estimable transition structure.

Current v2 surrogate result:

```text
PASS
```

### 13.8 F11 — Multiple-comparison guard

Pass condition:

Primary and exploratory tests remain separated and the selected result survives the registered guard.

Current v2 surrogate result:

```text
PASS
```

### 13.9 F12 — Epsilon saturation diagnostic

Current parameters:

```text
epsilon saturation value: 2.0
warning fraction: 0.95
```

Current v2 surrogate result:

```text
official saturation fraction: 0.0
PASS
```

F12 is diagnostic and must not erase an otherwise valid localized lead solely because a small number of rows approach the boundary.

---

## 14. Prospective F8 Revision for III.2E v3

### 14.1 Reason for revision

The conventional threshold:

```text
BIC improvement >= 0
```

is mathematically valid as a comparison between two conventional BIC values.

It was not, however, empirically calibrated or theoretically derived specifically for:

- repeated observations within subjects;
- temporally autocorrelated windows;
- sparse Markov transition tables;
- multichannel neural state spaces;
- localized effects occurring in a minority of subjects;
- effective rather than nominal parameter complexity.

Therefore, the III.2E v3 protocol will not allow the conventional BIC subtest alone to veto all evidence of a localized predictive structure.

This amendment is prospective. It must be implemented and frozen before analysis of real synchronized data.

### 14.2 F8 becomes a composite complexity-aware predictive gate

F8 will be decomposed into the following components.

#### F8a — Conventional global BIC

```text
PASS if:
BIC improvement >= 0
```

Purpose:

- report whether the nominal globally pooled augmented model is preferred under the conventional BIC penalty.

A negative F8a result is reported as:

```text
global_BIC_not_supported
```

It is not automatically equivalent to overall F8 failure.

#### F8b — Held-out predictive support

Minimum principle:

```text
held-out LL improvement > 0
```

The improvement must be obtained without test-set threshold selection.

Purpose:

- determine whether the cross-domain augmentation improves out-of-sample prediction.

#### F8c — Stability under resampling or subject/session validation

Candidate methods:

```text
cross-validated held-out likelihood
leave-one-subject-out evaluation
leave-one-session-out evaluation
bootstrap stability
null comparison against negative controls
```

Exact fold, subject, and stability thresholds must be fixed in the v3 configuration before real-data ingestion.

#### F8d — Effective complexity

Candidate corrections:

```text
effective sample size BIC
state-density-corrected BIC
effective parameter count
occupied-state complexity
estimable-transition complexity
hierarchical or subject-level model selection
```

The correction must use information available independently of the final real-data outcome.

### 14.3 Prospective v3 F8 decision classes

The v3 implementation should distinguish:

```text
F8_PASS_GLOBAL
```

Conventional BIC and predictive support both favor the augmented model.

```text
F8_PASS_BIC_LIMITED
```

Held-out predictive support, controls, registration, and effective-complexity diagnostics support the lead, while conventional global BIC remains negative.

```text
F8_INCONCLUSIVE_COMPLEXITY
```

Held-out improvement exists, but stability or effective-complexity support is insufficient.

```text
F8_FAIL_PREDICTIVE
```

No held-out predictive improvement survives the registered analysis.

### 14.4 Non-veto principle

For III.2E v3:

> A negative conventional BIC value alone will not invalidate a registered localized predictive lead and will not, by itself, force the entire F8 gate to fail.

The result must instead be reported with its exact complexity qualification.

### 14.5 Current v2 result under the prospective vocabulary

The current surrogate result would be described as:

```text
F8a conventional global BIC:
not supported

F8b held-out predictive support:
supported

F8c stability:
partially supported by F7 and controls, not yet fully implemented

F8d effective complexity:
diagnostic only, not yet an official v3 decision rule

overall interpretation:
BIC-limited localized predictive support
```

The current v2 files must not be retroactively altered. The prospective v3 implementation will generate new versioned outputs.

---

## 15. Gate-Sensitivity Diagnostic Layer

The sensitivity layer explains gate behavior without silently changing official v2 outputs.

### 15.1 F7 subject sensitivity

Current output:

```text
phase3_2e_f7_subject_sensitivity.csv
```

Questions:

- Is the lead present in at least the registered minimum number of subjects?
- Is the effect driven by one subject?
- Is a stable responder subgroup plausible?
- Does synchronization quality explain heterogeneity?

Current result:

```text
selected subjects: 1/10
required subjects: 1
official F7 status: PASS
```

### 15.2 F8 complexity sensitivity

Current output:

```text
phase3_2e_f8_complexity_sensitivity.csv
```

Current diagnostics:

```text
BIC improvement:                    -132.193071...
AIC improvement:                    -19.862965...
held-out LL improvement:            +6.068517...
critical complexity penalty scale:  approximately 0.0841
```

Interpretation:

- the augmented model improved held-out prediction;
- nominal global complexity penalties were not compensated;
- the result is predictive but globally BIC-limited;
- the v3 correction must address effective sample size and effective complexity prospectively.

### 15.3 F12 epsilon saturation sensitivity

Current output:

```text
phase3_2e_f12_epsilon_saturation_sensitivity.csv
```

Current result:

```text
official saturation fraction: 0.0
distance to warning: -0.95
official F12 status: PASS
```

### 15.4 Combined sensitivity outputs

```text
phase3_2e_gate_sensitivity_report.json
phase3_2e_gate_sensitivity_summary.txt
phase3_2e_f7_subject_sensitivity.csv
phase3_2e_f8_complexity_sensitivity.csv
phase3_2e_f12_epsilon_saturation_sensitivity.csv
```

---

## 16. Result Classes

### 16.1 Empirical strong candidate

Requires real synchronized data and:

- synchronization validation passes;
- required controls pass;
- held-out predictive support passes;
- registered conditional lift passes;
- adaptive subject consistency passes;
- overall v3 F8 complexity-aware support passes;
- primary registration and multiple-comparison guards pass;
- no critical saturation warning;
- sync-break and target-shuffle controls collapse;
- provenance permits an empirical claim.

Status:

```text
candidate_signal
```

### 16.2 Primary localized predictive lead

Requirements:

- lead occurs in a registered primary region;
- positive conditional lift;
- positive held-out likelihood;
- required controls pass;
- adaptive subject threshold passes;
- no critical saturation;
- conventional BIC may remain negative.

Possible status:

```text
primary_localized_predictive_lead
```

or:

```text
primary_localized_predictive_lead_bic_limited
```

### 16.3 Exploratory lead

A positive row occurs outside the primary registered region or lacks one or more required supports.

Status:

```text
exploratory_lead
```

### 16.4 Synchronized null result

Requires:

- valid synchronization;
- valid controls;
- no robust registered conditional lift;
- no hidden stable signal in diagnostics.

Status:

```text
synchronized_null_result
```

### 16.5 Invalid synchronization result

Status:

```text
sync_invalid
```

No synchronized cross-domain interpretation is permitted.

### 16.6 Surrogate result

Any result derived from surrogate input must include a surrogate prefix or explicit provenance flag.

The current result is:

```text
surrogate_primary_localized_predictive_lead_bic_limited
```

---

## 17. Required Outputs

### 17.1 Schema and synchronization

```text
phase3_2e_schema_report.json
phase3_2e_schema_summary.txt
phase3_2e_sync_report.json
phase3_2e_sync_summary.txt
phase3_2e_loader_report.json
phase3_2e_loader_summary.txt
```

### 17.2 Windows, features, and alignment

```text
phase3_2e_loaded_synchronized_windows.csv
phase3_2e_loader_metadata.json
phase3_2e_windows.csv
phase3_2e_window_inventory.csv
phase3_2e_windows_report.json
phase3_2e_windows_summary.txt
phase3_2e_features.csv
phase3_2e_feature_specs.json
phase3_2e_features_report.json
phase3_2e_features_summary.txt
phase3_2e_alignment_inventory.csv
phase3_2e_alignment_metadata.json
phase3_2e_alignment_report.json
phase3_2e_alignment_summary.txt
```

### 17.3 Conditional outputs

```text
phase3_2e_conditional_model_scores.csv
phase3_2e_conditional_pair_scores.csv
phase3_2e_conditional_summary.json
phase3_2e_conditional_summary.txt
phase3_2e_conditional_report.json
```

### 17.4 Control outputs

```text
phase3_2e_control_runs.csv
phase3_2e_control_pair_scores.csv
phase3_2e_controls.json
phase3_2e_controls_summary.txt
phase3_2e_controls_report.json
```

### 17.5 Metrics and gate outputs

```text
phase3_2e_gates.json
phase3_2e_multiple_comparison_guard.json
phase3_2e_metrics_table.csv
phase3_2e_metrics_summary.txt
phase3_2e_metrics_report.json
```

### 17.6 Gate-sensitivity outputs

```text
phase3_2e_gate_sensitivity_report.json
phase3_2e_gate_sensitivity_summary.txt
phase3_2e_f7_subject_sensitivity.csv
phase3_2e_f8_complexity_sensitivity.csv
phase3_2e_f12_epsilon_saturation_sensitivity.csv
```

### 17.7 Diagnostics and runner outputs

```text
diagnostics/phase3_2e_diagnostics_report.json
diagnostics/phase3_2e_diagnostics_summary.txt
phase3_2e_runner_report.json
phase3_2e_runner_summary.txt
```

### 17.8 Audit outputs

```text
audit_bundle/cdr_phase3_2e_audit_bundle_<timestamp>.zip
audit_bundle/cdr_phase3_2e_audit_bundle_<timestamp>_bundle_report.json
```

---

## 18. Safe-Resume Operational Policy

The Phase III.2E runner supports safe reuse of expensive outputs.

Recommended command after conditional and controls already exist:

```powershell
python -m src.phase3_2e_runner --safe-resume
```

This mode:

- validates existing conditional outputs;
- validates existing control outputs;
- does not execute conditional again;
- does not execute controls again;
- recalculates only the downstream lightweight stages as required.

Full heavy rebuild requires explicit confirmation:

```powershell
python -m src.phase3_2e_runner --force-rebuild --confirm-heavy-rebuild
```

The heavy rebuild command must not be used accidentally.

The audit-bundle builder is read-only and does not rerun any analytical stage.

---

## 19. Interpretation Guardrails

### 19.1 Allowed wording for the current surrogate result

```text
The Phase III.2E synchronized surrogate execution identified a registered
primary localized predictive lead that survived the required negative controls
and improved held-out likelihood. The conventional global BIC criterion did not
favor the augmented model, so the result is classified as BIC-limited.
```

### 19.2 Not allowed for the current surrogate result

```text
The surrogate proves EEG–QRNG coupling.
The surrogate establishes causal quantum-conscious interaction.
The result changes quantum mechanics or relativity.
The current lead is empirical biological evidence.
```

### 19.3 F8 wording

Allowed:

```text
The conventional global BIC subcriterion was not supported.
The lead remains predictively supported but BIC-limited.
A negative conventional BIC does not by itself prove absence of structure.
```

Avoid:

```text
F8 disproved the lead.
The structure is invalid because BIC was negative.
Any negative BIC automatically eliminates localized predictive evidence.
```

---

## 20. Requirements for the Next Real-Data Execution

Before III.2E v3 processes real synchronized data, the following must be frozen:

1. final schema contract;
2. acquisition clock and synchronization mode;
3. synchronization thresholds;
4. primary windows and lags;
5. primary model pairs;
6. subject and session split logic;
7. adaptive F7 thresholds;
8. composite F8 implementation;
9. effective complexity method;
10. cross-validation or resampling rule;
11. required negative controls;
12. multiple-comparison guard;
13. epsilon grid and saturation rule;
14. empirical claim vocabulary;
15. versioned output paths that do not overwrite v2.

The v3 configuration must be committed before inspecting final real-data outcomes.

---

## 21. Current Conclusion

Phase III.2E v2 is complete as a synchronized surrogate validation.

The pipeline successfully demonstrated that it can:

- detect a registered primary localized predictive lead;
- preserve leakage-safe estimation;
- distinguish primary and exploratory regions;
- enforce synchronized negative controls;
- support heterogeneous subject-level discovery;
- identify held-out predictive improvement;
- diagnose conventional complexity penalties;
- avoid epsilon saturation;
- preserve surrogate provenance;
- complete safe-resume execution;
- generate a reproducibility audit bundle.

The current result is:

```text
surrogate_primary_localized_predictive_lead_bic_limited
```

The conventional global BIC subcriterion was not supported. That limitation is preserved in the v2 record.

For III.2E v3, F8 will be implemented as a composite complexity-aware predictive gate. Conventional BIC will remain reported, but a negative BIC value alone will not invalidate a registered localized predictive lead or automatically force the complete F8 gate to fail.

The next scientific stage is a prospectively registered execution with real synchronized EEG–QRNG data.
