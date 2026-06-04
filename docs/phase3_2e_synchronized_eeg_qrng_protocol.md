# Phase III.2E — Synchronized EEG–QRNG Protocol and Gate-Sensitivity Validation

## 1. Protocol Status

**Phase:** III.2E  
**Title:** Synchronized EEG–QRNG Protocol and Gate-Sensitivity Validation  
**Project:** CDR — Core Distinguishability Relativity  
**Status:** `protocol_ready`  
**Empirical execution status:** `pending synchronized EEG–QRNG data`  
**Current role:** protocol definition, schema contract, synchronization validation design, and gate-sensitivity framework  

Phase III.2E is designed as the synchronized continuation of Phase III.2A/B/C/D.

The previous subphases tested regime-aware, lagged, conditional, and multichannel EEG–QRNG relations using Sleep-EDF EEG data and ANU QRNG data. Those sources were real, but they were **not co-acquired under a shared physical clock**.

Phase III.2E changes the experimental question.

Instead of asking whether EEG and QRNG can be aligned post hoc, Phase III.2E asks whether EEG–QRNG coupling becomes detectable when both systems are collected under true temporal synchronization.

---

## 2. Scientific Motivation

Phases III.1 and III.2A/B/C/D produced a non-confirmatory but scientifically useful result.

The current consolidated status after Phase III.2D is:

```text
status: exploratory_lead_with_saturation_warning
controls: passed
multichannel_available: True
MC EEG states: 9
valid MC conditional rows: 27587
multichannel positive lift rows: 7
multichannel strong candidate rows: 0
F7: failed
F8: failed
F12: warning

The correct interpretation is:

Phase III.2D increased exploratory sensitivity through multichannel EEG enrichment.
It produced multichannel positive-lift rows and passed negative controls.
However, it did not produce a strong candidate because subject consistency, BIC support, and ε-saturation constraints were not satisfied.

Therefore, Phase III.2E is justified not as an attempt to force a positive result, but as a stricter synchronized protocol capable of separating three possibilities:

A) No EEG–QRNG coupling exists under the tested conditions.
B) The current gates F7, F8, and F12 are too rigid for subtle neural signals.
C) The previous non-synchronized public-data design was methodologically insufficient.
3. Central Research Question
Does QRNG_t add robust predictive information about EEG_{t+τ}, or does EEG_t add robust predictive information about QRNG_{t+τ}, when EEG and QRNG are truly synchronized in time?

The main directional question is:

QRNG_t → EEG_{t+τ}

The reverse direction is also tested:

EEG_t → QRNG_{t+τ}

The reverse direction is included to identify asymmetry, apparent causality, bidirectional artifacts, temporal leakage, and methodological instability.

4. Null and Alternative Hypotheses
4.1 Null Hypothesis — H0
H0:
In synchronized EEG–QRNG data, QRNG_t does not add robust predictive information about EEG_{t+τ}, and EEG_t does not add robust predictive information about QRNG_{t+τ}, beyond each domain's own autoregressive transition structure.

Formally, the augmented conditional models should not improve held-out predictive structure relative to their corresponding baseline models:

P(EEG_{t+τ} | EEG_t, QRNG_t) ≈ P(EEG_{t+τ} | EEG_t)

P(QRNG_{t+τ} | QRNG_t, EEG_t) ≈ P(QRNG_{t+τ} | QRNG_t)
4.2 Alternative Hypothesis — H1
H1:
In synchronized EEG–QRNG data, there exists at least one pre-registered window scale, lag range, and regime in which the augmented conditional model exhibits robust conditional lift, passes negative controls, generalizes across subjects, improves held-out likelihood, survives model-complexity penalties, and avoids critical ε saturation.
5. Scope of Phase III.2E

Phase III.2E has two axes.

5.1 Axis 1 — Real Synchronization Protocol

This axis defines the conditions under which EEG and QRNG can be considered valid for synchronized CDR analysis.

Required properties:

shared or traceable clock
UTC timestamps
monotonic sample timing
bounded clock drift
valid synchronization quality
no cross-subject QRNG reuse
no unregistered artificial resampling
no missing timestamps in critical fields
5.2 Axis 2 — Gate-Sensitivity Diagnostic Layer

This axis investigates why F7, F8, and F12 failed in Phase III.2D.

This layer does not replace the official gates. It runs in parallel and produces explanatory diagnostics.

Target questions:

F7:
Did subject consistency fail because there is no signal, because only a subgroup responds, or because subject-level aggregation is too rigid?

F8:
Did BIC fail because the augmented model is genuinely worse, or because BIC over-penalizes subtle high-dimensional neural state spaces?

F12:
Did ε saturation occur because of overfitting, sparse states, grid miscalibration, or a real but nonlinearly expressed signal?
6. Required Data Contract

Phase III.2E expects synchronized EEG–QRNG data in file-based tabular form.

The minimum accepted unit is a synchronized analysis window.

Each row must represent one subject/session/window combination.

6.1 Required Identity Columns
subject_id
session_id
window_id
6.2 Required Timing Columns
window_start_utc
window_end_utc
eeg_timestamp_utc
qrng_timestamp_utc
sync_quality
clock_drift_ms
6.3 Required EEG Columns

At minimum:

eeg_channel
eeg_delta_power
eeg_theta_power
eeg_alpha_power
eeg_beta_power
eeg_entropy

Optional but recommended:

eeg_gamma_power
eeg_hjorth_mobility
eeg_hjorth_complexity
eeg_line_length
eeg_signal_std
eeg_artifact_score
eeg_quality_score
6.4 Required Multichannel EEG Columns

For multichannel III.2E analysis:

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
6.5 Required QRNG Columns

At minimum:

qrng_state
qrng_bit_balance
qrng_transition_rate
qrng_entropy

Optional but recommended:

qrng_run_length_mean
qrng_run_length_std
qrng_run_length_max
qrng_compressibility_proxy
qrng_surprise_index
qrng_transition_asymmetry
qrng_quality_score
7. Synchronization Requirements

A synchronized dataset is valid only if all minimum synchronization requirements are satisfied.

7.1 Timestamp Requirements
window_start_utc must be present.
window_end_utc must be present.
eeg_timestamp_utc must be present.
qrng_timestamp_utc must be present.
Timestamps must be parseable as UTC.
Timestamps must be monotonic within each subject/session.
No EEG or QRNG timestamp may cross subject/session boundaries.
7.2 Synchronization Quality Requirements

Default thresholds:

sync_quality >= 0.95
abs(clock_drift_ms) <= 50

A row failing either threshold is considered invalid for confirmatory analysis.

7.3 Clock Drift Requirements

Clock drift must be estimated or recorded for each window or session.

Accepted drift modes:

hardware_shared_clock
software_timestamp_aligned
posthoc_clock_drift_corrected
external_sync_marker

Unaccepted drift modes:

unknown_clock_relation
manual_alignment_without_log
random_pairing
posthoc_alignment_to_maximize_signal
7.4 Sync Failure Status

If synchronization validation fails, the empirical pipeline must stop before conditional CDR.

Failure status:

status: sync_invalid

No positive claim is allowed from a sync-invalid dataset.

8. Windowing Strategy

Phase III.2E uses multi-resolution windows.

8.1 Primary Window Sizes
5s
10s

These are the primary window sizes because they balance temporal sensitivity with EEG band-power stability.

8.2 Secondary Window Sizes
1s
2s
30s

Interpretation:

1s and 2s:
Higher temporal resolution, higher noise sensitivity.

5s and 10s:
Primary compromise between temporal precision and EEG feature stability.

30s:
Compatibility window with Sleep-EDF-style epochs, but less ideal for fine-grained synchronization.
8.3 Window Validity

A window is valid only if:

EEG features are available.
QRNG features are available.
sync_quality passes threshold.
clock_drift_ms passes threshold.
No timestamp is missing.
No cross-subject or cross-session leakage occurs.
9. Lag Strategy

Phase III.2E uses directional lags.

9.1 Primary Lags
0
+1
+2

Interpretation:

lag 0:
Concurrent window.

lag +1:
QRNG_t or EEG_t predicts the next synchronized window.

lag +2:
QRNG_t or EEG_t predicts two windows ahead.
9.2 Secondary Lags
-5
-3
-1
+3
+5

Secondary lags are exploratory and must remain separated from primary inference.

9.3 Lag Unit

The lag unit is the selected window size.

Example:

window_size = 5s
lag = +1
QRNG_t → EEG_{t+5s}
10. Conditional CDR Models
10.1 Single-Channel EEG Models
E0:
P(EEG_{t+τ} | EEG_t)

E1:
P(EEG_{t+τ} | EEG_t, QRNG_t)

Conditional lift:

lift_EEG = ε(E1) - ε(E0)
10.2 QRNG Models
E2:
P(QRNG_{t+τ} | QRNG_t)

E3:
P(QRNG_{t+τ} | QRNG_t, EEG_t)

Conditional lift:

lift_QRNG = ε(E3) - ε(E2)
10.3 Multichannel EEG Models
E4:
P(MC_EEG_{t+τ} | MC_EEG_t)

E5:
P(MC_EEG_{t+τ} | MC_EEG_t, QRNG_t)

E6:
P(QRNG_{t+τ} | QRNG_t, MC_EEG_t)

Conditional lifts:

lift_MC_EEG = ε(E5) - ε(E4)

lift_QRNG_MC = ε(E6) - ε(E2)
11. Leakage-Safe Estimator

Phase III.2E must preserve the leakage-safe estimator introduced in Phase III.2C.

The split must be:

reference_train
calibration
test_holdout

Rules:

P₀ is fitted only on reference_train.
Δχ is estimated only on calibration.
ε is selected only on calibration.
ε is confirmed only on test_holdout.
If ε does not improve held-out test likelihood over ε=0, eps_test is set to 0.0.

No confirmatory metric may be selected using the test set.

12. Negative Controls

Phase III.2E requires stricter controls than previous phases.

12.1 Required Controls
within_subject_qrng_shuffle
within_subject_eeg_shuffle
circular_qrng_shift
subject_mismatch
timestamp_jitter_control
phase_randomized_eeg_control
window_permutation_control
conditional_target_shuffle
sync_break_control
12.2 Sync-Break Control

The sync-break control is mandatory.

It deliberately breaks the temporal relationship between EEG and QRNG while preserving marginal distributions.

Expected behavior:

If the observed effect depends on real synchronization, it must collapse under sync_break_control.

If the effect survives sync_break_control, the result is not interpretable as synchronized EEG–QRNG coupling.

13. Official Gates

The official gates remain conservative.

13.1 F2 — Negative Controls

Pass condition:

All required negative controls collapse under the registered tolerance.
13.2 F3 — Holdout Generalization

Pass condition:

The selected ε improves held-out test likelihood relative to ε=0.
13.3 F6 — Conditional Lift

Pass condition:

The augmented model produces conditional lift above the registered threshold.
13.4 F7 — Subject Consistency

Pass condition:

subject_effect_fraction >= registered threshold

Default:

subject_effect_fraction >= 0.60
13.5 F8 — Complexity / BIC

Pass condition:

The augmented model is not rejected by model-complexity criteria.

Primary criterion:

BIC improvement >= 0
13.6 F9 — Ablation

Pass condition:

Removing the hypothesized cross-domain component reduces the effect or removes the candidate signal.
13.7 F10 — Density / Sparsity

Pass condition:

State density remains above the minimum registered transition-per-state threshold.
13.8 F11 — Multiple-Comparison Guard

Pass condition:

The result survives the registered primary-test and exploratory-test separation.
13.9 F12 — Epsilon Saturation

Pass condition:

The result does not depend on critical ε saturation.

Warning condition:

Effect appears only at the maximum ε grid boundary.
14. Gate-Sensitivity Diagnostic Layer

The gate-sensitivity diagnostic layer is explanatory only.

It does not replace official gates.

14.1 F7 Subject Sensitivity

Diagnostics:

F7_official
F7_bootstrap
F7_weighted
F7_cluster
F7_LOSO

Questions:

Is the effect present across subjects?
Is the effect driven by one outlier?
Is there a stable responder subgroup?
Does synchronization quality explain subject-level variability?

Outputs:

phase3_2e_f7_subject_sensitivity.csv
phase3_2e_f7_subject_sensitivity.json
14.2 F8 Complexity Sensitivity

Diagnostics:

F8_BIC_official
F8_AIC_secondary
F8_heldout_LL
F8_cross_validated_LL
F8_state_density_corrected_BIC
F8_effective_complexity
F8_min_transitions_per_state

Questions:

Is the augmented model genuinely worse?
Does held-out likelihood improve while BIC fails?
Is the state space too sparse?
Is the BIC penalty too severe for the available sample size?

Outputs:

phase3_2e_f8_complexity_diagnostic.csv
phase3_2e_f8_complexity_diagnostic.json
14.3 F12 Epsilon Saturation Sensitivity

Diagnostics:

epsilon_grid_short: 0.00–0.20
epsilon_grid_medium: 0.00–0.50
epsilon_grid_full: 0.00–0.80
epsilon_stability_curve
epsilon_selected_on_calibration
epsilon_confirmed_on_test
epsilon_saturation_fraction

Questions:

Does the effect exist at moderate ε?
Does the effect appear only at ε=0.8?
Does the same saturation appear in controls?
Does synchronization reduce saturation?

Outputs:

phase3_2e_f12_epsilon_saturation_diagnostic.csv
phase3_2e_f12_epsilon_saturation_diagnostic.json
15. Result Classes
15.1 Strong Candidate Signal

A strong candidate requires:

sync validation passes
F2 controls pass
F3 holdout passes
F6 conditional lift passes
F7 subject consistency passes
F8 BIC/complexity passes
F9 ablation passes
F10 density passes
F11 multiple-comparison guard passes
F12 has no critical saturation warning
sync_break_control collapses the effect
target_shuffle collapses the effect
effect does not depend on a single subject

Status:

candidate_signal
15.2 Exploratory Lead

An exploratory lead is allowed when:

conditional lift appears
controls pass
but one or more of F7/F8/F12 blocks strong interpretation

Status:

exploratory_lead

or, if ε saturation is involved:

exploratory_lead_with_saturation_warning
15.3 Strong Null Result

A strong null result requires:

sync validation passes
controls are valid
no robust conditional lift appears
gate diagnostics show no hidden stable signal

Status:

synchronized_null_result
15.4 Invalid Sync Result

If synchronization validation fails:

sync_invalid

No EEG–QRNG coupling interpretation is allowed.

16. Required Outputs
16.1 Protocol Outputs
phase3_2e_protocol_status.json
phase3_2e_protocol_summary.txt
16.2 Schema Outputs
phase3_2e_schema_report.json
phase3_2e_schema_failures.csv
16.3 Sync Outputs
phase3_2e_sync_report.json
phase3_2e_sync_summary.txt
phase3_2e_sync_failures.csv
16.4 Feature and Window Outputs
phase3_2e_windows.csv
phase3_2e_features.csv
phase3_2e_feature_specs.json
16.5 Conditional Outputs
phase3_2e_conditional_model_scores.csv
phase3_2e_conditional_pair_scores.csv
phase3_2e_conditional_summary.json
phase3_2e_conditional_summary.txt
16.6 Control Outputs
phase3_2e_control_runs.csv
phase3_2e_control_pair_scores.csv
phase3_2e_controls.json
phase3_2e_controls_summary.txt
16.7 Gate-Sensitivity Outputs
phase3_2e_f7_subject_sensitivity.csv
phase3_2e_f7_subject_sensitivity.json
phase3_2e_f8_complexity_diagnostic.csv
phase3_2e_f8_complexity_diagnostic.json
phase3_2e_f12_epsilon_saturation_diagnostic.csv
phase3_2e_f12_epsilon_saturation_diagnostic.json
16.8 Final Outputs
phase3_2e_gates.json
phase3_2e_metrics_table.csv
phase3_2e_metrics_summary.txt
phase3_2e_diagnostics_report.json
phase3_2e_diagnostics_summary.txt
phase3_2e_runner_report.json
phase3_2e_runner_summary.txt
17. Interpretation Guardrails

Phase III.2E must not claim quantum or relativistic revision from an exploratory lead alone.

Allowed wording for exploratory results:

The synchronized EEG–QRNG protocol produced an exploratory conditional lead.

Not allowed unless all strong-candidate criteria pass:

The experiment proves EEG–QRNG coupling.
The result demonstrates causal quantum-conscious interaction.
The result changes quantum mechanics or relativity.

The framework may discuss theoretical implications only after satisfying the registered empirical criteria.

18. Operational Principle

The operational principle of Phase III.2E is:

Do not search for a positive result.
Build a protocol strict enough that, if the signal exists, it has no methodological escape route.

The prior phases provide the motivation. Phase III.2E provides the synchronized test.

19. Current Protocol Conclusion

Phase III.2E is approved for implementation as:

protocol-ready
sync-contract-based
file-based
offline-compatible
gate-sensitivity-aware

Empirical claims require future synchronized EEG–QRNG data satisfying the schema and synchronization validator.

Until then, Phase III.2E remains a rigorous protocol and validation architecture for the next empirical stage of CDR.