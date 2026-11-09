# Phase 1 — Business Understanding

*CRISP-DM phase:* 1 of 6
*Status:* complete
*Artifacts produced by this phase:* problem statement, KPIs, cost model, evaluation protocol,
risk register (in `PROJECT_PLAN.md` §7).

---

## 1.1 Domain context

Geothermal power plants (PLTP, *Pembangkit Listrik Tenaga Panas Bumi*) such as Kamojang operate
continuous, round-the-clock processes: steam gathering, separation, turbine expansion, condensing,
cooling-water circulation, and non-condensable-gas (NCG) extraction. Plant availability and
net output depend less on the main turbine than on balance-of-plant auxiliary equipment
(cooling-water pumps, vacuum systems, NCG compressors, and instrumentation). Degradation of these
systems appears first as a slow drift in energy consumption, bearing temperature, or flow, and only
later as a trip.

Two properties make this a suitable target for AIoT analytics:

1. Dense telemetry. Modern plants already log thousands of tags at minute-to-hour resolution;
   the bottleneck is analysis rather than sensing.
2. Rare, expensive failures. A missed fault can cost days of lost generation; too many false
   alarms cause operators to ignore the system entirely (alarm fatigue).

## 1.2 Business problem

> Operators of a geothermal plant cannot reliably distinguish normal load-following behaviour from
> early equipment degradation using static thresholds, because normal behaviour varies with
> ambient conditions, season, and operating mode. The result is either missed incipient faults or
> alarm fatigue from threshold breaches that are not faults.

## 1.3 Data-science problem (translation)

The task is framed as supervised, point-wise binary classification on hourly telemetry, with a
*building-disjoint* (asset-disjoint) evaluation protocol:

- Input: causal features derived from the asset's own recent history plus calendar context.
- Output: probability that the reading at time *t* is anomalous.
- Evaluation: ranking quality on unseen assets (ROC-AUC), operating-point behaviour
  (precision/recall at a chosen threshold), and calibration (Brier).

The task is supervised rather than purely unsupervised because historical annotated incidents
exist in the target dataset. A supervised model with an asset-disjoint split answers the business question of whether a reading on a *new* asset will appear abnormal, while
unsupervised
baselines are kept as comparison points because real plants rarely have complete labels.

The split is asset-disjoint because, in production, every new meter/asset is unseen. A random row
split lets the model memorise asset-specific baselines and reports an accuracy that will not
survive deployment.

## 1.4 KPIs and cost model

| KPI | Definition | Target |
|---|---|---|
| Detection quality | ROC-AUC on held-out assets, 95% bootstrap CI | Report; beat strongest baseline |
| Alarm precision at operating point | TP / (TP + FP) at validation-selected threshold | ≥ 0.50 preferred for maintenance triage |
| Missed-fault rate | 1 − recall at operating point | ≤ 0.25 |
| Early-warning capability | Recall at FPR ≤ 1% | Report (this is the "quiet alarm" regime) |
| Operational cost | `C = c_FN·FN + c_FP·FP` | Minimised at the chosen threshold |

The cost model uses relative costs `c_FN : c_FP = 10 : 1`, since a missed degradation event costs
roughly an order of magnitude more than an unnecessary inspection. The cost-minimising threshold
is derived on the validation set and applied unchanged to the test set. The 10:1 ratio is an
explicit, documented assumption rather than a measurement, and the sensitivity of the chosen
threshold to that ratio is reported in Phase 5.

## 1.5 Stakeholders and how they consume the output

| Stakeholder | Needs | Interface |
|---|---|---|
| Plant operator | "Is this asset behaving abnormally *now*?" | Dashboard live view, threshold slider |
| Maintenance engineer | "Why was this flagged?" | Feature-contribution (SHAP) panel |
| Reliability analyst | "How good is the detector on assets it has never seen?" | Evaluation report, calibration curve |
| Researcher / reviewer | "Can I reproduce this?" | `scripts/reproduce_all.py`, pinned env, seeds |

## 1.6 Success criteria for the whole project

Restated from `PROJECT_PLAN.md` §4 in business language:

1. A detector that generalises to unseen assets and reports its uncertainty faithfully.
2. An operator-facing interface that shows *why* an alarm fired.
3. A reproducible artifact set (code + metrics + figures) such that a third party can regenerate
   every number in the manuscript.
4. A demonstrated transfer path to geothermal auxiliary equipment, with the simulation clearly
   distinguished from plant evidence.

## 1.7 Scope boundaries (explicit non-goals)

- This is not a predictive-maintenance remaining-useful-life model (no failure-time labels available).
- It is not a claim about PLTP Kamojang's real equipment behaviour: no plant data was provided.
- It is not a leaderboard entry: the competition split is unavailable (`BLOCKERS.md` B-001).
- It is not a real-time streaming system; the deployment target is batch/near-real-time scoring over
  an hourly cadence, with an API that could be polled by a SCADA historian.

## 1.8 Exit criteria for Phase 1 (met)

- [x] Problem stated in business terms and translated to a testable ML task.
- [x] KPIs defined with thresholds and an explicit cost model.
- [x] Evaluation protocol fixed *before* seeing model results (asset-disjoint, ROC-AUC primary).
- [x] Stakeholders and interfaces identified.
- [x] Non-goals written down to prevent scope creep and overclaiming.
