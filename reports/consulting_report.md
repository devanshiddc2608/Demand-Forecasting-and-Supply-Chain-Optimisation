# Demand Forecasting & Supply Chain Optimisation
## Consulting Engagement Report — Rossmann Retail Network

---

## Executive Summary

This engagement applied machine learning-based demand forecasting (Facebook
Prophet, XGBoost, and a weighted ensemble) across 1,115 retail locations to
replace a uniform, non-differentiated safety stock policy with a
forecast-driven, segment-specific inventory strategy.

**Key Findings:**
- Current blanket safety stock policy costs €27,011,618 annually in carrying
  cost; the recommended forecast-driven policy reduces this to €9,566,005 —
  a **64.6% reduction**, saving **€17,445,613 per year**
- The store network splits into 775 A-class (69.5% of revenue), 235 B-class
  (21.1%), and 105 C-class (9.4%) locations, each requiring differentiated
  service level targets (99% / 95% / 90% respectively)
- The Ensemble forecasting model achieves the lowest error rate of the three
  approaches tested, with an overall MAPE of 8.8% across the network

**Expected Financial Impact:** €17.4M in annual carrying cost reduction,
achieved without increasing stockout risk, by replacing an undifferentiated
14-day blanket buffer with statistically-sized, class-specific safety stock.

---

## Current State Assessment

**Demand Pattern Analysis:** Daily sales across the network show strong,
consistent weekly seasonality (Saturday peaks) and clear year-on-year growth.
Demand variability differs sharply by store: 83 stores exhibit stable (X-class)
demand, 1,030 show moderate variability (Y-class), and 2 show highly erratic
(Z-class) demand patterns requiring specialised handling.

**Inventory Policy Analysis:** The current policy holds a flat 14 days of
supply as safety stock for every store regardless of size or volatility. This
overprotects low-variability, low-volume C-class stores while still leaving
volatile A-class locations under-buffered relative to their actual demand risk.

**Key Risk Areas:** The two Z-class stores carry disproportionate forecasting
risk and should be flagged for manual planner review rather than automated
policy. A-class stores with high Store_MAPE values represent the highest
financial exposure per unit of forecast error, given their revenue contribution.

---

## Demand Forecast Results

**Methodology:** Three forecasting approaches were tested and compared
out-of-sample: Facebook Prophet (component-based, interpretable), XGBoost
(feature-engineered gradient boosting), and an inverse-MAPE-weighted ensemble
of both.

**Accuracy Results:** The Ensemble model achieved the lowest overall MAPE,
outperforming both individual models. Forecast difficulty (MAPE) varies
modestly by store class, with B-class stores showing marginally higher error
than A-class — likely reflecting their smaller, noisier sales base relative
to A-class stores' more stable high-volume patterns.

**Confidence Interval Interpretation:** Forecast uncertainty bands should be
used by replenishment planners to set order quantities above the point
forecast — ordering to the point estimate alone results in stockouts in
approximately half of all replenishment cycles by definition.

**Limitations:** This analysis operates on store-level aggregate revenue
rather than individual SKU-level unit data, since the underlying dataset does
not provide product-level granularity. The same methodology applies directly
to SKU-level forecasting in a production ERP environment.

---

## Inventory Optimisation Findings

**Total Potential Cost Reduction:** €17,445,613 annually, a 64.6% reduction
in safety stock carrying cost, achieved by replacing the blanket 14-day
policy with statistically-derived, class-specific safety stock levels.

**ABC-XYZ Policy Recommendations:**
- **AX/AY stores:** Lean automated reordering with elevated service level
  (99%), weekly review cycle
- **AZ stores:** Highest safety stock buffer, mandatory manual planner
  oversight given high revenue stake combined with erratic demand
- **BX/BY/BZ stores:** Standard automated policy at 95% service level,
  fortnightly review
- **CX/CY/CZ stores:** Lean min-max policy at 90% service level; CZ stores
  (2 in the network) are candidates for SKU/store rationalisation review

**Highest-Impact Stores:** The ten highest-saving-potential stores (led by
Store 817, Store 262, and Store 1114) collectively represent a
disproportionate share of total improvement opportunity and should be
prioritised for the first implementation wave.

---

## Recommendations

**1. Deploy the Ensemble Forecasting Model for A and B Class Stores**
Replace ad-hoc/manual forecasting with the Prophet-XGBoost ensemble for all
A and B class locations, where forecast accuracy carries the highest
financial stakes. *Expected impact: majority of the €17.4M saving.* *Effort:
Medium (requires monthly model retraining pipeline).* *Owner: Demand
Planning Lead, 0-60 days.*

**2. Recalibrate Safety Stock to Differentiated Service Levels**
Replace the uniform 14-day blanket policy with 99% / 95% / 90% targets for
A/B/C classes respectively. *Expected impact: €17.4M annual reduction.*
*Effort: Low (formula-driven, no system change required).* *Owner: Inventory
Planning, 0-30 days.*

**3. Align Replenishment Delivery Windows to Weekly Demand Peaks**
EDA confirms Saturday as the consistent peak demand day; shift delivery
schedules to a Thursday-Friday arrival window network-wide. *Expected
impact: reduced weekend stockout incidence.* *Effort: Low.* *Owner:
Logistics, 0-30 days.*

**4. Mandate 4-Week Advance Sharing of Promotional Calendars**
Promotional activity is consistently one of the top predictive features in
the forecasting model; current late notification causes avoidable
promotional stockouts. *Expected impact: reduction in promotion-period
stockout cost.* *Effort: Low (process change only).* *Owner: Commercial /
Supply Chain liaison, 0-30 days.*

**5. Establish Dedicated Planner Coverage for Top 10 Priority Stores**
Assign a dedicated demand planner to manually review the ten highest-saving-
potential stores weekly for one quarter. *Expected impact: captures
disproportionate share of total saving fastest.* *Effort: Medium (staffing
allocation).* *Owner: Regional Planning Manager, 0-90 days.*

---

## Implementation Roadmap

**Phase 1 — Quick Wins (0-30 days):** Recalibrate safety stock formulas
network-wide; shift replenishment delivery windows; establish promotional
calendar sharing protocol.

**Phase 2 — Process Changes (30-90 days):** Deploy ensemble forecasting
model into the weekly planning cycle for A/B class stores; assign dedicated
planner coverage to top 10 priority stores; formalise ABC-XYZ policy matrix
into standard operating procedure.

**Phase 3 — System & Capability Investment (90+ days):** Evaluate
integration of the forecasting pipeline into an enterprise planning system
(SAP IBP or equivalent); extend methodology to SKU-level granularity;
build automated model retraining and monitoring infrastructure.