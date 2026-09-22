# LumenStay Cancellation Risk Analytics

> A reproducible machine-learning project that predicts hotel reservation cancellation risk at booking confirmation and translates risk scores into targeted, customer-friendly operational actions.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-02569B)
![Status](https://img.shields.io/badge/status-portfolio%20project-2E8B57)

## Business Problem

Hotel reservation cancellations create revenue loss and operational uncertainty. When a reservation cancels, a room may go unsold, occupancy forecasts become less reliable, and staffing, inventory, pricing, and waitlist decisions become harder.

This project develops a **booking-time cancellation-risk model** for LumenStay, a fictional hotel business. The model estimates the probability that a reservation will cancel before arrival so the business can prioritize retention support and inventory recovery.

The purpose is **not** to penalize guests or automatically discount every booking. It is to support proportionate decisions:

- Standard service for low-risk bookings.
- Low-cost confirmation or self-service modification options for moderate-risk bookings.
- Personalized support for high-risk bookings when contact permission allows.
- Early inventory recovery attention for high-risk bookings in limited-control channels.

## Target and Prediction Point

**Target:** `cancelled_flag`

- `1` = reservation cancelled before arrival.
- `0` = reservation remained active through arrival.

The model is a supervised binary classifier that estimates:

\[
P(\text{cancelled\_flag}=1)
\]

**Prediction point:** booking confirmation. Only information available at or immediately after booking is used. This prevents future-information leakage and makes the score operationally usable.

Excluded from the booking-time model:

- `booking_id`: unique record identifier.
- `pre_arrival_confirmation_response`: collected during later outreach and unavailable at booking.
- `external_score_version`: constant in the available dataset.
- Raw booking and arrival dates after interpretable date-derived features are created.

## Project Highlights

- Model-ready data: **27,000 reservations**.
- Cancellation rate: **17.43%**.
- Candidate booking-time predictors: **48**.
- Stratified data partition: **60% training / 20% validation / 20% untouched test**.
- Six classification models compared.
- Final selected model: **LightGBM**.
- Validation ROC-AUC: **0.7552**.
- Untouched test ROC-AUC: **0.7494**.
- Untouched test recall: **0.6227**.
- Untouched test average precision: **0.3693**.

## Key EDA Findings

### Cancellation risk by lead time

| Lead-time group | Bookings | Cancellations | Cancellation rate |
|---|---:|---:|---:|
| 0–7 days | 1,647 | 113 | 6.86% |
| 8–30 days | 3,947 | 470 | 11.91% |
| 31–90 days | 13,419 | 2,159 | 16.09% |
| 91–180 days | 7,032 | 1,671 | 23.76% |
| 181–365 days | 955 | 292 | 30.58% |

### Booking channel × lead-time interaction

| Segment | Bookings | Cancellations | Cancellation rate |
|---|---:|---:|---:|
| Online Travel Agency, 181–365 days | 273 | 115 | 42.12% |
| Online Travel Agency, 91–180 days | 2,002 | 719 | 35.91% |
| Online Travel Agency, 31–90 days | 3,854 | 992 | 25.74% |
| Travel Agency, 181–365 days | 85 | 34 | 40.00% |

### Customer type × deposit policy

| Segment | Bookings | Cancellations | Cancellation rate |
|---|---:|---:|---:|
| First-time, refundable deposit | 668 | 177 | 26.50% |
| First-time, no deposit | 13,613 | 3,539 | 26.00% |
| First-time, nonrefundable deposit | 324 | 62 | 19.14% |
| Returning, no deposit | 8,019 | 621 | 7.74% |
| Corporate, no deposit | 3,112 | 218 | 7.01% |

These are predictive associations, not causal estimates. For example, different deposit policies may be selected by guests with different travel certainty.

## Data Preparation

The pipeline handles the following issues:

1. Removes blank import artifacts and duplicate booking IDs.
2. Validates that the target contains only `0` and `1`.
3. Converts booking and arrival dates into calendar features.
4. Standardizes text labels, including payment-method variants.
5. Corrects likely nightly-rate values stored in cents.
6. Converts occupancy values stored as percentage points into proportions.
7. Handles meaningful missingness:
   - Blank agency values become `Direct / No agency`.
   - Missing days since last stay becomes a `no_prior_stay` indicator.
   - Missingness flags are retained for selected variables.
8. Creates booking-time features, including:
   - Lead-time groups.
   - Total nights.
   - Weekend-night share.
   - Party size.
   - Past cancellation rate.
   - Above-market-rate flag.
   - Customer type × lead-time interaction.

## Models Evaluated

All models were trained on the same feature set and compared on the validation set. Hyperparameters were tuned with cross-validation on the training data only.

| Model | Description | Class-imbalance handling | Validation ROC-AUC | Validation average precision |
|---|---|---|---:|---:|
| LASSO Logistic Regression | L1-regularized linear probability model; selects weaker/redundant features by shrinking coefficients toward zero | SMOTE inside the training/CV pipeline | 0.7515 | 0.3743 |
| Decision Tree | Interpretable if-then classification rules | Balanced class weights | 0.7319 | 0.3429 |
| Random Forest | Ensemble of decorrelated decision trees; captures nonlinear relationships and interactions | `balanced_subsample` | 0.7544 | 0.3809 |
| XGBoost | Sequential gradient-boosted trees that correct prior errors | `scale_pos_weight` | 0.7548 | 0.3923 |
| **LightGBM** | Efficient gradient-boosted trees for structured tabular data | `scale_pos_weight` | **0.7552** | **0.3954** |
| Histogram Gradient Boosting | Scikit-learn histogram-based boosting benchmark | Higher positive-class sample weights | 0.7534 | 0.3900 |

### Why LightGBM was selected

LightGBM had the highest validation ROC-AUC and average precision among the six models. Its performance advantage was modest, so LASSO remains a useful transparent benchmark; however, LightGBM was chosen as the final model because it best captured nonlinear effects and interactions in the available validation results.

## Final Model Evaluation

After selecting LightGBM and the validation-based operating threshold, the final pipeline was refit on the combined training and validation data. The untouched test set was used only once for final evaluation.

| Metric | Untouched test result |
|---|---:|
| ROC-AUC | **0.7494** |
| Average precision | **0.3693** |
| Accuracy | 0.7198 |
| Balanced accuracy | 0.6815 |
| Precision | 0.3360 |
| Recall | 0.6227 |
| F1 score | 0.4365 |
| High-risk threshold | 0.5582 |

The high-risk cutoff was selected on the validation set by maximizing F1, which balances precision and recall for the minority cancellation class:

\[
\text{High-risk flag} =
\begin{cases}
1 & \text{if } P(\text{cancellation}) \geq 0.5582 \\
0 & \text{otherwise}
\end{cases}
\]

The model should be used for prioritization, not as a guarantee that a reservation will cancel.

## Feature Interpretation

LightGBM gain-based importance identified these leading predictive signals:

| Rank | Feature | Relative gain importance |
|---:|---|---:|
| 1 | External cancellation-risk score | 16.78% |
| 2 | Days since last stay | 14.26% |
| 3 | Lead time in days | 12.77% |
| 4 | Prior bookings | 10.41% |
| 5 | No loyalty tier recorded | 6.21% |
| 6 | Market-price index | 4.77% |
| 7 | Nightly rate | 3.30% |
| 8 | Full channel-control level | 3.18% |
| 9 | Special requests | 2.32% |
| 10 | Online Travel Agency channel | 2.19% |

Feature importance indicates predictive contribution, not causality. SHAP analysis is included to examine direction and heterogeneous feature effects at the booking level.

## Business Recommendations

### 1. OTA early-recovery workflow

**Target:** High-risk Online Travel Agency reservations made 91 or more days before arrival.

**Why:** OTA cancellations were 35.91% for 91–180-day bookings and 42.12% for 181–365-day bookings, compared with 17.43% overall.

**Action:** Use permitted OTA messages to provide a clear modification path. When a reservation modifies or cancels, trigger early waitlist activation and replacement-demand monitoring rather than automatic broad discounts or overbooking.

### 2. First-time self-service modification pathway

**Target:** First-time, no-deposit bookings with elevated predicted risk and contact permission.

**Why:** This segment represents 50.4% of bookings and has a 26.00% cancellation rate, compared with 7.74% among returning/no-deposit bookings.

**Action:** Send a post-booking self-service message that lets guests modify dates, update details, view policies, and adjust selected stay options. Test value-added benefits only among eligible, highest-risk bookings.

### 3. Risk-adjusted occupancy forecasting

**Target:** Property × arrival-date planning.

**Action:** Aggregate individual cancellation probabilities to estimate expected completed stays:

\[
\text{Expected completed stays}_{p,d} = \sum_{i \in (p,d)} \left(1 - P(\text{cancellation}_i)\right)
\]

Use this as a planning signal for staffing, waitlists, and replacement-demand actions. It is **not** an automatic overbooking rule.

## Minimal Repository Structure

```text
lumenstay-cancellation-risk-analytics/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── data/
│   ├── raw/
│   │   ├── .gitkeep
│   │   └── README.md
│   └── processed/
│       ├── .gitkeep
│       └── data_dictionary.md
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing_feature_engineering.ipynb
│   ├── 03_modeling_validation.ipynb
│   └── 04_interpretation_recommendations.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   └── visualization.py
│
└── docs/
    ├── methodology.md
    └── model_card.md
```

### Directory guide

| Location | Purpose |
|---|---|
| `data/raw/` | Original data if publication is allowed. Otherwise, keep only `README.md` with data-access instructions. |
| `data/processed/` | Model-ready data or a synthetic sample, only if sharing is permitted. |
| `notebooks/` | Numbered, reproducible notebooks for EDA, preprocessing, modeling, and interpretation. |
| `src/` | Reusable Python functions extracted from notebooks. |
| `docs/` | Methodology, model-card, governance, limitations, and monitoring documentation. |

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/lumenstay-cancellation-risk-analytics.git
cd lumenstay-cancellation-risk-analytics
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add data

Place an authorized model-ready data file in `data/raw/` or follow the instructions in `data/raw/README.md`.

Expected target column:

```text
cancelled_flag
```

### 5. Run notebooks in order

```text
01_eda.ipynb
02_preprocessing_feature_engineering.ipynb
03_modeling_validation.ipynb
04_interpretation_recommendations.ipynb
```

## Data Privacy and Publication

Do not publish restricted, proprietary, course-only, or personally identifiable booking data without explicit permission.

Before making this repository public:

- Exclude raw datasets through `.gitignore`.
- Remove booking IDs and other identifying fields from any shared sample.
- Share a synthetic or schema-only sample when source data cannot be distributed.
- Do not commit secrets, credentials, or cloud keys.
- Verify that charts and tables do not reveal confidential operational information.

## Limitations

- The model identifies risk associations; it does not establish causal reasons for cancellation.
- The model predicts whether a booking may cancel, not when it will cancel.
- Customer interventions must be tested through controlled pilots before claiming they reduce cancellations.
- New properties, channels, agencies, and vendor score recalibrations can cause data or model drift.
- Probability thresholds should be reviewed against staffing capacity, outreach cost, customer experience, and incremental revenue.

## Future Work

- Add time-based validation using later booking periods as holdout data.
- Assess probability calibration by risk tier.
- Conduct randomized intervention pilots and estimate uplift.
- Build property × arrival-date expected completed-stay forecasts.
- Monitor data drift, unseen categories, feature distributions, and model performance over time.
- Deploy an internal dashboard for revenue-management and property teams.

## License

Use an MIT License for code if appropriate. The code license does not grant rights to distribute the original booking dataset.


