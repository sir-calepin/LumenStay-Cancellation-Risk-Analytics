# Methodology

## Objective

Predict the probability that a confirmed hotel reservation will cancel before arrival.

## Prediction point

Booking confirmation. Only booking-time information is allowed.

## Data split

- Training: 60%
- Validation: 20%
- Untouched test: 20%

All splits are stratified by `cancelled_flag`.

## Model selection

Candidate models are tuned with cross-validation using training data. Validation ROC-AUC is the primary model-selection criterion, with average precision used as a secondary imbalanced-class metric.

## Final model

LightGBM was selected because it achieved the highest validation ROC-AUC and average precision among the tested models.

## Threshold selection

The high-risk threshold is selected from validation predictions by maximizing F1. The resulting LightGBM threshold was approximately 0.558.

## Leakage prevention

- `booking_id` excluded.
- Later-stage confirmation response excluded.
- Preprocessing is fitted on training data only.
- SMOTE is applied only inside LASSO training folds.
- Test data are not used for tuning or model selection.
