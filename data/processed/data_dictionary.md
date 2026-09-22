# Processed Feature Dictionary

## Target

| Feature | Definition |
|---|---|
| `cancelled_flag` | Binary cancellation outcome: 1 = cancelled before arrival; 0 = active through arrival |

## Excluded fields

| Field | Reason excluded |
|---|---|
| `booking_id` | Unique identifier; not a reusable predictor |
| `pre_arrival_confirmation_response` | Collected after booking; causes future-information leakage |
| `external_score_version` | Constant in the available data |
| Raw booking/arrival dates | Replaced by interpretable calendar features |

## Engineered features

| Feature | Definition |
|---|---|
| `booking_month` | Month in which reservation was created |
| `arrival_month` | Scheduled arrival month |
| `arrival_day_of_week` | Scheduled arrival weekday |
| `no_prior_stay` | Indicates that days since last stay was structurally unavailable |
| `children_count_missing` | Indicates missing child-count value |
| `market_price_index_missing` | Indicates unavailable market-price index |
| `external_risk_score_missing` | Indicates unavailable external risk score |
| `lead_time_group` | Binned booking lead time |
| `total_nights` | Weekend nights plus weekday nights |
| `weekend_night_share` | Weekend nights divided by total nights |
| `party_size` | Adults plus children plus infants |
| `past_cancellation_rate` | Prior cancellations divided by prior bookings, with defined zero-history handling |
| `above_market_rate` | Indicator that market-price index is above 1 |
| `customer_type_lead_time` | Interaction between customer type and lead-time group |
