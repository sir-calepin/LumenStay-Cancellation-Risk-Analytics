import numpy as np
import pandas as pd


def build_booking_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create booking-time features for cancellation prediction."""
    result = df.copy()

    for column in ["booking_created_date", "arrival_date"]:
        if column in result.columns:
            result[column] = pd.to_datetime(
                result[column],
                errors="coerce"
            )

    if "booking_created_date" in result.columns:
        result["booking_month"] = (
            result["booking_created_date"].dt.month
        )

    if "arrival_date" in result.columns:
        result["arrival_month"] = (
            result["arrival_date"].dt.month
        )
        result["arrival_day_of_week"] = (
            result["arrival_date"].dt.day_name()
        )

    if "agency_code" in result.columns:
        result["agency_code"] = (
            result["agency_code"]
            .fillna("No Agency")
            .replace("", "No Agency")
        )

    if "days_since_last_stay" in result.columns:
        result["no_prior_stay"] = (
            result["days_since_last_stay"].isna()
        ).astype(int)

    for column in [
        "children_count",
        "market_price_index",
        "external_risk_score"
    ]:
        if column in result.columns:
            result[f"{column}_missing"] = (
                result[column].isna()
            ).astype(int)

    if {"weekend_nights", "weekday_nights"}.issubset(result.columns):
        result["total_nights"] = (
            result["weekend_nights"].fillna(0)
            + result["weekday_nights"].fillna(0)
        )

        result["weekend_night_share"] = np.where(
            result["total_nights"] > 0,
            result["weekend_nights"].fillna(0)
            / result["total_nights"],
            0
        )

    guest_columns = [
        column for column in [
            "adults_count",
            "children_count",
            "infants_count"
        ] if column in result.columns
    ]

    if guest_columns:
        result["party_size"] = (
            result[guest_columns]
            .fillna(0)
            .sum(axis=1)
        )

    if {"prior_bookings", "prior_cancellations"}.issubset(result.columns):
        result["past_cancellation_rate"] = np.where(
            result["prior_bookings"] > 0,
            result["prior_cancellations"]
            / result["prior_bookings"],
            0
        )

    if "market_price_index" in result.columns:
        result["above_market_rate"] = (
            result["market_price_index"] > 1
        ).astype(int)

    if "lead_time_days" in result.columns:
        bins = [-1, 7, 30, 90, 180, 365]
        labels = [
            "0-7 days",
            "8-30 days",
            "31-90 days",
            "91-180 days",
            "181-365 days"
        ]

        result["lead_time_group"] = pd.cut(
            result["lead_time_days"],
            bins=bins,
            labels=labels,
            include_lowest=True
        ).astype("string")

    if {"customer_type", "lead_time_group"}.issubset(result.columns):
        result["customer_type_lead_time"] = (
            result["customer_type"].astype("string")
            + "_"
            + result["lead_time_group"].astype("string")
        )

    return result


def get_candidate_features(df: pd.DataFrame) -> list[str]:
    """Return booking-time candidate predictors."""
    excluded = {
        "booking_id",
        "cancelled_flag",
        "booking_created_date",
        "arrival_date",
        "pre_arrival_confirmation_response",
        "external_score_version"
    }

    return [
        column for column in df.columns
        if column not in excluded
    ]
