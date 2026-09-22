import numpy as np
import pandas as pd


TARGET = "cancelled_flag"


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names for reproducible processing."""
    result = df.copy()
    result.columns = (
        result.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return result


def remove_blank_and_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove completely blank rows and keep one record per booking ID."""
    result = df.dropna(how="all").copy()

    if "booking_id" in result.columns:
        result = result.drop_duplicates(
            subset=["booking_id"],
            keep="first"
        )

    return result


def validate_target(
    df: pd.DataFrame,
    target: str = TARGET
) -> pd.DataFrame:
    """Keep rows with a known binary target."""
    result = df.copy()
    result[target] = pd.to_numeric(
        result[target],
        errors="coerce"
    )
    result = result[result[target].isin([0, 1])].copy()
    result[target] = result[target].astype(int)
    return result


def standardize_payment_method(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize known payment-method label variants."""
    result = df.copy()

    if "payment_method" not in result.columns:
        return result

    mapping = {
        "credit card": "Credit Card",
        "debit card": "Debit Card",
        "digital wallet": "Digital Wallet",
        "e-wallet": "Digital Wallet",
        "bank transfer": "Bank Transfer"
    }

    result["payment_method"] = (
        result["payment_method"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map(mapping)
        .fillna("Unknown")
    )

    return result


def correct_rate_and_occupancy_scales(df: pd.DataFrame) -> pd.DataFrame:
    """Correct likely cents-based rates and percentage-point occupancy values."""
    result = df.copy()

    if "nightly_rate" in result.columns:
        result["nightly_rate"] = pd.to_numeric(
            result["nightly_rate"],
            errors="coerce"
        )

        result["nightly_rate_was_cents"] = (
            result["nightly_rate"] > 650
        ).astype(int)

        cents_mask = (
            (result["nightly_rate"] > 650)
            & (result["nightly_rate"] / 100 >= 45)
            & (result["nightly_rate"] / 100 <= 650)
        )

        result.loc[cents_mask, "nightly_rate"] = (
            result.loc[cents_mask, "nightly_rate"] / 100
        )

        result.loc[
            (result["nightly_rate"] < 45)
            | (result["nightly_rate"] > 650),
            "nightly_rate"
        ] = np.nan

    if "expected_occupancy_pct" in result.columns:
        result["expected_occupancy_pct"] = pd.to_numeric(
            result["expected_occupancy_pct"],
            errors="coerce"
        )

        result["occupancy_was_percent_points"] = (
            result["expected_occupancy_pct"] > 1
        ).astype(int)

        percent_mask = (
            (result["expected_occupancy_pct"] > 1)
            & (result["expected_occupancy_pct"] <= 100)
        )

        result.loc[percent_mask, "expected_occupancy_pct"] = (
            result.loc[percent_mask, "expected_occupancy_pct"] / 100
        )

        result.loc[
            (result["expected_occupancy_pct"] < 0)
            | (result["expected_occupancy_pct"] > 1),
            "expected_occupancy_pct"
        ] = np.nan

    return result


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run the core raw-data cleaning sequence."""
    result = standardize_columns(df)
    result = remove_blank_and_duplicate_rows(result)
    result = validate_target(result)
    result = standardize_payment_method(result)
    result = correct_rate_and_occupancy_scales(result)
    return result
