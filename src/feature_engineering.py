import pandas as pd

def build_sales_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate lag and rolling features from historical sales.
    """
    already_sorted = df.equals(df.sort_values(by=["Store", "Date"]))
    if not already_sorted:
        pass

    sales_history = df[["Store", "Date", "Sales"]].copy()

    # previous available sales observations
    grouped_sales = sales_history.groupby("Store")["Sales"]

    sales_history["sales_lag_1"] = grouped_sales.shift(1)
    sales_history["sales_lag_7"] = grouped_sales.shift(7)

    # rolling averages using only previous sales observations
    sales_history["sales_rolling_mean_7"] = grouped_sales.transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=7).mean()
    )

    sales_history["sales_rolling_mean_28"] = grouped_sales.transform(
        lambda x: x.shift(1).rolling(window=28, min_periods=28).mean()
    )

    # use origin_date as marge key
    sales_history = sales_history.rename(columns={"Date": "origin_date"})

    return sales_history

def build_target_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare target-date calendar and promotional features."""

    target_features = df[
        ["Store", "Date", "DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday"]
    ].copy()

    target_features = target_features.rename(
        columns={
            "Date": "target_date",
            "DayOfWeek": "target_day_of_week",
            "Promo": "target_promo",
            "StateHoliday": "target_state_holiday",
            "SchoolHoliday": "target_school_holiday",
        }
    )

    # Extract calendar features from the target date
    target_features["target_month"] = target_features["target_date"].dt.month

    target_features["target_year"] = target_features["target_date"].dt.year

    return target_features


def build_features(
    fold_df: pd.DataFrame, sales_history: pd.DataFrame, target_features: pd.DataFrame
) -> pd.DataFrame:
    """Merge both feature tables into a fold's dataset and one-hot encode holidays."""

    # Attach historical Sales features using Store and origin_date
    feature_df = fold_df.merge(
        sales_history[
            [
                "Store",
                "origin_date",
                "sales_lag_1",
                "sales_lag_7",
                "sales_rolling_mean_7",
                "sales_rolling_mean_28",
            ]
        ],
        on=["Store", "origin_date"],
        how="left",
        validate="many_to_one",
    )

    # Attach features known in advance for the target date
    feature_df = feature_df.merge(
        target_features, on=["Store", "target_date"], how="left", validate="many_to_one"
    )

    # One-hot encode StateHoliday with fixed categories
    for holiday in ["a", "b", "c", "0"]:
        feature_df[f"target_state_holiday_{holiday}"] = (
            feature_df["target_state_holiday"] == holiday
        ).astype("int8")

    feature_df = feature_df.drop(columns=["target_state_holiday"])

    return feature_df
