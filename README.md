# Sales Forecasting: Evaluating Temporal Validation Strategies

## Overview

This project investigates a business question:

**How should I evaluate a four-week forecasting model to establish whether its predictions will remain reliable when predicting future sales?**

I used the Rossmann Store Sales dataset to study this problem. It provides multiple time series (stores) with historical observations, making it useful for exploring short-horizon forecasting.

The focus of this project was not just training a forecasting model. It was understanding how to evaluate its performance on future periods without introducing temporal leakage.

## Business Problem

A forecasting model may perform well on historical data but struggle when predicting future periods.

For a business relying on forecasts to make operational decisions, a good average score is not enough. Its performance should also be evaluated across different periods to understand how reliable its predictions are over time.

I wanted to investigate how the choice of training window affects this evaluation.

## Reproduce

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate
# Git Bash (Windows)
. .venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt

# Then Run notebooks/02_validation_experiments.ipynb
```

## Dataset

**Dataset:** Rossmann Store Sales (Kaggle)

The dataset contains daily sales records for 1,115 stores.

After filtering to records where `Open == 1`, the training data contained:

- 844,392 observations
- 1,115 stores
- Date range: January 1, 2013 – July 31, 2015

I excluded `Customers` because customer counts for a future target date would not be available when making a forecast.

I also excluded `Open` after filtering to open-store observations.

## Forecasting Setup

I formulated the problem as direct multi-horizon forecasting.

For each store and forecast origin, the model predicts sales at five different horizons:

- 1 day
- 7 days
- 14 days
- 21 days
- 28 days

Each training example contains:

- `origin_date`: when the forecast is made.
- `target_date`: the date whose sales are being predicted.
- `forcast_horizon`: the number of days between the origin and target dates.
- `target_sales`: actual sales on the target date.

I used one global XGBoost model with forecast horizon as an input feature, rather than training a separate model for each horizon.

## Preventing Temporal Leakage

One important decision was how to apply the training cutoff.

Splitting only by `origin_date` is not sufficient because a training example can have an origin date before the cutoff but a target date after it.

For example, if the training cutoff is January 10, a forecast example with:

- Origin date: January 5
- Target date: January 28

cannot be used for training. The actual outcome would not have been available on January 10.

Therefore, I enforced:

`target_date <= training_cutoff`

for every training example.

I also kept all forecast horizons belonging to the same origin date together when splitting the data.

## Validation Strategies

I compared two time-series validation strategies.

### 1. Expanding Window

The training window grows as time progresses. Each new fold retains the historical training data from previous folds and adds newer observations.

### 2. Rolling Window

The training window has a fixed size of 365 days. As time progresses, older observations are removed and newer observations are added.

### Validation Design

Both strategies used the same validation design:

- Initial training window: 365 days.
- Validation block: 28 days (four weeks).
- Step between validation blocks: 28 days.
- Forecast horizons: 1, 7, 14, 21 and 28 days.
- Model configuration and feature definitions: kept the same across strategies.

At the beginning of each validation block, the model was trained using the available training history.

During the validation block, the model parameters remained fixed, but historical features were updated using newly available actual sales as each forecast origin progressed.

This simulates retraining every four weeks while generating fresh forecasts as new observations become available.

## Feature Engineering

The initial feature set included:

**Store and forecast information**

- Store ID
- Forecast horizon

**Target-date information**

- Day of week
- Month
- Year
- Promotion
- School holiday
- State holiday (one-hot encoded)

**Historical sales**

- Lag 1
- Lag 7
- 7-observation rolling mean
- 28-observation rolling mean

Historical sales features were shifted by one observation to ensure that sales on the forecast origin date were not used when generating that day's forecast.

Target-date promotional and calendar features were included under the assumption that this information was known in advance.

## Model

I used `XGBRegressor` with the following configuration:

| Parameter        | Value |
| ---------------- | ----: |
| Objective        |   reg |
| n_estimators     |   500 |
| learning_rate    |  0.05 |
| max_depth        |     6 |
| subsample        |   0.8 |
| colsample_bytree |   0.8 |
| tree_method      |  hist |
| random_state     |    42 |

The same configuration was used across folds and validation strategies.

## Evaluation Metric

My primary metric was Weighted Absolute Percentage Error (WAPE)

WAPE measures total absolute forecasting error relative to total actual sales.

I calculated WAPE separately for each forecast horizon and validation fold.

To compare the two strategies, I calculated:

WAPE difference = ROlling WAPE - Expanding WAPE

A positive difference favours expanding-window validation, while a negative difference favours rolling-window validation.

## Results

### Average WAPE by Forecast Horizon

| Horizon | Expanding | Rolling |
| ------: | --------: | ------: |
|   1 day |    12.87% |  12.65% |
|  7 days |    13.48% |  14.08% |
| 14 days |    13.42% |  14.10% |
| 21 days |    13.37% |  14.08% |
| 28 days |    13.27% |  14.14% |

### Paired WAPE Difference

| Horizon | Mean difference |
| ------: | --------------: |
|   1 day |           -0.21 |
|  7 days |           +0.60 |
| 14 days |           +0.68 |
| 21 days |           +0.71 |
| 28 days |           +0.87 |

The results varied across validation folds. Neither strategy consistently outperformed the other in every fold.

However, expanding-window validation had lower average WAPE at four out of five forecast horizons. Rolling-window validation performed slightly better at the 1-day horizon.

Expanding-window validation also had lower WAPE standard deviation across folds in the horizon-level summaries.

### Decision

I chose expanding-window validation for this experiment.

It performed better on average across most forecast horizons and showed lower variation in WAPE across validation periods.

This is an empirical decision based on this dataset, feature set and model configuration. It does not establish that expanding-window validation is universally better.

## Limitations

- The data was filtered to open-store observations. Consequently, lag and rolling features operate over previous available open-day observations, not necessarily consecutive calendar days.
- Promotional and calendar information for target dates was assumed to be known in advance.
- Only one model configuration and one fixed rolling-window size were evaluated.
- The results depend on the selected features, model and validation periods.
- The separate Kaggle test dataset was not used in this validation comparison.

## Conclusion

This experiment helped me understand that evaluating a forecasting model requires more than choosing a time-based train-test split.

The training cutoff, forecast horizon, availability of historical information and validation window all matter.