# Synthetic Load Working Guide

## Purpose

This module is Brain 1a of IEMS. It creates a realistic hourly demand profile for a residential consumer, especially when only limited user inputs (such as monthly kWh) are available.

The module learns patterns from historical smart meter data and generates a 30-day hourly synthetic series for downstream optimization and control modules.

## End-to-End Pipeline

Input dataset files in ../Dataset
-> Preprocessing and hourly aggregation
-> Daily profile extraction (24 points/day)
-> KMeans clustering of day types
-> Markov transition learning between day types
-> Validation on holdout split
-> 30-day synthetic generation (with optional target-kWh scaling)
-> Output CSV for downstream modules

## Stage 1: Data Preprocessing

Script: data_preprocessing.py

What it does:
- Scans all CSV files matching ../Dataset/*.csv.
- Auto-detects schema format:
  - Interval schema: x_Timestamp, t_kWh, meter
  - Daily schema: meter, Date, t_kWh
- Converts data into one hourly t_kWh time series.

Two internal modes:
1. Interval input mode
- Used for 3-minute or similar data.
- Reads files in chunks for memory-safe processing.
- Aggregates to hourly by resampling and summing interval energy.

2. Daily input mode
- Used when only daily totals are available.
- Expands daily totals to hourly values using a fixed hourly distribution profile.

Meter combination behavior:
- mean mode: representative household-style profile.
- sum mode: feeder-level aggregate demand.

Output:
- outputs/cleaned_hourly.csv with columns:
  - x_Timestamp
  - t_kWh

## Stage 2: Daily Profile Learning

Script: train_kmeans.py

What it does:
- Reads outputs/cleaned_hourly.csv.
- Reshapes hourly data into daily vectors of length 24.
- Trains KMeans on day vectors.

Why:
- Each cluster represents a typical day type (for example, low-load day, high evening-peak day, etc.).

Current tuned setup:
- k = 9
- seed = 42

Outputs:
- synthetic_load/kmeans_model.pkl
- outputs/daily_profiles.npy
- synthetic_load/metadata.json

## Stage 3: Day-to-Day Dynamics (Markov)

Script: markov_model.py

What it does:
- Predicts cluster label for each historical day.
- Learns transition probabilities from one day type to the next day type.
- Applies Laplace smoothing for numeric stability.

Why:
- Captures realistic sequencing, not just isolated day shapes.

Output:
- synthetic_load/markov_transition.npy

## Stage 4: Validation

Script: validate_model.py

What it does:
- Uses time-based train/holdout split.
- Generates synthetic holdout series without target leakage.
- Compares against a flat baseline.
- Reports MAPE and RMSE.

Important design choice:
- Validation is leakage-free. No post-hoc scaling using holdout targets.

Why baseline is included:
- A synthetic model must beat or match simple baselines to be considered useful.

## Stage 5: Final Synthetic Generation

Script: generate_synthetic.py

What it does:
- Uses trained KMeans and Markov artifacts.
- Generates hourly values for requested number of days (default 30).
- Supports stability controls:
  - noise-scale
  - blend-alpha
- Supports user monthly energy target with target-kwh scaling.

Default downstream contract:
- 30 days x 24 hours = 720 rows

Output file:
- outputs/synthetic_30d_hourly.csv

Columns:
- x_Timestamp
- t_kWh
- day
- hour

## How User Monthly kWh Fits In

Training is done once from historical data. After that:
- User provides monthly energy consumption in kWh.
- Generator scales synthetic monthly total to that target.

Example:
- target-kwh 350 creates a 30-day profile summing close to 350 kWh.

This makes the module suitable for API-driven usage in later IEMS steps.

## Recommended Operational Flows

### First-time setup or after dataset change

1. python3 synthetic_load/data_preprocessing.py --combine-mode mean --chunksize 300000
2. python3 synthetic_load/train_kmeans.py --k 9 --seed 42
3. python3 synthetic_load/markov_model.py --laplace 1.0
4. python3 synthetic_load/validate_model.py --train-ratio 0.8 --seed 42 --no-plot
5. python3 synthetic_load/generate_synthetic.py --days 30 --seed 42 --noise-scale 0.03 --blend-alpha 0.05 --output outputs/synthetic_30d_hourly.csv

### Regular use after training artifacts already exist

1. python3 synthetic_load/generate_synthetic.py --days 30 --seed 42 --noise-scale 0.03 --blend-alpha 0.05 --target-kwh 350 --output outputs/synthetic_30d_hourly.csv

Optional quality check:
- python3 synthetic_load/validate_model.py --train-ratio 0.8 --seed 42 --no-plot

## Practical Notes

- If input files are very large, keep chunksize moderate (for example 300000).
- For reproducibility, keep seed fixed.
- For household profile generation, use combine-mode mean.
- For feeder-level studies, use combine-mode sum.
- If metrics degrade after new data ingestion, rerun full build flow.

## Output Readiness for Next Modules

The module output is ready for optimization and simulation components because:
- It is hourly.
- It is exactly 30 days by default.
- It preserves learned historical usage structure.
- It can be scaled to user monthly kWh.
