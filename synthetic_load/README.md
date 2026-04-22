# Synthetic Load Module

Brain 1a module for generating a synthetic 30-day hourly load profile from smart meter data.

## Prerequisites

- Run all commands from the project root folder.
- Input CSV files must be inside ../Dataset.
- Python 3 environment must include pandas, numpy, scikit-learn, matplotlib, joblib.

## First-Time Run (Full Build)

Use this flow when setting up from scratch, changing source data, or retraining models.

1. Preprocess and build hourly training series from all Dataset CSV files:

python3 synthetic_load/data_preprocessing.py --combine-mode mean --chunksize 300000

2. Train KMeans daily profile model (current tuned value is k=9):

python3 synthetic_load/train_kmeans.py --k 9 --seed 42

3. Build Markov transition matrix:

python3 synthetic_load/markov_model.py --laplace 1.0

4. Validate quality on holdout split:

python3 synthetic_load/validate_model.py --train-ratio 0.8 --seed 42 --no-plot

5. Generate final synthetic output for next modules (interactive: asks days and monthly kWh):

python3 synthetic_load/generate_synthetic.py

## Further Use (After Training)

Once training artifacts already exist (kmeans_model.pkl, markov_transition.npy, metadata.json), you do not need retraining for every run.

### A) Generate a new 30-day profile quickly

python3 synthetic_load/generate_synthetic.py

### B) Generate from user monthly energy input

python3 synthetic_load/generate_synthetic.py --days 30 --seed 42 --noise-scale 0.03 --blend-alpha 0.05 --target-kwh 350 --target-mode monthly --output outputs/synthetic_30d_hourly.csv

Replace 350 with the user monthly energy consumption (kWh).

Note: with target-mode monthly (default), total energy scales by days/30.
Example: 300 kWh monthly -> 30 days gives 300 kWh, 60 days gives 600 kWh.

If you need an exact total for the full horizon instead, use:

python3 synthetic_load/generate_synthetic.py --days 60 --target-kwh 300 --target-mode total --output outputs/synthetic_30d_hourly.csv

If --days or --target-kwh is omitted, the script asks for input interactively.

### C) Re-validate without retraining

python3 synthetic_load/validate_model.py --train-ratio 0.8 --seed 42 --no-plot

## Input Data Support

- Folder: ../Dataset
- All matching CSV files in Dataset are merged automatically.
- Supported schemas:
  - Interval schema: x_Timestamp, t_kWh, meter
  - Daily schema: meter, Date, t_kWh

## Aggregation Strategy

- combine-mode mean:
  - Best for representative household-level synthetic profile.
  - Recommended default for current pipeline.
- combine-mode sum:
  - Use for feeder/aggregate demand scenarios.

## Script Responsibilities

- data_preprocessing.py
  - Auto-detects schema type.
  - Merges all input CSVs.
  - Converts to hourly t_kWh series.
  - Saves outputs/cleaned_hourly.csv.

- train_kmeans.py
  - Converts hourly series to daily 24-point vectors.
  - Trains KMeans model and saves metadata.

- markov_model.py
  - Creates day-to-day transition probabilities across clusters.

- validate_model.py
  - Performs leakage-free holdout validation.
  - Prints Synthetic vs Baseline MAPE/RMSE.

- generate_synthetic.py
  - Produces final synthetic hourly CSV used by next modules.
  - Output shape default: 720 rows (30 x 24).

## Generated Artifacts

- synthetic_load/kmeans_model.pkl
- synthetic_load/markov_transition.npy
- synthetic_load/metadata.json
- outputs/cleaned_hourly.csv
- outputs/daily_profiles.npy
- outputs/synthetic_30d_hourly.csv

## Quick Check

To verify output shape is exactly 24 x 30:

python3 - <<'PY'
import pandas as pd
df = pd.read_csv('outputs/synthetic_30d_hourly.csv')
print('rows=', len(df), 'days=', df['day'].nunique(), 'hours/day ok=', df.groupby('day')['hour'].nunique().eq(24).all())
PY

## Team Demo Animation (Separate from Core Code)

Use this standalone script to visualize pipeline progress while stages run:

python3 synthetic_load/pipeline_animation_demo.py --days 30 --target-kwh 350 --target-mode monthly

Notes:
- This does not modify core Brain 1a logic; it only orchestrates and visualizes execution.
- On Linux without GUI display, run terminal mode:

python3 synthetic_load/pipeline_animation_demo.py --days 30 --target-kwh 350 --no-gui

- For a fast presentation using existing artifacts, skip retraining:

python3 synthetic_load/pipeline_animation_demo.py --days 30 --target-kwh 350 --skip-train
