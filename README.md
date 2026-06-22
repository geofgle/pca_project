# PCA Running Analysis

A Python project that uses **Principal Component Analysis (PCA)** to explore what factors drive running pace (min/km) across 100 runs over a 2-year period.

## Goal

Understand which combinations of conditions — distance, weather, weight, fatigue, fitness progression — most explain variation in running pace, as a first step toward building a pace predictor.

## Files

| File | Description |
|---|---|
| `generate_runs.py` | Generates the synthetic `runs.csv` dataset with realistic patterns |
| `runs.csv` | 100 runs from 2023-01-10 to 2024-08-06 |
| `pca_example.py` | Main analysis script — loads data, runs PCA, outputs plots |
| `pca_example_2d.py` | Original 2D toy example used to learn PCA basics |
| `pca_output.png` | Latest plot output |

## Dataset Features

| Feature | Description |
|---|---|
| `distance_km` | Distance of the run |
| `elevation_m` | Elevation gain of the route |
| `weight_kg` | Runner's weight on the day |
| `heart_rate_avg` | Average heart rate during the run |
| `temperature_c` | Air temperature |
| `humidity_pct` | Humidity percentage |
| `weather` | Conditions (sunny / cloudy / rainy / windy / cold) |
| `sleep_hrs` | Hours of sleep the night before |
| `fatigue_1_10` | Subjective fatigue level (1 = fresh, 10 = exhausted) |
| `route` | Named route (Flat Loop, River Path, Hill Circuit, Long Trail, Park Run) |

**Target variable:** `pace_min_km` — minutes per kilometre (lower = faster)

## How to Run

```bash
# Install dependencies
pip install numpy pandas matplotlib scikit-learn

# (Optional) Regenerate the dataset
python generate_runs.py

# Run the PCA analysis
python pca_example.py
```

## Key Findings

- **10 components** needed to explain 90% of variance — pace is genuinely multi-factor
- **PC2** (distance + elevation + fitness over time) is the strongest predictor of pace (`r = 0.82`)
- Weather, fatigue, and sleep each capture their own independent slice of variance

## What's Next

- Build a regression model on top of the PCA components to predict pace for a future run
- Load real running data (e.g. from Garmin / Strava export) to replace the synthetic dataset
