import numpy as np
import pandas as pd
from datetime import date, timedelta

np.random.seed(7)

# --- Config ---
start_date = date(2023, 1, 1)
n_runs = 100

# Routes: (name, typical_distance_km, elevation_gain_m)
routes = [
    ("Flat Loop",    5.0,  20),
    ("River Path",   8.0,  35),
    ("Hill Circuit", 10.0, 180),
    ("Long Trail",   15.0, 120),
    ("Park Run",     5.0,  10),
]

weather_options = ["sunny", "cloudy", "rainy", "windy", "cold"]

records = []
current_date = start_date
weight = 83.0  # starting weight kg

for i in range(n_runs):
    # Space runs ~7-9 days apart with some variance (more frequent in summer)
    day_of_year = current_date.timetuple().tm_yday
    summer = 1 if 120 < day_of_year < 270 else 0
    gap = int(np.random.normal(7 - summer * 1.5, 1.5))
    gap = max(3, gap)
    current_date += timedelta(days=gap)

    # Season-based temperature
    month = current_date.month
    base_temp = 10 + 12 * np.sin((month - 3) / 12 * 2 * np.pi)
    temperature = round(base_temp + np.random.normal(0, 4), 1)
    humidity = int(np.clip(np.random.normal(65, 15), 30, 95))

    # Weather probabilities shift by season
    if month in [12, 1, 2]:
        w_probs = [0.15, 0.35, 0.20, 0.10, 0.20]
    elif month in [6, 7, 8]:
        w_probs = [0.50, 0.30, 0.10, 0.05, 0.05]
    else:
        w_probs = [0.30, 0.35, 0.20, 0.10, 0.05]
    weather = np.random.choice(weather_options, p=w_probs)

    # Weight drifts slowly (down overall, with noise)
    weight += np.random.normal(-0.03, 0.3)
    weight = round(np.clip(weight, 74, 87), 1)

    # Pick a route (longer runs less frequent)
    route_probs = [0.25, 0.30, 0.20, 0.10, 0.15]
    route_name, base_dist, elevation = routes[np.random.choice(len(routes), p=route_probs)]
    distance = round(base_dist + np.random.normal(0, 0.5), 2)

    # Fitness improves over time (pace drops ~8% over 2 years)
    fitness_factor = 1 - (i / n_runs) * 0.08

    # Base pace (min/km) — affected by distance, elevation, weather, weight, fitness
    base_pace = 5.8 * fitness_factor
    base_pace += (weight - 79) * 0.04         # heavier = slower
    base_pace += (elevation / 100) * 0.25     # hills slow you down
    base_pace += (distance - 8) * 0.03        # longer = slower pace
    if weather == "rainy":   base_pace += 0.15
    if weather == "windy":   base_pace += 0.10
    if weather == "cold":    base_pace += 0.08
    if temperature > 28:     base_pace += 0.12  # heat penalty
    if temperature < 2:      base_pace += 0.10  # cold penalty
    pace = round(base_pace + np.random.normal(0, 0.12), 2)

    time_min = round(pace * distance, 1)

    # Heart rate: higher on hills, heat, and fast pace
    hr_base = 148 + (5.8 - pace) * 8 + (elevation / 100) * 5
    heart_rate = int(np.clip(hr_base + np.random.normal(0, 5), 120, 185))

    # Sleep the night before
    sleep = round(np.clip(np.random.normal(7.2, 0.8), 4.5, 9.5), 1)

    # Subjective fatigue (1-10): worse after long runs, poor sleep
    fatigue = int(np.clip(np.random.normal(4, 1.5) + (distance > 12) * 1.5 + (sleep < 6) * 1.5, 1, 10))

    records.append({
        "date":           current_date.isoformat(),
        "route":          route_name,
        "distance_km":    distance,
        "time_min":       time_min,
        "pace_min_km":    pace,
        "elevation_m":    elevation,
        "weight_kg":      weight,
        "heart_rate_avg": heart_rate,
        "temperature_c":  temperature,
        "humidity_pct":   humidity,
        "weather":        weather,
        "sleep_hrs":      sleep,
        "fatigue_1_10":   fatigue,
    })

df = pd.DataFrame(records)
df.to_csv("/Users/geofgle/repos/pca_project/runs.csv", index=False)
print(f"Generated {len(df)} runs from {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
print(f"\nSample:\n{df.head(5).to_string()}")
print(f"\nStats:\n{df[['distance_km','time_min','pace_min_km','weight_kg','temperature_c']].describe().round(2).to_string()}")
