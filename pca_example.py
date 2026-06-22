import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- 1. Load data ---
df = pd.read_csv("/Users/geofgle/repos/pca_project/runs.csv")
print(f"Loaded {len(df)} runs from {df['date'].iloc[0]} to {df['date'].iloc[-1]}\n")

# --- 2. Feature engineering ---
df["date"] = pd.to_datetime(df["date"])
df["days_since_start"] = (df["date"] - df["date"].min()).dt.days
df["month"] = df["date"].dt.month

# One-hot encode weather (drop one to avoid collinearity)
weather_dummies = pd.get_dummies(df["weather"], prefix="weather", drop_first=True)

# Encode route as elevation is already captured — use label encoding for route identity
route_map = {r: i for i, r in enumerate(df["route"].unique())}
df["route_id"] = df["route"].map(route_map)

# Features used for PCA (exclude target pace and derived pace_min_km)
feature_cols = [
    "distance_km",
    "elevation_m",
    "weight_kg",
    "heart_rate_avg",
    "temperature_c",
    "humidity_pct",
    "sleep_hrs",
    "fatigue_1_10",
    "days_since_start",
    "route_id",
]
X = pd.concat([df[feature_cols], weather_dummies], axis=1)
y = df["pace_min_km"].values

print(f"Features ({len(X.columns)}): {list(X.columns)}\n")

# --- 3. Standardize ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 4. Find ideal number of components ---
pca_full = PCA()
pca_full.fit(X_scaled)
cumulative = np.cumsum(pca_full.explained_variance_ratio_)
n_for_90 = int(np.argmax(cumulative >= 0.90)) + 1
print(f"Components to explain 90% variance: {n_for_90} (out of {X.shape[1]} features)")

# --- 5. Fit PCA ---
pca = PCA(n_components=n_for_90)
X_pca = pca.fit_transform(X_scaled)

print("\nVariance explained per component:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var:.1%}  (cumulative: {cumulative[i]:.1%})")

print("\nTop feature contributions per component:")
feature_names = list(X.columns)
for i, component in enumerate(pca.components_):
    top3 = sorted(zip(feature_names, component), key=lambda x: abs(x[1]), reverse=True)[:3]
    print(f"  PC{i+1}: " + "  |  ".join(f"{n} {v:+.2f}" for n, v in top3))

# --- 6. Correlation of each PC with run time ---
print("\nCorrelation of PCs with pace_min_km:")
for i in range(X_pca.shape[1]):
    corr = np.corrcoef(X_pca[:, i], y)[0, 1]
    print(f"  PC{i+1}: r = {corr:+.3f}")

# --- 7. Plots ---
fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(2, 3, figure=fig)

# Scree plot
ax0 = fig.add_subplot(gs[0, 0])
ax0.bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
        pca_full.explained_variance_ratio_, color="steelblue", alpha=0.8)
ax0.plot(range(1, len(cumulative) + 1), cumulative, "o-", color="tomato", label="Cumulative")
ax0.axhline(0.90, color="gray", linestyle="--", linewidth=0.8, label="90% threshold")
ax0.set_xlabel("Principal Component")
ax0.set_ylabel("Variance Explained")
ax0.set_title("Scree Plot")
ax0.legend(fontsize=8)

# PC1 vs PC2 coloured by run time
ax1 = fig.add_subplot(gs[0, 1])
sc = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="RdYlGn_r", edgecolors="black", linewidths=0.4, s=60)
plt.colorbar(sc, ax=ax1, label="pace_min_km")
ax1.set_xlabel("PC1")
ax1.set_ylabel("PC2")
ax1.set_title("PC1 vs PC2 (colour = run time)")

# PC1 vs PC2 coloured by season
ax2 = fig.add_subplot(gs[0, 2])
season = df["month"].apply(lambda m: 0 if m in [12,1,2] else 1 if m in [3,4,5] else 2 if m in [6,7,8] else 3)
season_labels = {0: "Winter", 1: "Spring", 2: "Summer", 3: "Autumn"}
colors = ["#5b9bd5", "#70ad47", "#ed7d31", "#a5a5a5"]
for s, (label, color) in enumerate(zip(season_labels.values(), colors)):
    mask = season == s
    ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], label=label, color=color,
                edgecolors="black", linewidths=0.3, s=55, alpha=0.85)
ax2.set_xlabel("PC1")
ax2.set_ylabel("PC2")
ax2.set_title("PC1 vs PC2 (colour = season)")
ax2.legend(fontsize=8)

# Loadings heatmap
ax3 = fig.add_subplot(gs[1, :2])
im = ax3.imshow(pca.components_[:5], cmap="coolwarm", aspect="auto", vmin=-0.6, vmax=0.6)
ax3.set_xticks(range(len(feature_names)))
ax3.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
ax3.set_yticks(range(min(5, pca.n_components_)))
ax3.set_yticklabels([f"PC{i+1}" for i in range(min(5, pca.n_components_))], fontsize=9)
ax3.set_title("Feature Loadings Heatmap (first 5 PCs)")
plt.colorbar(im, ax=ax3)

# PC1 over time (fitness trend)
ax4 = fig.add_subplot(gs[1, 2])
ax4.scatter(df["days_since_start"], X_pca[:, 0], c=y, cmap="RdYlGn_r",
            edgecolors="black", linewidths=0.3, s=50)
z = np.polyfit(df["days_since_start"], X_pca[:, 0], 1)
trendline = np.poly1d(z)
x_line = np.linspace(df["days_since_start"].min(), df["days_since_start"].max(), 100)
ax4.plot(x_line, trendline(x_line), "k--", linewidth=1.2, label="trend")
ax4.set_xlabel("Days since first run")
ax4.set_ylabel("PC1 score")
ax4.set_title("PC1 over time (fitness evolution)")
ax4.legend(fontsize=8)

plt.suptitle("PCA of Running Data (100 runs over 2 years)", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("/Users/geofgle/repos/pca_project/pca_output.png", dpi=150, bbox_inches="tight")
print("\nPlot saved to pca_project/pca_output.png")
