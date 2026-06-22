import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 6 features, 20 samples (multi-dimensional example)
np.random.seed(42)
n_samples = 20

# Simulate correlated features so PCA has something interesting to find
base = np.random.randn(n_samples)
X = np.column_stack([
    base + np.random.randn(n_samples) * 0.3,        # feature_0 — strongly correlated with base
    base + np.random.randn(n_samples) * 0.3,        # feature_1 — strongly correlated with base
    base * 0.5 + np.random.randn(n_samples) * 0.5,  # feature_2 — moderately correlated
    np.random.randn(n_samples),                      # feature_3 — independent noise
    np.random.randn(n_samples) * 2,                  # feature_4 — independent noise (different scale)
    base * -1 + np.random.randn(n_samples) * 0.4,   # feature_5 — inversely correlated with base
])

print(f"Data shape: {X.shape}  ({X.shape[0]} samples, {X.shape[1]} features)\n")

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Find ideal number of components ---
pca_full = PCA()
pca_full.fit(X_scaled)

cumulative = np.cumsum(pca_full.explained_variance_ratio_)
n_for_95 = int(np.argmax(cumulative >= 0.95)) + 1
print(f"Components needed to explain 95% variance: {n_for_95}")

# --- Apply PCA keeping top components ---
pca = PCA(n_components=n_for_95)
X_pca = pca.fit_transform(X_scaled)

print(f"\nExplained variance per component:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var:.2%}  (cumulative: {cumulative[i]:.2%})")

print(f"\nComponent loadings (which features drive each PC):")
feature_names = [f"feature_{i}" for i in range(X.shape[1])]
for i, component in enumerate(pca.components_):
    top = sorted(zip(feature_names, component), key=lambda x: abs(x[1]), reverse=True)
    print(f"  PC{i+1}: " + ", ".join(f"{n}={v:+.3f}" for n, v in top))

# --- Plots ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Scree plot
bars = axes[0].bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
                   pca_full.explained_variance_ratio_, color="steelblue")
axes[0].plot(range(1, len(cumulative) + 1), cumulative, "o-", color="tomato", label="Cumulative")
axes[0].axhline(0.95, color="gray", linestyle="--", linewidth=0.8, label="95% threshold")
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Variance Explained")
axes[0].set_title("Scree Plot")
axes[0].legend()

# 2D projection (PC1 vs PC2)
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], color="steelblue", edgecolors="black")
for i, (x, y) in enumerate(X_pca[:, :2]):
    axes[1].annotate(f"S{i+1}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
axes[1].set_title("Projection: PC1 vs PC2")

# Loading heatmap
im = axes[2].imshow(pca.components_, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
axes[2].set_xticks(range(len(feature_names)))
axes[2].set_xticklabels(feature_names, rotation=45, ha="right")
axes[2].set_yticks(range(pca.n_components_))
axes[2].set_yticklabels([f"PC{i+1}" for i in range(pca.n_components_)])
axes[2].set_title("Loadings Heatmap")
plt.colorbar(im, ax=axes[2])

plt.tight_layout()
plt.savefig("/Users/geofgle/repos/pca_project/pca_output.png", dpi=150)
print("\nPlot saved to pca_project/pca_output.png")
