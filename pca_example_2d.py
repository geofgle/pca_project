import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- 1. Your data (rows = samples, columns = features) ---
# Replace this with your actual data
X = np.array([
    [2.5, 2.4],
    [0.5, 0.7],
    [2.2, 2.9],
    [1.9, 2.2],
    [3.1, 3.0],
    [2.3, 2.7],
    [2.0, 1.6],
    [1.0, 1.1],
    [1.5, 1.6],
    [1.1, 0.9],
])

# --- 2. Standardize: zero mean, unit variance ---
# PCA is sensitive to scale — always standardize unless features are already comparable
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. Fit PCA ---
# n_components: how many principal components to keep
# Set to None to keep all, or a float (e.g. 0.95) to keep enough to explain 95% of variance
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# --- 4. Inspect results ---
print("Explained variance ratio per component:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var:.2%}")

print(f"\nTotal variance explained: {pca.explained_variance_ratio_.sum():.2%}")

print("\nComponent loadings (how much each original feature contributes):")
feature_names = [f"feature_{i}" for i in range(X.shape[1])]
for i, component in enumerate(pca.components_):
    print(f"  PC{i+1}: {dict(zip(feature_names, component.round(3)))}")

# --- 5. Scree plot: helps decide how many components to keep ---
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained")
plt.title("Scree Plot")

# --- 6. Plot the transformed data (first 2 PCs) ---
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection")

plt.tight_layout()
plt.show()
