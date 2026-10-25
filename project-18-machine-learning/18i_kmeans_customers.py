
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Customer data: [visits per month, spending per visit]
customers = [
    [25, 4], [22, 5], [28, 3], [24, 6],
    [3, 30], [5, 25], [2, 35], [4, 28],
    [12, 12], [14, 15], [10, 14], [13, 11]
]

# 2. Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(customers)

# 3. Create K-Means with 3 clusters
model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# 4. Train model and assign clusters
clusters = model.fit_predict(X)

# 5. Print results
for customer, cluster in zip(customers, clusters):
    print(f"Visits: {customer[0]}, Spending: €{customer[1]} -> Cluster {cluster}")

# 6. Visualize customer clusters
plt.figure(figsize=(8, 5))

plt.scatter(
    [customer[0] for customer in customers],
    [customer[1] for customer in customers],
    c=clusters,
    cmap="viridis",
    s=100
)

plt.xlabel("Visits per Month")
plt.ylabel("Spending per Visit (€)")
plt.title("Cafe Customer Segmentation - K-Means")
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.show()

# 7. Elbow Method
inertias = []

for k in range(1, 9):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X)
    inertias.append(model.inertia_)

# 8. Visualize Elbow Method
plt.figure(figsize=(8, 5))

plt.plot(range(1, 9), inertias, marker="o")

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method - Finding the Best K")
plt.xticks(range(1, 9))
plt.grid(True)
plt.tight_layout()
plt.show()
