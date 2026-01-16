import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# ===========================
# 1️⃣ Cargar los datos
# ===========================
csv_path = "../data/bootstrap_representations.csv"
df = pd.read_csv(csv_path)

# ===========================
# 2️⃣ Preparar los datos para t-SNE
# ===========================
# task_id será nuestra etiqueta de dataset
dataset_ids = df['task_id'].values

# Eliminamos task_id y cualquier columna no numérica
X = df.drop(columns=['task_id']).astype(float).fillna(0).values  # rellenar NaN con 0

# ===========================
# 3️⃣ Ejecutar t-SNE
# ===========================
tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    init='random',
    learning_rate='auto'
)

X_2d = tsne.fit_transform(X)

# ===========================
# 4️⃣ Graficar
# ===========================
plt.figure(figsize=(12, 9))

for tid in np.unique(dataset_ids):
    mask = dataset_ids == tid
    plt.scatter(
        X_2d[mask, 0],
        X_2d[mask, 1],
        label=f'Dataset {tid}',
        alpha=0.7,
        s=50
    )

plt.legend()
plt.title("t-SNE 2D visualization of bootstrapped datasets")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.grid(True)
# plt.show()
plt.savefig("cluster_plot.png", dpi=300, bbox_inches='tight')
print("Figura guardada como cluster_plot.png")
