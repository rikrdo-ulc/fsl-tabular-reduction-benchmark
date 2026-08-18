import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path


# =============================
# Generadores estilo TF Playground
# =============================
def tf_circle(n=400, noise=0.0, seed=0):
    np.random.seed(seed)
    r = np.random.uniform(0, 1, n)
    theta = np.random.uniform(0, 2*np.pi, n)
    X = np.c_[r * np.cos(theta), r * np.sin(theta)]
    y = (r > 0.5).astype(int)
    X += noise * np.random.randn(*X.shape)
    return X, y


def tf_xor(n=400, noise=0.0, seed=1):
    np.random.seed(seed)
    X = np.random.uniform(-1, 1, (n, 2))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    X += noise * np.random.randn(*X.shape)
    return X, y


def tf_moons(n=200, noise=0.05, seed=2):
    np.random.seed(seed)
    theta = np.random.uniform(0, np.pi, n)
    x1 = np.c_[np.cos(theta), np.sin(theta)]
    x2 = np.c_[1 - np.cos(theta), 1 - np.sin(theta) - 0.5]
    X = np.vstack([x1, x2])
    y = np.hstack([np.zeros(n), np.ones(n)])
    X += noise * np.random.randn(*X.shape)
    return X, y


def tf_radial(n=400, noise=0.0, seed=3):
    np.random.seed(seed)
    X = np.random.uniform(-1, 1, (n, 2))
    r = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
    y = (np.sin(4 * r) > 0).astype(int)
    X += noise * np.random.randn(*X.shape)
    return X, y


def tf_complex(n=400, noise=0.05, seed=4):
    np.random.seed(seed)
    X = np.random.uniform(-1, 1, (n, 2))
    y = ((X[:, 0]**2 + X[:, 1]**2 + 0.3*np.sin(3*X[:, 0])) > 0.7).astype(int)
    X += noise * np.random.randn(*X.shape)
    return X, y


def sine_wave_dataset(
    n_samples=1000,
    x_range=(0, 4),
    y_range=(-2, 2),
    amplitude=1.0,
    frequency=2 * np.pi / 2,
    phase=0.0,
    noise=0.0,
    seed=42
):
    """
    Genera un dataset de clasificación binaria con frontera senoidal.
    """
    np.random.seed(seed)

    X = np.zeros((n_samples, 2))
    X[:, 0] = np.random.uniform(*x_range, n_samples)
    X[:, 1] = np.random.uniform(*y_range, n_samples)

    y_boundary = amplitude * np.sin(frequency * X[:, 0] + phase)
    y = (X[:, 1] > y_boundary).astype(int)
    X += noise * np.random.randn(*X.shape)

    return X, y, y_boundary


# =============================
# Datasets guardados en variables
# =============================
datasets = []

def add_dataset(name, X, y):
    X = StandardScaler().fit_transform(X)
    datasets.append({
        "name": name,
        "X": X,
        "y": y
    })


def save_datasets_to_csv(output_dir="generated_datasets"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        df = pd.DataFrame(dataset["X"], columns=["x1", "x2"])
        df["target"] = dataset["y"]
        file_name = dataset["name"].lower().replace(" ", "_").replace("(", "").replace(")", "")
        df.to_csv(output_path / f"{file_name}.csv", index=False)


add_dataset("Circle (clean)", *tf_circle(seed=0))
add_dataset("Circle (noisy)", *tf_circle(noise=0.15, seed=1))

add_dataset("XOR (clean)", *tf_xor(seed=2))
add_dataset("XOR (noisy)", *tf_xor(noise=0.15, seed=3))

add_dataset("Moons (clean)", *tf_moons(seed=4))
add_dataset("Moons (noisy)", *tf_moons(noise=0.15, seed=5))

add_dataset("Radial (clean)", *tf_radial(seed=6))
add_dataset("Radial (noisy)", *tf_radial(noise=0.15, seed=7))

add_dataset("Complex (clean)", *tf_complex(seed=8))
add_dataset("Complex (noisy)", *tf_complex(noise=0.15, seed=9))

add_dataset(
    "Sine Wave (clean)",
    *sine_wave_dataset(
        n_samples=400,
        amplitude=1.0,
        frequency=np.pi,
        noise=0.0,
        seed=10
    )[:2]
)
add_dataset(
    "Sine Wave (noisy)",
    *sine_wave_dataset(
        n_samples=600,
        amplitude=1.0,
        frequency=np.pi,
        noise=0.15,
        seed=11
    )[:2]
)

save_datasets_to_csv()


# =============================
# Ejemplo de uso
# =============================
# X = datasets[0]["X"]
# y = datasets[0]["y"]
# print(datasets[0]["name"])


# =============================
# Visualización opcional
# =============================
fig, axes = plt.subplots(2, 6, figsize=(20, 7))

for i, d in enumerate(datasets):
    ax = axes.flat[i]
    ax.scatter(d["X"][:, 0], d["X"][:, 1],
               c=d["y"], cmap="coolwarm", s=12)
    ax.set_title(d["name"])
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
