#!/usr/bin/env python3
"""
Reduccion de instancias sobre datasets CSV con los metodos:
- cnn   (Condensed Nearest Neighbour)
- enn   (Edited Nearest Neighbours)
- renn  (Repeated Edited Nearest Neighbours)
- tomek (Tomek Links)
- drop3 (implementacion propia)
- psc   (Prototype Selection by Clustering, implementacion propia)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans, vq
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors


@dataclass
class ReductionResult:
    X: np.ndarray
    y: np.ndarray


def load_csv_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().replace(" ", "") for c in df.columns]
    df = df.dropna().reset_index(drop=True)

    if "target" not in df.columns:
        raise RuntimeError(f"No se encontro la columna target en {csv_path}.")

    feature_names = [c for c in df.columns if c != "target"]
    X = df[feature_names].astype(float).to_numpy()
    y = pd.to_numeric(df["target"], errors="raise").astype(int).to_numpy()
    dataset_name = csv_path.stem
    return X, y, feature_names, dataset_name


def iter_input_datasets(input_dir: Path) -> list[Path]:
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"No se encontraron CSV en {input_dir}.")
    return csv_files


def _as_numpy(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(X, dtype=float), np.asarray(y)


def reduce_cnn(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> ReductionResult:
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes = np.unique(y)

    selected = []
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        selected.append(int(rng.choice(cls_idx)))

    selected_idx = np.array(sorted(set(selected)), dtype=int)
    remaining_idx = np.array([i for i in range(len(y)) if i not in selected_idx], dtype=int)

    changed = True
    while changed and remaining_idx.size > 0:
        changed = False
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(X[selected_idx], y[selected_idx])

        new_selected = []
        for idx in remaining_idx:
            pred = clf.predict(X[idx : idx + 1])[0]
            if pred != y[idx]:
                new_selected.append(int(idx))
                changed = True

        if new_selected:
            selected_idx = np.array(sorted(set(selected_idx.tolist() + new_selected)), dtype=int)
            remaining_idx = np.array(
                [i for i in range(len(y)) if i not in selected_idx],
                dtype=int,
            )

    return ReductionResult(X[selected_idx], y[selected_idx])


def reduce_enn(X: np.ndarray, y: np.ndarray, k: int = 3) -> ReductionResult:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if len(y) <= 1:
        return ReductionResult(X.copy(), y.copy())

    nn = NearestNeighbors(n_neighbors=min(k + 1, len(y)))
    nn.fit(X)
    indices = nn.kneighbors(X, return_distance=False)
    neigh = indices[:, 1:]

    keep_mask = np.ones(len(y), dtype=bool)
    for i, neighbors in enumerate(neigh):
        if neighbors.size == 0:
            continue
        neighbor_labels = y[neighbors]
        values, counts = np.unique(neighbor_labels, return_counts=True)
        majority = values[np.argmax(counts)]
        if majority != y[i]:
            keep_mask[i] = False

    return ReductionResult(X[keep_mask], y[keep_mask])


def reduce_renn(X: np.ndarray, y: np.ndarray, k: int = 3) -> ReductionResult:
    X_curr = np.asarray(X, dtype=float)
    y_curr = np.asarray(y)

    while True:
        result = reduce_enn(X_curr, y_curr, k=k)
        if len(result.y) == len(y_curr):
            return result
        X_curr, y_curr = result.X, result.y


def reduce_tomek(X: np.ndarray, y: np.ndarray) -> ReductionResult:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if len(y) <= 1:
        return ReductionResult(X.copy(), y.copy())

    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(X)
    nearest = nn.kneighbors(X, return_distance=False)[:, 1]

    remove_idx: set[int] = set()
    class_counts = {cls: int(np.sum(y == cls)) for cls in np.unique(y)}

    for i, j in enumerate(nearest):
        if y[i] == y[j]:
            continue
        if nearest[j] != i:
            continue

        if class_counts[y[i]] > class_counts[y[j]]:
            remove_idx.add(i)
        elif class_counts[y[j]] > class_counts[y[i]]:
            remove_idx.add(int(j))
        else:
            remove_idx.add(i)
            remove_idx.add(int(j))

    keep_mask = np.array([i not in remove_idx for i in range(len(y))], dtype=bool)
    return ReductionResult(X[keep_mask], y[keep_mask])


def _nearest_enemy_distance(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(y)
    dists = np.full(n, np.inf, dtype=float)
    for i in range(n):
        mask = y != y[i]
        if np.any(mask):
            d = np.linalg.norm(X[mask] - X[i], axis=1)
            dists[i] = d.min()
    return dists


def _assoc_indices(X_active: np.ndarray, k: int) -> tuple[np.ndarray, list[set[int]]]:
    n = len(X_active)
    if n <= 1:
        return np.empty((n, 0), dtype=int), [set() for _ in range(n)]

    k_eff = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1)
    nn.fit(X_active)
    indices = nn.kneighbors(X_active, return_distance=False)
    neigh = indices[:, 1:]

    associates = [set() for _ in range(n)]
    for p in range(n):
        for nb in neigh[p]:
            associates[nb].add(p)
    return neigh, associates


def _assoc_accuracy(
    point_idx: int,
    active_idx: np.ndarray,
    associates: set[int],
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    remove_point: bool,
) -> int:
    if not associates:
        return 0

    train = active_idx
    if remove_point:
        train = train[train != point_idx]

    if train.size == 0:
        return 0

    k_eff = min(k, train.size)
    clf = KNeighborsClassifier(n_neighbors=k_eff)
    clf.fit(X[train], y[train])

    assoc_arr = np.fromiter(associates, dtype=int)
    pred = clf.predict(X[assoc_arr])
    return int(np.sum(pred == y[assoc_arr]))


def reduce_drop3(X: np.ndarray, y: np.ndarray, k: int = 3) -> ReductionResult:
    # Paso 1 (como en DROP3): limpieza inicial con ENN.
    enn_result = reduce_enn(X, y, k=k)
    X_enn, y_enn = enn_result.X, enn_result.y

    # Mapear filas ENN al indice original para operar con un conjunto activo.
    original_rows = [tuple(row) for row in X]
    map_idx: dict[tuple[float, ...], list[int]] = {}
    for i, row in enumerate(original_rows):
        map_idx.setdefault(row, []).append(i)

    active_list: list[int] = []
    for row in X_enn:
        key = tuple(row)
        active_list.append(map_idx[key].pop())

    active_idx = np.array(active_list, dtype=int)
    if active_idx.size <= max(2, k):
        return ReductionResult(X[active_idx], y[active_idx])

    # Ordenar por distancia al enemigo mas cercano (descendente).
    enemy_dist = _nearest_enemy_distance(X[active_idx], y[active_idx])
    order_local = np.argsort(-enemy_dist)
    ordered_candidates = active_idx[order_local]

    for cand in ordered_candidates:
        if cand not in active_idx:
            continue

        local_pos = np.where(active_idx == cand)[0]
        if local_pos.size == 0:
            continue
        local_pos = int(local_pos[0])

        _, associates = _assoc_indices(X[active_idx], k=k)
        cand_associates_local = associates[local_pos]
        cand_associates_global = {int(active_idx[a]) for a in cand_associates_local}

        with_point = _assoc_accuracy(
            point_idx=cand,
            active_idx=active_idx,
            associates=cand_associates_global,
            X=X,
            y=y,
            k=k,
            remove_point=False,
        )
        without_point = _assoc_accuracy(
            point_idx=cand,
            active_idx=active_idx,
            associates=cand_associates_global,
            X=X,
            y=y,
            k=k,
            remove_point=True,
        )

        if without_point >= with_point and active_idx.size > k:
            active_idx = active_idx[active_idx != cand]

    return ReductionResult(X[active_idx], y[active_idx])


def reduce_psc(
    X: np.ndarray,
    y: np.ndarray,
    C: int | None = None,
    distance_metric: str = "euclidean",
) -> ReductionResult:
    """
    Prototype Selection by Clustering (PSC).
    Basado en Olvera Lopez et al.
    """
    if distance_metric != "euclidean":
        raise ValueError("PSC solo soporta distancia euclidiana.")

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    unique_classes = np.unique(y)
    r = len(unique_classes)

    if C is None:
        C = 6 * r
    C = max(1, min(int(C), len(y)))

    centroids, _ = kmeans(X, C)
    cluster_labels = vq(X, centroids)[0]

    selected_indices: set[int] = set()

    for j in range(C):
        mask = cluster_labels == j
        if not np.any(mask):
            continue

        X_j = X[mask]
        y_j = y[mask]
        indices_j = np.where(mask)[0]

        unique_classes_j, counts_j = np.unique(y_j, return_counts=True)

        if len(unique_classes_j) == 1:
            # Cluster homogeneo: conservar el punto mas cercano a su centro.
            m = np.mean(X_j, axis=0)
            dists = cdist(X_j, [m], metric="sqeuclidean").ravel()
            closest_idx = int(np.argmin(dists))
            selected_indices.add(int(indices_j[closest_idx]))
            continue

        # Cluster mixto: aplicar emparejamiento mayoria-minoria del metodo PSC.
        maj_class = unique_classes_j[int(np.argmax(counts_j))]
        maj_mask = y_j == maj_class
        X_maj = X_j[maj_mask]
        indices_maj = indices_j[maj_mask]

        for k_class in unique_classes_j:
            if k_class == maj_class:
                continue
            min_mask = y_j == k_class
            X_min = X_j[min_mask]
            indices_min = indices_j[min_mask]

            for i_min in range(len(X_min)):
                p_j = X_min[i_min : i_min + 1]
                dists_to_maj = cdist(p_j, X_maj, metric="sqeuclidean").ravel()
                c_idx = int(np.argmin(dists_to_maj))
                p_c_idx = int(indices_maj[c_idx])
                selected_indices.add(p_c_idx)

                p_c = X[p_c_idx : p_c_idx + 1]
                dists_to_min = cdist(p_c, X_min, metric="sqeuclidean").ravel()
                m_k_idx = int(np.argmin(dists_to_min))
                p_m_k_idx = int(indices_min[m_k_idx])
                selected_indices.add(p_m_k_idx)

    selected_idx = np.array(sorted(selected_indices), dtype=int)
    return ReductionResult(X[selected_idx], y[selected_idx])


def run_method(
    name: str, X: np.ndarray, y: np.ndarray, psc_c: int | None = None
) -> ReductionResult:
    m = name.lower().strip()
    if m == "cnn":
        return reduce_cnn(X, y)
    if m == "enn":
        return reduce_enn(X, y)
    if m == "renn":
        return reduce_renn(X, y)
    if m == "tomek":
        return reduce_tomek(X, y)
    if m == "drop3":
        return reduce_drop3(X, y)
    if m == "psc":
        return reduce_psc(X, y, C=psc_c)
    raise ValueError(f"Metodo no soportado: {name}")


def save_reduction_csv(
    result: ReductionResult,
    method: str,
    feature_names: list[str],
    outdir: Path,
    dataset_name: str,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(result.X, columns=feature_names)
    df["target"] = result.y
    output_path = outdir / f"{dataset_name}_{method.lower()}_reduced.csv"
    df.to_csv(output_path, index=False)
    return output_path


def save_final_report(report_rows: list[dict[str, object]], outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    report_df = pd.DataFrame(report_rows)
    report_path = outdir / "reduction_report.csv"
    report_df.to_csv(report_path, index=False)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reduccion de instancias para datasets CSV."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["cnn", "enn", "renn", "tomek", "drop3", "psc"],
        help="Lista de metodos a ejecutar.",
    )
    parser.add_argument(
        "--psc-c",
        type=int,
        default=None,
        help="Numero de clusters para PSC. Por defecto usa C=6*r.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="generated_datasets",
        help="Carpeta con los CSV de entrada.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="reduced_csv",
        help="Carpeta donde se guardan los CSV reducidos.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    outdir = Path(args.outdir)
    report_rows: list[dict[str, object]] = []

    for dataset_path in iter_input_datasets(input_dir):
        X, y, feature_names, dataset_name = load_csv_dataset(dataset_path)
        X, y = _as_numpy(X, y)

        print(f"\nDataset: {dataset_name}")
        print(f"Archivo origen: {dataset_path}")
        print(f"Instancias originales: {len(y)}")

        for method in args.methods:
            result = run_method(method, X, y, psc_c=args.psc_c)
            n_new = len(result.y)
            reduction = 100.0 * (1 - n_new / len(y))
            csv_path = save_reduction_csv(
                result, method, feature_names, outdir, dataset_name=dataset_name
            )
            report_rows.append(
                {
                    "dataset": dataset_name,
                    "method": method.lower(),
                    "original_instances": len(y),
                    "reduced_instances": n_new,
                    "reduction_percent": round(reduction, 2),
                    "output_csv": str(csv_path),
                }
            )
            print(f"\nMetodo: {method.lower()}")
            print(f"Instancias reducidas: {n_new}")
            print(f"Reduccion: {reduction:.2f}%")
            print(f"CSV guardado en: {csv_path}")

    report_path = save_final_report(report_rows, outdir)
    report_df = pd.DataFrame(report_rows)
    summary_df = (
        report_df.groupby("method", as_index=False)["reduction_percent"]
        .mean()
        .sort_values("reduction_percent", ascending=False)
    )

    print("\nReporte final")
    print(f"CSV resumen guardado en: {report_path}")
    print("Promedio de reduccion por metodo:")
    for _, row in summary_df.iterrows():
        print(f"- {row['method']}: {row['reduction_percent']:.2f}%")


if __name__ == "__main__":
    main()
