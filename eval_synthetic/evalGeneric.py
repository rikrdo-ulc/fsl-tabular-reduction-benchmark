import argparse
import gc
import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.stats import rankdata, wilcoxon

# Evita errores de Tcl/Tk en ejecuciones multihilo/headless.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Configuracion general
SEEDS = [7, 21, 42, 123, 456]
EMB_DIM = 32
LEARNING_RATE = 1e-3
KNN_K_GRID = [1, 3, 5, 7, 9]
KNN_WEIGHTS_GRID = ["uniform", "distance"]
KNN_P_GRID = [1, 2]
LOGREG_C_GRID = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
LOGREG_PENALTY_GRID = ["l1", "l2"]
MLP_HIDDEN_GRID = [(8,), (16,), (32,), (16, 8), (32, 16)]
MLP_ALPHA_GRID = [1e-5, 1e-4, 1e-3, 1e-2]
MLP_ACTIVATION_GRID = ["relu", "tanh"]
RF_ESTIMATORS_GRID = [60, 120, 240]
RF_MAX_DEPTH_GRID = [None, 4, 8, 12]
RF_MIN_SAMPLES_LEAF_GRID = [1, 2, 4]
SVM_C_GRID = [0.1, 1.0, 3.0, 10.0, 30.0, 100.0]
SVM_GAMMA_GRID = ["scale", "auto", 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
PROTO_LR_GRID = [3e-4, 1e-3]
MATCHING_LR_GRID = [3e-4, 1e-3]
RELATION_LR_GRID = [3e-4, 1e-3]
COSINE_SCALE_GRID = [5.0, 10.0, 20.0]
COV_REG_GRID = [1e-4, 1e-3]
TPN_ALPHA_GRID = [0.8, 0.9]
TPN_SIGMA_GRID = [0.5, 1.0, 2.0]
SIAMESE_HIDDEN_GRID = [32, 64]
SIAMESE_LR_GRID = [3e-4, 1e-3]
N_JOBS = -1
PARALLEL_BACKEND_AVAILABLE = True
AUTO_N_WAY_CAP = 5
SAVE_EMBEDDINGS = False
SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class FewShotConfig:
    requested_n_way: int = 2
    requested_k_shot: int = 5
    requested_q_query: int = 10
    test_size: float = 0.3
    train_episodes: int = 60
    eval_episodes: int = 40
    tune_episodes: int = 40
    classical_tune_episodes: int = 12
    meta_tune_eval_episodes: int = 12
    meta_tune_train_episodes: int = 40
    siamese_tune_train_episodes: int = 40
    explicit_config: bool = False
    quick_mode: bool = False
    allow_replacement: bool = True
    effective_n_way: int | None = None
    effective_k_shot: int | None = None
    effective_q_query: int | None = None
    inner_val_per_class: int | None = None
    dataset_regime: str = "manual"
    class_disjoint_n_way: int = 0
    class_disjoint_test_classes: int = 0
    class_disjoint_status: str = ""
    class_disjoint_status_code: str = ""

    @property
    def sample_n_way(self):
        return self.requested_n_way if self.effective_n_way is None else self.effective_n_way

    @property
    def k_shot(self):
        return self.requested_k_shot if self.effective_k_shot is None else self.effective_k_shot

    @property
    def q_query(self):
        return self.requested_q_query if self.effective_q_query is None else self.effective_q_query

    @property
    def inner_val(self):
        return self.inner_val_per_class if self.inner_val_per_class is not None else self.k_shot + self.q_query

    def as_label_dict(self, n_way=None):
        return {
            "N_WAY": self.sample_n_way if n_way is None else int(n_way),
            "K_SHOT": self.k_shot,
            "Q_QUERY": self.q_query,
        }

    def with_override(self, n_way, k_shot, q_query):
        return replace(
            self,
            requested_n_way=int(n_way),
            requested_k_shot=int(k_shot),
            requested_q_query=int(q_query),
            explicit_config=True,
        )

    def with_quick_mode(self):
        return replace(
            self,
            quick_mode=True,
            train_episodes=min(self.train_episodes, 12),
            eval_episodes=min(self.eval_episodes, 8),
            tune_episodes=min(self.tune_episodes, 8),
            classical_tune_episodes=min(self.classical_tune_episodes, 4),
            meta_tune_eval_episodes=min(self.meta_tune_eval_episodes, 4),
            meta_tune_train_episodes=min(self.meta_tune_train_episodes, 12),
            siamese_tune_train_episodes=min(self.siamese_tune_train_episodes, 12),
        )


BASE_FEWSHOT_CONFIG = FewShotConfig()


@dataclass(frozen=True)
class EncoderSpec:
    input_dim: int
    emb_dim: int
    hidden_units: tuple[int, ...]
    l2_strength: float


def parse_int_list(text):
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if not parts:
        raise ValueError("La lista no puede estar vacia.")
    return [int(p) for p in parts]


def parse_fewshot_config(text):
    parts = [p.strip() for p in str(text).replace("x", ":").split(":") if p.strip()]
    if len(parts) != 3:
        raise make_skip_error(
            "invalid_fewshot_config",
            f"Configuracion few-shot invalida '{text}'. Usa el formato N:K:Q, por ejemplo 3:5:10."
        )
    n_way, k_shot, q_query = (int(p) for p in parts)
    if min(n_way, k_shot, q_query) < 1:
        raise make_skip_error(
            "invalid_fewshot_config",
            f"Configuracion few-shot invalida '{text}'. Todos los valores deben ser >= 1.",
        )
    return {"N_WAY": n_way, "K_SHOT": k_shot, "Q_QUERY": q_query}


def config_to_label(config):
    if isinstance(config, FewShotConfig):
        config = config.as_label_dict()
    return f"N{config['N_WAY']}_K{config['K_SHOT']}_Q{config['Q_QUERY']}"


def fewshot_config_to_dict(config):
    payload = asdict(config)
    payload["sample_n_way"] = int(config.sample_n_way)
    payload["k_shot"] = int(config.k_shot)
    payload["q_query"] = int(config.q_query)
    payload["inner_val"] = int(config.inner_val)
    payload["fewshot_config_label"] = config_to_label(config)
    return payload


def fewshot_config_to_json(config):
    return json.dumps(fewshot_config_to_dict(config), ensure_ascii=False, sort_keys=True)


def model_params_to_json(params):
    return json.dumps(params or {}, ensure_ascii=False, sort_keys=True)


class SkipDatasetError(ValueError):
    def __init__(self, reason, skip_reason_code):
        super().__init__(reason)
        self.skip_reason_code = str(skip_reason_code)


def make_skip_error(skip_reason_code, reason):
    return SkipDatasetError(reason, skip_reason_code)


def get_skip_reason_code(exc, default="unknown_skip_reason"):
    return str(getattr(exc, "skip_reason_code", default))


def aggregate_skip_reason_field(seed_runs, field_name, default_value=""):
    values = [str(item[field_name]) for item in seed_runs if item.get(field_name)]
    if not values:
        return default_value
    unique_values = sorted(set(values))
    if len(unique_values) == 1:
        return unique_values[0]
    return "[varies_by_seed]"


DATA_SCARCITY_SKIP_CODES = {
    "insufficient_global_classes",
    "insufficient_classes_for_n_way",
    "insufficient_classes_for_class_disjoint",
    "insufficient_meta_train_test_classes",
    "class_disjoint_partition_infeasible",
    "insufficient_samples_per_class",
    "insufficient_samples_for_episode",
    "insufficient_samples_for_inner_validation",
    "data_scarcity_during_meta_training",
}


def infer_adaptation_mode(config):
    return "explicit_config" if config.explicit_config else "auto_dataset_adaptation"


def aggregate_tuning_score(seed_runs):
    values = [
        float(item["tuning_score"])
        for item in seed_runs
        if "tuning_score" in item and not pd.isna(item["tuning_score"])
    ]
    return float(np.mean(values)) if values else np.nan


def aggregate_best_params(seed_runs):
    values = [
        str(item["best_params_json"])
        for item in seed_runs
        if item.get("best_params_json")
    ]
    if not values:
        return ""
    unique_values = sorted(set(values))
    if len(unique_values) == 1:
        return unique_values[0]
    return "[varies_by_seed]"


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_runtime(cpu_only=False, disable_xla=False, disable_mixed_precision=False, n_jobs=-1):
    global N_JOBS, PARALLEL_BACKEND_AVAILABLE
    N_JOBS = n_jobs
    PARALLEL_BACKEND_AVAILABLE = True

    if not disable_xla:
        tf.config.optimizer.set_jit(True)
        print("[accel] XLA activado.")
    else:
        print("[accel] XLA desactivado por argumento.")

    if cpu_only:
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        print(f"[accel] Modo CPU forzado. n_jobs={N_JOBS}.")
        return

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print(f"[accel] No se detecto GPU. Ejecucion en CPU con n_jobs={N_JOBS}.")
        return

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    if disable_mixed_precision:
        print(f"[accel] GPU detectada ({len(gpus)}). mixed_precision desactivado. n_jobs={N_JOBS}.")
    else:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print(f"[accel] GPU detectada ({len(gpus)}). mixed_precision=ON. n_jobs={N_JOBS}.")
        except Exception:
            print(
                f"[accel] GPU detectada ({len(gpus)}), pero no se pudo activar mixed_precision. "
                f"n_jobs={N_JOBS}."
            )


def clear_runtime_memory():
    # Libera grafo/estado retenido por Keras y fuerza GC entre corridas largas.
    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()


def resolve_input_path(path_text):
    path = Path(path_text)
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    script_candidate = SCRIPT_DIR / path
    return script_candidate


def reset_fewshot_config():
    return replace(BASE_FEWSHOT_CONFIG)


def apply_fewshot_config_override(base_config, config_override):
    if config_override is None:
        return base_config
    return base_config.with_override(
        config_override["N_WAY"],
        config_override["K_SHOT"],
        config_override["Q_QUERY"],
    )


def apply_quick_mode(config):
    return config.with_quick_mode()


def build_requested_fewshot_config(config_override=None, quick=False, explicit_config=False):
    config = reset_fewshot_config()
    config = apply_fewshot_config_override(config, config_override)
    config = replace(config, explicit_config=bool(explicit_config))
    if quick:
        config = apply_quick_mode(config)
    return config


def choose_auto_n_way(n_classes, requested_n_way, explicit_config):
    if explicit_config:
        return min(requested_n_way, n_classes)
    if n_classes <= 2:
        return 2
    if n_classes <= 5:
        return min(3, n_classes)
    if n_classes <= 10:
        return min(4, n_classes)
    return min(AUTO_N_WAY_CAP, n_classes)


def choose_class_disjoint_setup(n_classes, sample_n_way):
    if n_classes < 4:
        return (
            0,
            0,
            "dataset con menos de 4 clases; class-disjoint no aplica",
            "insufficient_classes_for_class_disjoint",
        )

    class_disjoint_n_way = min(sample_n_way, n_classes // 2)
    if class_disjoint_n_way < 2:
        return (
            0,
            0,
            "no hay suficientes clases para meta-train y meta-test",
            "insufficient_meta_train_test_classes",
        )

    # Repartimos las clases aproximadamente a la mitad para evitar splits muy sesgados.
    test_classes = n_classes // 2
    if test_classes < class_disjoint_n_way:
        test_classes = class_disjoint_n_way
    if (n_classes - test_classes) < class_disjoint_n_way:
        test_classes = n_classes - class_disjoint_n_way

    if test_classes < 2 or (n_classes - test_classes) < 2:
        return (
            0,
            0,
            "particion class-disjoint inviable para este dataset",
            "class_disjoint_partition_infeasible",
        )

    return class_disjoint_n_way, test_classes, "", ""


def choose_episode_shots(max_total_per_class, requested_k_shot, requested_q_query, explicit_config=False):
    if max_total_per_class < 2:
        raise make_skip_error(
            "insufficient_samples_for_episode",
            "Cada episodio necesita al menos 1 support y 1 query por clase.",
        )

    if explicit_config:
        k_shot = min(requested_k_shot, max_total_per_class - 1)
        q_query = min(requested_q_query, max_total_per_class - k_shot)
        return k_shot, max(1, q_query)

    if max_total_per_class <= 3:
        return 1, 1
    if max_total_per_class <= 5:
        return 1, 2
    if max_total_per_class <= 8:
        return 2, 3
    if max_total_per_class <= 12:
        return 3, 5

    k_shot = 5
    q_query = min(10, max_total_per_class - k_shot)
    return k_shot, max(1, q_query)


def choose_episode_budgets(n_samples):
    if n_samples <= 40:
        return 20, 10, 8
    if n_samples <= 100:
        return 30, 15, 10
    if n_samples <= 300:
        return 40, 20, 12
    return 50, 30, 16


def ci95(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    half = 1.96 * std / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return mean, std, mean - half, mean + half


def init_metrics(model_names):
    return {
        name: {"acc": [], "f1": [], "acc_by_seed": [], "f1_by_seed": [], "seed_runs": []}
        for name in model_names
    }


def init_seed_metrics(model_names):
    return {name: {"acc": [], "f1": []} for name in model_names}


def record_episode_metrics(metric_store, model_name, y_true, y_pred):
    metric_store[model_name]["acc"].append(accuracy_score(y_true, y_pred))
    metric_store[model_name]["f1"].append(f1_score(y_true, y_pred, average="weighted"))

def finalize_seed_metrics(metrics, seed_metrics, seed, seed_model_meta=None):
    seed_model_meta = seed_model_meta or {}
    for name, values in seed_metrics.items():
        seed_has_observations = bool(values["acc"])
        model_meta = seed_model_meta.get(name, {})
        metrics[name]["seed_runs"].append(
            {
                "seed": int(seed),
                "acc_mean": float(np.mean(values["acc"])) if seed_has_observations else np.nan,
                "f1_mean": float(np.mean(values["f1"])) if seed_has_observations else np.nan,
                "n_observations": len(values["acc"]),
                "status": "ok" if seed_has_observations else "skipped",
                "best_params_json": model_meta.get("best_params_json", ""),
                "tuning_score": model_meta.get("tuning_score", np.nan),
                "with_tuning": model_meta.get("with_tuning", np.nan),
                "skip_reason_code": model_meta.get("skip_reason_code", ""),
                "skip_reason": model_meta.get("skip_reason", ""),
                "tuning_status": model_meta.get("tuning_status", ""),
                "tuning_fallback_reason_code": model_meta.get("tuning_fallback_reason_code", ""),
                "tuning_fallback_reason": model_meta.get("tuning_fallback_reason", ""),
            }
        )
        if values["acc"]:
            metrics[name]["acc"].extend(values["acc"])
            metrics[name]["f1"].extend(values["f1"])
            metrics[name]["acc_by_seed"].append(float(np.mean(values["acc"])))
            metrics[name]["f1_by_seed"].append(float(np.mean(values["f1"])))


def metric_summary_values(metrics, name, metric_key):
    seed_key = f"{metric_key}_by_seed"
    if seed_key in metrics[name] and metrics[name][seed_key]:
        return metrics[name][seed_key]
    return metrics[name][metric_key]


def make_rng(base_seed, *parts):
    entropy = [int(base_seed)]
    for part in parts:
        text = str(part)
        rolling = 0
        for ch in text:
            rolling = ((rolling * 131) + ord(ch)) % (2**32)
        entropy.append(rolling)
    return np.random.default_rng(np.random.SeedSequence(entropy))


def episode_signature(episode_spec):
    signature = []
    for item in sorted(episode_spec, key=lambda x: int(x["class"])):
        signature.append(
            (
                int(item["class"]),
                tuple(np.sort(item["support_idx"]).tolist()),
                tuple(np.sort(item["query_idx"]).tolist()),
            )
        )
    return tuple(signature)


def build_class_indices(y_pool, class_pool=None):
    if class_pool is None:
        class_pool = np.unique(y_pool)
    return {
        int(cls): np.where(y_pool == cls)[0].astype(np.int32, copy=False)
        for cls in np.asarray(class_pool)
    }


def as_float_tensor(x):
    if tf.is_tensor(x):
        return tf.cast(x, tf.float32)
    return tf.convert_to_tensor(x, dtype=tf.float32)


def _finalize_episode_specs(specs, n_episodes):
    if len(specs) < n_episodes:
        print(
            f"[WARN] Solo se generaron {len(specs)} episodios unicos de {n_episodes} solicitados."
        )
    return specs


def build_episode_specs(y_pool, class_pool, n_way, k_shot, q_query, n_episodes, rng, class_indices=None):
    specs = []
    class_pool = np.asarray(class_pool)
    if class_indices is None:
        class_indices = build_class_indices(y_pool, class_pool)
    seen = set()
    max_attempts = max(n_episodes * 10, 50)
    attempts = 0
    while len(specs) < n_episodes and attempts < max_attempts:
        attempts += 1
        selected = rng.choice(class_pool, size=n_way, replace=False)
        per_class = []
        for cls in selected:
            idx = class_indices[int(cls)]
            need = k_shot + q_query
            if len(idx) == 0:
                raise make_skip_error(
                    "insufficient_samples_for_episode",
                    f"No hay muestras para la clase {cls} en el pool episodico.",
                )
            chosen = rng.choice(idx, size=need, replace=(len(idx) < need))
            per_class.append(
                {
                    "class": int(cls),
                    "support_idx": chosen[:k_shot].astype(np.int32, copy=False),
                    "query_idx": chosen[k_shot:].astype(np.int32, copy=False),
                }
            )
        signature = episode_signature(per_class)
        if signature in seen:
            continue
        seen.add(signature)
        specs.append(per_class)
    return _finalize_episode_specs(specs, n_episodes)


def materialize_episode_specs(X_pool, y_pool, episode_spec):
    support_x, support_y = [], []
    query_x, query_y = [], []

    for item in episode_spec:
        cls = item["class"]
        support_idx = item["support_idx"]
        query_idx = item["query_idx"]
        support_x.append(X_pool[support_idx])
        support_y.append(np.full(len(support_idx), cls))
        query_x.append(X_pool[query_idx])
        query_y.append(np.full(len(query_idx), cls))

    return (
        np.vstack(support_x),
        np.concatenate(support_y),
        np.vstack(query_x),
        np.concatenate(query_y),
    )


def build_cross_episode_specs(
    y_support_pool,
    y_query_pool,
    class_pool,
    n_way,
    k_shot,
    q_query,
    n_episodes,
    rng,
    support_class_indices=None,
    query_class_indices=None,
):
    specs = []
    class_pool = np.asarray(class_pool)
    if support_class_indices is None:
        support_class_indices = build_class_indices(y_support_pool, class_pool)
    if query_class_indices is None:
        query_class_indices = build_class_indices(y_query_pool, class_pool)
    seen = set()
    max_attempts = max(n_episodes * 10, 50)
    attempts = 0
    while len(specs) < n_episodes and attempts < max_attempts:
        attempts += 1
        selected = rng.choice(class_pool, size=n_way, replace=False)
        per_class = []
        for cls in selected:
            idx_s = support_class_indices[int(cls)]
            idx_q = query_class_indices[int(cls)]
            if len(idx_s) == 0 or len(idx_q) == 0:
                raise make_skip_error(
                    "insufficient_samples_for_episode",
                    f"No hay muestras suficientes para clase {cls} en split cruzado.",
                )
            support_idx = rng.choice(idx_s, size=k_shot, replace=(len(idx_s) < k_shot))
            query_idx = rng.choice(idx_q, size=q_query, replace=(len(idx_q) < q_query))
            per_class.append(
                {
                    "class": int(cls),
                    "support_idx": support_idx.astype(np.int32, copy=False),
                    "query_idx": query_idx.astype(np.int32, copy=False),
                }
            )
        signature = episode_signature(per_class)
        if signature in seen:
            continue
        seen.add(signature)
        specs.append(per_class)
    return _finalize_episode_specs(specs, n_episodes)


def materialize_cross_episode_specs(X_support_pool, y_support_pool, X_query_pool, y_query_pool, episode_spec):
    support_x, support_y = [], []
    query_x, query_y = [], []

    for item in episode_spec:
        cls = item["class"]
        support_idx = item["support_idx"]
        query_idx = item["query_idx"]
        support_x.append(X_support_pool[support_idx])
        support_y.append(np.full(len(support_idx), cls))
        query_x.append(X_query_pool[query_idx])
        query_y.append(np.full(len(query_idx), cls))

    return (
        np.vstack(support_x),
        np.concatenate(support_y),
        np.vstack(query_x),
        np.concatenate(query_y),
    )


def sample_episode(X_pool, y_pool, class_pool, n_way, k_shot, q_query, rng, class_indices=None):
    selected = rng.choice(class_pool, size=n_way, replace=False)
    if class_indices is None:
        class_indices = build_class_indices(y_pool, class_pool)

    support_x, support_y = [], []
    query_x, query_y = [], []

    for cls in selected:
        idx = class_indices[int(cls)]
        need = k_shot + q_query
        if len(idx) == 0:
            raise make_skip_error(
                "insufficient_samples_for_episode",
                f"No hay muestras para la clase {cls} en el pool episodico.",
            )
        chosen = rng.choice(idx, size=need, replace=(len(idx) < need))
        support_idx = chosen[:k_shot]
        query_idx = chosen[k_shot:]

        support_x.append(X_pool[support_idx])
        support_y.append(np.full(k_shot, cls))
        query_x.append(X_pool[query_idx])
        query_y.append(np.full(q_query, cls))

    return (
        np.vstack(support_x),
        np.concatenate(support_y),
        np.vstack(query_x),
        np.concatenate(query_y),
    )


def sample_episode_cross(
    X_support_pool,
    y_support_pool,
    X_query_pool,
    y_query_pool,
    class_pool,
    n_way,
    k_shot,
    q_query,
    rng,
    support_class_indices=None,
    query_class_indices=None,
):
    selected = rng.choice(class_pool, size=n_way, replace=False)
    if support_class_indices is None:
        support_class_indices = build_class_indices(y_support_pool, class_pool)
    if query_class_indices is None:
        query_class_indices = build_class_indices(y_query_pool, class_pool)

    support_x, support_y = [], []
    query_x, query_y = [], []

    for cls in selected:
        idx_s = support_class_indices[int(cls)]
        idx_q = query_class_indices[int(cls)]
        if len(idx_s) == 0 or len(idx_q) == 0:
            raise make_skip_error(
                "insufficient_samples_for_episode",
                f"No hay muestras suficientes para clase {cls} en split cruzado.",
            )
        support_idx = rng.choice(idx_s, size=k_shot, replace=(len(idx_s) < k_shot))
        query_idx = rng.choice(idx_q, size=q_query, replace=(len(idx_q) < q_query))

        support_x.append(X_support_pool[support_idx])
        support_y.append(np.full(k_shot, cls))
        query_x.append(X_query_pool[query_idx])
        query_y.append(np.full(q_query, cls))

    return (
        np.vstack(support_x),
        np.concatenate(support_y),
        np.vstack(query_x),
        np.concatenate(query_y),
    )


def split_classwise_train_val(X_pool, y_pool, class_pool, val_per_class, rng, class_indices=None):
    tr_idx, va_idx = [], []
    if class_indices is None:
        class_indices = build_class_indices(y_pool, class_pool)
    for cls in class_pool:
        idx = class_indices[int(cls)]
        idx = rng.permutation(idx)
        if len(idx) < 2:
            tr_idx.extend(idx.tolist())
            continue
        val_count = min(val_per_class, len(idx) - 1)
        va_idx.extend(idx[:val_count].tolist())
        tr_idx.extend(idx[val_count:].tolist())

    tr_idx = np.asarray(tr_idx, dtype=np.int32)
    va_idx = np.asarray(va_idx, dtype=np.int32)
    if len(va_idx) == 0:
        raise make_skip_error(
            "insufficient_samples_for_inner_validation",
            "No se pudo crear conjunto de validacion interno con las clases disponibles.",
        )
    return X_pool[tr_idx], y_pool[tr_idx], X_pool[va_idx], y_pool[va_idx]


def choose_encoder_spec(input_dim, n_samples=None, emb_dim=EMB_DIM):
    input_dim = max(1, int(input_dim))
    sample_count = None if n_samples is None else max(1, int(n_samples))

    if sample_count is None or sample_count > 100:
        hidden_units = (64, 64)
        effective_emb_dim = emb_dim
        l2_strength = 1e-5
    elif sample_count <= 12:
        hidden_units = ()
        effective_emb_dim = min(8, emb_dim)
        l2_strength = 5e-4
    elif sample_count <= 32:
        hidden_units = (16,)
        effective_emb_dim = min(12, emb_dim)
        l2_strength = 3e-4
    else:
        hidden_units = (32, 16)
        effective_emb_dim = min(16, emb_dim)
        l2_strength = 1e-4

    if sample_count is not None and hidden_units:
        width_cap = max(8, min(64, sample_count * 2))
        hidden_units = tuple(
            max(8, min(int(units), width_cap, max(8, input_dim)))
            for units in hidden_units
        )

    effective_emb_dim = max(4, min(int(effective_emb_dim), max(4, input_dim)))
    return EncoderSpec(
        input_dim=input_dim,
        emb_dim=effective_emb_dim,
        hidden_units=tuple(hidden_units),
        l2_strength=float(l2_strength),
    )


def describe_encoder_spec(spec):
    hidden_label = "linear" if not spec.hidden_units else "-".join(str(v) for v in spec.hidden_units)
    return (
        f"input_dim={spec.input_dim}, hidden={hidden_label}, emb_dim={spec.emb_dim}, "
        f"l2={spec.l2_strength:.1e}"
    )


def encoder_embedding_dim(encoder):
    return int(encoder.output_shape[-1])


def make_encoder(input_dim, emb_dim=EMB_DIM, n_samples=None):
    spec = choose_encoder_spec(input_dim=input_dim, n_samples=n_samples, emb_dim=emb_dim)
    regularizer = tf.keras.regularizers.l2(spec.l2_strength)
    layers = [tf.keras.layers.Input(shape=(spec.input_dim,))]
    for units in spec.hidden_units:
        layers.append(
            tf.keras.layers.Dense(
                units,
                activation="relu",
                kernel_regularizer=regularizer,
            )
        )
    layers.append(tf.keras.layers.Dense(spec.emb_dim, kernel_regularizer=regularizer))
    encoder = tf.keras.Sequential(layers)
    encoder.encoder_spec = spec
    return encoder


def encode_pair(encoder, support_x, query_x, training=False):
    support_x = as_float_tensor(support_x)
    query_x = as_float_tensor(query_x)
    support_emb = encoder(support_x, training=training)
    query_emb = encoder(query_x, training=training)
    return support_emb, query_emb


def build_relation_pair_tensor(support_emb, query_emb):
    q_exp = query_emb[:, None, :]
    s_exp = support_emb[None, :, :]
    q_tiled = tf.broadcast_to(q_exp, [tf.shape(query_emb)[0], tf.shape(support_emb)[0], tf.shape(query_emb)[1]])
    s_tiled = tf.broadcast_to(s_exp, [tf.shape(query_emb)[0], tf.shape(support_emb)[0], tf.shape(support_emb)[1]])
    pair_feat = tf.concat([q_tiled, s_tiled, tf.abs(q_tiled - s_tiled)], axis=-1)
    return tf.reshape(pair_feat, [-1, tf.shape(pair_feat)[-1]])


def class_mean_scores(pair_scores, support_y, classes):
    score_rows = []
    for cls in classes:
        mask = tf.convert_to_tensor(support_y == cls)
        cls_scores = tf.boolean_mask(pair_scores, mask, axis=1)
        score_rows.append(tf.reduce_mean(cls_scores, axis=1))
    return tf.stack(score_rows, axis=1)


def relation_pair_matrix(encoder, relation_head, support_x, query_x, training):
    support_emb, query_emb = encode_pair(encoder, support_x, query_x, training=training)
    flat_pairs = build_relation_pair_tensor(support_emb, query_emb)
    rel = relation_head(flat_pairs, training=training)
    return tf.reshape(rel, [tf.shape(query_emb)[0], tf.shape(support_emb)[0]])


def prototypical_logits(encoder, support_x, support_y, query_x, classes, training):
    support_emb, query_emb = encode_pair(encoder, support_x, query_x, training=training)

    prototypes = []
    for cls in classes:
        mask = tf.convert_to_tensor(support_y == cls)
        cls_emb = tf.boolean_mask(support_emb, mask)
        prototypes.append(tf.reduce_mean(cls_emb, axis=0))

    prototypes = tf.stack(prototypes, axis=0)
    dists = tf.reduce_sum(tf.square(query_emb[:, None, :] - prototypes[None, :, :]), axis=-1)
    return -dists


def cosine_logits(encoder, support_x, support_y, query_x, classes, training, scale=10.0):
    support_emb, query_emb = encode_pair(encoder, support_x, query_x, training=training)

    support_emb = tf.math.l2_normalize(support_emb, axis=-1)
    query_emb = tf.math.l2_normalize(query_emb, axis=-1)

    prototypes = []
    for cls in classes:
        mask = tf.convert_to_tensor(support_y == cls)
        cls_emb = tf.boolean_mask(support_emb, mask)
        p = tf.reduce_mean(cls_emb, axis=0)
        p = tf.math.l2_normalize(p, axis=-1)
        prototypes.append(p)

    prototypes = tf.stack(prototypes, axis=0)
    prototypes = tf.cast(prototypes, query_emb.dtype)
    logits = scale * tf.matmul(query_emb, prototypes, transpose_b=True)
    return logits


def matching_probs(encoder, support_x, support_y, query_x, classes, training, scale=10.0):
    support_emb, query_emb = encode_pair(encoder, support_x, query_x, training=training)

    support_emb = tf.math.l2_normalize(support_emb, axis=-1)
    query_emb = tf.math.l2_normalize(query_emb, axis=-1)

    sim = scale * tf.matmul(query_emb, support_emb, transpose_b=True)
    attn = tf.nn.softmax(sim, axis=1)

    class_map = {cls: i for i, cls in enumerate(classes)}
    support_local = np.array([class_map[c] for c in support_y], dtype=np.int32)
    onehot_support = tf.one_hot(support_local, depth=len(classes), dtype=attn.dtype)

    probs = tf.matmul(attn, onehot_support)
    return probs


def relation_scores(encoder, relation_head, support_x, support_y, query_x, classes, training):
    rel = relation_pair_matrix(encoder, relation_head, support_x, query_x, training=training)
    return class_mean_scores(rel, support_y, classes)


def build_relation_head(emb_dim=EMB_DIM):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(emb_dim * 3,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )


def build_siamese_head(emb_dim=EMB_DIM, hidden_units=64):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(emb_dim,)),
            tf.keras.layers.Dense(hidden_units, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )


def siamese_pair_scores(encoder, siamese_head, x1, x2, training):
    emb1, emb2 = encode_pair(encoder, x1, x2, training=training)
    feat = tf.abs(emb1 - emb2)
    scores = siamese_head(feat, training=training)
    return tf.squeeze(scores, axis=1)


def train_proto_step(encoder, optimizer, support_x, support_y, query_x, y_local, classes):
    with tf.GradientTape() as tape:
        logits = prototypical_logits(encoder, support_x, support_y, query_x, classes, training=True)
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(y_local, logits, from_logits=True)
        )
    grads = tape.gradient(loss, encoder.trainable_variables)
    optimizer.apply_gradients(zip(grads, encoder.trainable_variables))


def train_matching_step(encoder, optimizer, support_x, support_y, query_x, y_local, classes):
    with tf.GradientTape() as tape:
        probs = matching_probs(encoder, support_x, support_y, query_x, classes, training=True)
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(y_local, probs, from_logits=False)
        )
    grads = tape.gradient(loss, encoder.trainable_variables)
    optimizer.apply_gradients(zip(grads, encoder.trainable_variables))


def train_relation_step(encoder, relation_head, optimizer, support_x, query_x, pair_labels):
    with tf.GradientTape() as tape:
        pair_scores = relation_pair_matrix(encoder, relation_head, support_x, query_x, training=True)
        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(pair_labels, pair_scores))
    vars_all = encoder.trainable_variables + relation_head.trainable_variables
    grads = tape.gradient(loss, vars_all)
    optimizer.apply_gradients(zip(grads, vars_all))


def train_siamese_step(encoder, siamese_head, optimizer, x1, x2, y_bin):
    with tf.GradientTape() as tape:
        scores = siamese_pair_scores(encoder, siamese_head, x1, x2, training=True)
        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_bin, scores))
    vars_all = encoder.trainable_variables + siamese_head.trainable_variables
    grads = tape.gradient(loss, vars_all)
    optimizer.apply_gradients(zip(grads, vars_all))


def train_meta_encoder_proto(
    encoder,
    X_train,
    y_train,
    train_classes,
    rng,
    config,
    learning_rate=LEARNING_RATE,
    train_episodes=None,
    episode_specs=None,
    n_way=None,
):
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_episodes = config.train_episodes if train_episodes is None else train_episodes
    episode_n_way = config.sample_n_way if n_way is None else n_way

    if episode_specs is None:
        episode_specs = build_episode_specs(
            y_train, train_classes, episode_n_way, config.k_shot, config.q_query, train_episodes, rng
        )

    for episode_spec in episode_specs:
        s_x, s_y, q_x, q_y = materialize_episode_specs(X_train, y_train, episode_spec)
        classes = np.unique(s_y)
        c2i = {c: i for i, c in enumerate(classes)}
        y_local = tf.convert_to_tensor([c2i[v] for v in q_y], dtype=tf.int32)
        train_proto_step(encoder, opt, s_x, s_y, q_x, y_local, classes)


def train_meta_encoder_matching(
    encoder,
    X_train,
    y_train,
    train_classes,
    rng,
    config,
    learning_rate=LEARNING_RATE,
    train_episodes=None,
    episode_specs=None,
    n_way=None,
):
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_episodes = config.train_episodes if train_episodes is None else train_episodes
    episode_n_way = config.sample_n_way if n_way is None else n_way

    if episode_specs is None:
        episode_specs = build_episode_specs(
            y_train, train_classes, episode_n_way, config.k_shot, config.q_query, train_episodes, rng
        )

    for episode_spec in episode_specs:
        s_x, s_y, q_x, q_y = materialize_episode_specs(X_train, y_train, episode_spec)
        classes = np.unique(s_y)
        c2i = {c: i for i, c in enumerate(classes)}
        y_local = tf.convert_to_tensor([c2i[v] for v in q_y], dtype=tf.int32)
        train_matching_step(encoder, opt, s_x, s_y, q_x, y_local, classes)


def train_meta_relation(
    encoder,
    relation_head,
    X_train,
    y_train,
    train_classes,
    rng,
    config,
    learning_rate=LEARNING_RATE,
    train_episodes=None,
    episode_specs=None,
    n_way=None,
):
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_episodes = config.train_episodes if train_episodes is None else train_episodes
    episode_n_way = config.sample_n_way if n_way is None else n_way

    if episode_specs is None:
        episode_specs = build_episode_specs(
            y_train, train_classes, episode_n_way, config.k_shot, config.q_query, train_episodes, rng
        )

    for episode_spec in episode_specs:
        s_x, s_y, q_x, q_y = materialize_episode_specs(X_train, y_train, episode_spec)
        labels = tf.convert_to_tensor(
            np.asarray([(s_y == q_label).astype(np.float32) for q_label in q_y]),
            dtype=tf.float32,
        )
        train_relation_step(encoder, relation_head, opt, s_x, q_x, labels)


def train_meta_siamese(
    encoder,
    siamese_head,
    X_train,
    y_train,
    train_classes,
    rng,
    config,
    learning_rate=LEARNING_RATE,
    train_episodes=None,
    episode_specs=None,
    n_way=None,
):
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_episodes = config.train_episodes if train_episodes is None else train_episodes
    episode_n_way = config.sample_n_way if n_way is None else n_way

    if episode_specs is None:
        episode_specs = build_episode_specs(
            y_train, train_classes, episode_n_way, config.k_shot, config.q_query, train_episodes, rng
        )

    for episode_spec in episode_specs:
        s_x, s_y, q_x, q_y = materialize_episode_specs(X_train, y_train, episode_spec)

        ep_x = np.vstack([s_x, q_x])
        ep_y = np.concatenate([s_y, q_y])
        classes = np.unique(ep_y)
        idx_by_class = {cls: np.where(ep_y == cls)[0] for cls in classes}

        pos_i, pos_j = [], []
        for cls in classes:
            idx = idx_by_class[cls]
            if len(idx) < 2:
                continue
            n_pos_cls = max(1, len(idx) // 2)
            i1 = rng.choice(idx, size=n_pos_cls, replace=True)
            i2 = rng.choice(idx, size=n_pos_cls, replace=True)
            pos_i.append(i1)
            pos_j.append(i2)

        if not pos_i:
            continue

        pos_i = np.concatenate(pos_i).astype(np.int32)
        pos_j = np.concatenate(pos_j).astype(np.int32)
        n_pos = len(pos_i)

        neg_i = np.empty(n_pos, dtype=np.int32)
        neg_j = np.empty(n_pos, dtype=np.int32)
        for k in range(n_pos):
            c1, c2 = rng.choice(classes, size=2, replace=False)
            neg_i[k] = rng.choice(idx_by_class[c1])
            neg_j[k] = rng.choice(idx_by_class[c2])

        x1 = np.vstack([ep_x[pos_i], ep_x[neg_i]])
        x2 = np.vstack([ep_x[pos_j], ep_x[neg_j]])
        y_bin = np.concatenate(
            [np.ones(n_pos, dtype=np.float32), np.zeros(n_pos, dtype=np.float32)]
        )

        perm = rng.permutation(len(y_bin))
        x1 = x1[perm]
        x2 = x2[perm]
        y_bin = y_bin[perm]
        y_bin = tf.convert_to_tensor(y_bin, dtype=tf.float32)
        train_siamese_step(encoder, siamese_head, opt, x1, x2, y_bin)


def tune_classical_model(
    X_inner_train,
    y_inner_train,
    X_inner_val,
    y_inner_val,
    classes,
    rng,
    config,
    param_grid,
    builder_fn,
    episodes=None,
):
    episodes = config.classical_tune_episodes if episodes is None else episodes
    best_score = -1.0
    best_params = {}
    episode_specs = build_cross_episode_specs(
        y_inner_train, y_inner_val, classes, config.sample_n_way, config.k_shot, config.q_query, episodes, rng
    )
    for params in param_grid:
        accs = []
        for episode_spec in episode_specs:
            s_x, s_y, q_x, q_y = materialize_cross_episode_specs(
                X_inner_train, y_inner_train, X_inner_val, y_inner_val, episode_spec
            )
            if np.unique(s_y).size < 2:
                pred = constant_prediction(s_y, len(q_y))
                accs.append(accuracy_score(q_y, pred))
                continue
            clf = builder_fn(params)
            if hasattr(clf, "n_neighbors"):
                clf.set_params(n_neighbors=min(int(clf.n_neighbors), len(s_x)))
            clf = fit_classifier_with_fallback(clf, s_x, s_y, builder_fn=builder_fn, params=params)
            pred = clf.predict(q_x)
            accs.append(accuracy_score(q_y, pred))
        score = float(np.mean(accs))
        if score > best_score:
            best_score = score
            best_params = dict(params)
    return best_params, best_score


def tune_knn(X_inner_train, y_inner_train, X_inner_val, y_inner_val, classes, rng, config, episodes=None):
    max_neighbors = max(1, min(len(X_inner_train), len(classes) * max(1, config.k_shot)))
    valid_ks = sorted({1} | {k for k in KNN_K_GRID if k <= max_neighbors})
    grid = [
        {"n_neighbors": k, "weights": w, "p": p}
        for k in valid_ks
        for w in KNN_WEIGHTS_GRID
        for p in KNN_P_GRID
    ]
    return tune_classical_model(
        X_inner_train,
        y_inner_train,
        X_inner_val,
        y_inner_val,
        classes,
        rng,
        config,
        grid,
        lambda p: build_knn_classifier(
            n_neighbors=p["n_neighbors"],
            weights=p["weights"],
            p=p["p"],
            n_samples=len(X_inner_train),
        ),
        episodes=episodes,
    )


def tune_logreg(X_inner_train, y_inner_train, X_inner_val, y_inner_val, classes, rng, config, episodes=None):
    grid = [{"C": c, "penalty": penalty} for c in LOGREG_C_GRID for penalty in LOGREG_PENALTY_GRID]
    return tune_classical_model(
        X_inner_train,
        y_inner_train,
        X_inner_val,
        y_inner_val,
        classes,
        rng,
        config,
        grid,
        lambda p: build_logreg_classifier(
            C=p["C"],
            penalty=p["penalty"],
            random_state=0,
            n_samples=len(X_inner_train),
        ),
        episodes=episodes,
    )


def tune_mlp(X_inner_train, y_inner_train, X_inner_val, y_inner_val, classes, rng, config, episodes=None):
    grid = [
        {"hidden_layer_sizes": h, "alpha": a, "activation": act}
        for h in MLP_HIDDEN_GRID
        for a in MLP_ALPHA_GRID
        for act in MLP_ACTIVATION_GRID
    ]
    return tune_classical_model(
        X_inner_train,
        y_inner_train,
        X_inner_val,
        y_inner_val,
        classes,
        rng,
        config,
        grid,
        lambda p: build_mlp_classifier(
            hidden_layer_sizes=p["hidden_layer_sizes"],
            alpha=p["alpha"],
            activation=p["activation"],
            random_state=0,
            n_samples=len(X_inner_train),
        ),
        episodes=episodes,
    )


def tune_rf(X_inner_train, y_inner_train, X_inner_val, y_inner_val, classes, rng, config, episodes=None):
    grid = [
        {"n_estimators": n, "max_depth": d, "min_samples_leaf": leaf}
        for n in RF_ESTIMATORS_GRID
        for d in RF_MAX_DEPTH_GRID
        for leaf in RF_MIN_SAMPLES_LEAF_GRID
    ]
    return tune_classical_model(
        X_inner_train,
        y_inner_train,
        X_inner_val,
        y_inner_val,
        classes,
        rng,
        config,
        grid,
        lambda p: build_rf_classifier(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            min_samples_leaf=p["min_samples_leaf"],
            random_state=0,
        ),
        episodes=episodes,
    )


def tune_svm_rbf(X_inner_train, y_inner_train, X_inner_val, y_inner_val, classes, rng, config, episodes=None):
    grid = [{"C": c, "gamma": g} for c in SVM_C_GRID for g in SVM_GAMMA_GRID]
    return tune_classical_model(
        X_inner_train,
        y_inner_train,
        X_inner_val,
        y_inner_val,
        classes,
        rng,
        config,
        grid,
        lambda p: build_svm_classifier(C=p["C"], gamma=p["gamma"], n_samples=len(X_inner_train)),
        episodes=episodes,
    )


def eval_meta_predictor(
    predictor_fn,
    X_inner_train,
    y_inner_train,
    X_inner_val,
    y_inner_val,
    classes,
    rng,
    config,
    episodes=None,
):
    episodes = config.meta_tune_eval_episodes if episodes is None else episodes
    episode_specs = build_cross_episode_specs(
        y_inner_train, y_inner_val, classes, config.sample_n_way, config.k_shot, config.q_query, episodes, rng
    )
    return eval_meta_predictor_on_specs(
        predictor_fn, X_inner_train, y_inner_train, X_inner_val, y_inner_val, episode_specs
    )


def eval_meta_predictor_on_specs(
    predictor_fn, X_inner_train, y_inner_train, X_inner_val, y_inner_val, episode_specs
):
    accs = []
    for episode_spec in episode_specs:
        s_x, s_y, q_x, q_y = materialize_cross_episode_specs(
            X_inner_train, y_inner_train, X_inner_val, y_inner_val, episode_spec
        )
        pred = predictor_fn(s_x, s_y, q_x)
        accs.append(accuracy_score(q_y, pred))
    return float(np.mean(accs))


def tune_proto_lr(
    X_inner_train,
    y_inner_train,
    X_inner_val,
    y_inner_val,
    classes,
    rng,
    input_dim,
    config,
    train_episodes=None,
):
    train_episodes = config.meta_tune_train_episodes if train_episodes is None else train_episodes
    best_score = -1.0
    best_params = {"learning_rate": LEARNING_RATE}
    train_episode_specs = build_episode_specs(
        y_inner_train,
        classes,
        config.sample_n_way,
        config.k_shot,
        config.q_query,
        train_episodes,
        make_rng(rng.integers(2**32), "proto_train"),
    )
    eval_episode_specs = build_cross_episode_specs(
        y_inner_train,
        y_inner_val,
        classes,
        config.sample_n_way,
        config.k_shot,
        config.q_query,
        config.meta_tune_eval_episodes,
        make_rng(rng.integers(2**32), "proto_eval"),
    )
    for learning_rate in PROTO_LR_GRID:
        encoder = make_encoder(input_dim, n_samples=len(X_inner_train))
        train_meta_encoder_proto(
            encoder,
            X_inner_train,
            y_inner_train,
            classes,
            rng,
            config,
            learning_rate=learning_rate,
            train_episodes=train_episodes,
            episode_specs=train_episode_specs,
        )
        score = eval_meta_predictor_on_specs(
            lambda s_x, s_y, q_x: predict_protonet(encoder, s_x, s_y, q_x),
            X_inner_train,
            y_inner_train,
            X_inner_val,
            y_inner_val,
            eval_episode_specs,
        )
        if score > best_score:
            best_score = score
            best_params = {"learning_rate": learning_rate}
    return best_params, best_score


def tune_matching_lr(
    X_inner_train,
    y_inner_train,
    X_inner_val,
    y_inner_val,
    classes,
    rng,
    input_dim,
    config,
    train_episodes=None,
):
    train_episodes = config.meta_tune_train_episodes if train_episodes is None else train_episodes
    best_score = -1.0
    best_params = {"learning_rate": LEARNING_RATE}
    train_episode_specs = build_episode_specs(
        y_inner_train,
        classes,
        config.sample_n_way,
        config.k_shot,
        config.q_query,
        train_episodes,
        make_rng(rng.integers(2**32), "matching_train"),
    )
    eval_episode_specs = build_cross_episode_specs(
        y_inner_train,
        y_inner_val,
        classes,
        config.sample_n_way,
        config.k_shot,
        config.q_query,
        config.meta_tune_eval_episodes,
        make_rng(rng.integers(2**32), "matching_eval"),
    )
    for learning_rate in MATCHING_LR_GRID:
        encoder = make_encoder(input_dim, n_samples=len(X_inner_train))
        train_meta_encoder_matching(
            encoder,
            X_inner_train,
            y_inner_train,
            classes,
            rng,
            config,
            learning_rate=learning_rate,
            train_episodes=train_episodes,
            episode_specs=train_episode_specs,
        )
        score = eval_meta_predictor_on_specs(
            lambda s_x, s_y, q_x: predict_matching(encoder, s_x, s_y, q_x),
            X_inner_train,
            y_inner_train,
            X_inner_val,
            y_inner_val,
            eval_episode_specs,
        )
        if score > best_score:
            best_score = score
            best_params = {"learning_rate": learning_rate}
    return best_params, best_score


def tune_relation_lr(
    X_inner_train,
    y_inner_train,
    X_inner_val,
    y_inner_val,
    classes,
    rng,
    input_dim,
    config,
    train_episodes=None,
):
    train_episodes = config.meta_tune_train_episodes if train_episodes is None else train_episodes
    best_score = -1.0
    best_params = {"learning_rate": LEARNING_RATE}
    train_episode_specs = build_episode_specs(
        y_inner_train,
        classes,
        config.sample_n_way,
        config.k_shot,
        config.q_query,
        train_episodes,
        make_rng(rng.integers(2**32), "relation_train"),
    )
    eval_episode_specs = build_cross_episode_specs(
        y_inner_train,
        y_inner_val,
        classes,
        config.sample_n_way,
        config.k_shot,
        config.q_query,
        config.meta_tune_eval_episodes,
        make_rng(rng.integers(2**32), "relation_eval"),
    )
    for learning_rate in RELATION_LR_GRID:
        encoder = make_encoder(input_dim, n_samples=len(X_inner_train))
        relation_head = build_relation_head(emb_dim=encoder_embedding_dim(encoder))
        train_meta_relation(
            encoder,
            relation_head,
            X_inner_train,
            y_inner_train,
            classes,
            rng,
            config,
            learning_rate=learning_rate,
            train_episodes=train_episodes,
            episode_specs=train_episode_specs,
        )
        score = eval_meta_predictor_on_specs(
            lambda s_x, s_y, q_x: predict_relation(encoder, relation_head, s_x, s_y, q_x),
            X_inner_train,
            y_inner_train,
            X_inner_val,
            y_inner_val,
            eval_episode_specs,
        )
        if score > best_score:
            best_score = score
            best_params = {"learning_rate": learning_rate}
    return best_params, best_score


def tune_siamese_hparams(
    X_inner_train,
    y_inner_train,
    X_inner_val,
    y_inner_val,
    classes,
    rng,
    input_dim,
    config,
    train_episodes=None,
):
    train_episodes = config.siamese_tune_train_episodes if train_episodes is None else train_episodes
    best_score = -1.0
    best_params = {"hidden_units": 64, "learning_rate": LEARNING_RATE}
    eval_episode_specs = build_cross_episode_specs(
        y_inner_train,
        y_inner_val,
        classes,
        config.sample_n_way,
        config.k_shot,
        config.q_query,
        config.meta_tune_eval_episodes,
        make_rng(rng.integers(2**32), "siamese_eval"),
    )

    for hidden_units in SIAMESE_HIDDEN_GRID:
        for learning_rate in SIAMESE_LR_GRID:
            train_episode_specs = build_episode_specs(
                y_inner_train,
                classes,
                config.sample_n_way,
                config.k_shot,
                config.q_query,
                train_episodes,
                make_rng(rng.integers(2**32), "siamese_train", hidden_units, learning_rate),
            )
            encoder = make_encoder(input_dim, n_samples=len(X_inner_train))
            siamese_head = build_siamese_head(
                emb_dim=encoder_embedding_dim(encoder),
                hidden_units=hidden_units,
            )

            train_meta_siamese(
                encoder,
                siamese_head,
                X_inner_train,
                y_inner_train,
                classes,
                rng,
                config,
                learning_rate=learning_rate,
                train_episodes=train_episodes,
                episode_specs=train_episode_specs,
            )

            score = eval_meta_predictor_on_specs(
                lambda s_x, s_y, q_x: predict_siamese(encoder, siamese_head, s_x, s_y, q_x),
                X_inner_train,
                y_inner_train,
                X_inner_val,
                y_inner_val,
                eval_episode_specs,
            )
            if score > best_score:
                best_score = score
                best_params = {"hidden_units": hidden_units, "learning_rate": learning_rate}

    return best_params, best_score


def tune_proto_based_heads(
    encoder,
    X_inner_train,
    y_inner_train,
    X_inner_val,
    y_inner_val,
    classes,
    rng,
    config,
):
    eval_episode_specs = build_cross_episode_specs(
        y_inner_train,
        y_inner_val,
        classes,
        config.sample_n_way,
        config.k_shot,
        config.q_query,
        config.meta_tune_eval_episodes,
        rng,
    )
    best_cosine = {"scale": 10.0}
    best_cosine_score = -1.0
    for scale in COSINE_SCALE_GRID:
        score = eval_meta_predictor_on_specs(
            lambda s_x, s_y, q_x: predict_cosine(encoder, s_x, s_y, q_x, scale=scale),
            X_inner_train,
            y_inner_train,
            X_inner_val,
            y_inner_val,
            eval_episode_specs,
        )
        if score > best_cosine_score:
            best_cosine_score = score
            best_cosine = {"scale": scale}

    best_cov = {"reg": 1e-3}
    best_cov_score = -1.0
    for reg in COV_REG_GRID:
        score = eval_meta_predictor_on_specs(
            lambda s_x, s_y, q_x: predict_covariance(encoder, s_x, s_y, q_x, reg=reg),
            X_inner_train,
            y_inner_train,
            X_inner_val,
            y_inner_val,
            eval_episode_specs,
        )
        if score > best_cov_score:
            best_cov_score = score
            best_cov = {"reg": reg}

    best_tpn = {"alpha": 0.8, "sigma": 1.0}
    best_tpn_score = -1.0
    for alpha in TPN_ALPHA_GRID:
        for sigma in TPN_SIGMA_GRID:
            score = eval_meta_predictor_on_specs(
                lambda s_x, s_y, q_x: predict_tpn(encoder, s_x, s_y, q_x, alpha=alpha, sigma=sigma),
                X_inner_train,
                y_inner_train,
                X_inner_val,
                y_inner_val,
                eval_episode_specs,
            )
            if score > best_tpn_score:
                best_tpn_score = score
                best_tpn = {"alpha": alpha, "sigma": sigma}

    return (
        best_cosine,
        best_cosine_score,
        best_cov,
        best_cov_score,
        best_tpn,
        best_tpn_score,
    )


def compute_episode_embeddings(encoder, s_x, q_x):
    s_emb, q_emb = encode_pair(encoder, s_x, q_x, training=False)
    return s_emb.numpy(), q_emb.numpy()


def predict_protonet(encoder, s_x, s_y, q_x, support_emb=None, query_emb=None):
    classes = np.unique(s_y)
    if support_emb is None or query_emb is None:
        logits = prototypical_logits(encoder, s_x, s_y, q_x, classes, training=False)
    else:
        support_emb = as_float_tensor(support_emb)
        query_emb = as_float_tensor(query_emb)
        prototypes = []
        for cls in classes:
            mask = tf.convert_to_tensor(s_y == cls)
            cls_emb = tf.boolean_mask(support_emb, mask)
            prototypes.append(tf.reduce_mean(cls_emb, axis=0))
        prototypes = tf.stack(prototypes, axis=0)
        logits = -tf.reduce_sum(tf.square(query_emb[:, None, :] - prototypes[None, :, :]), axis=-1)
    pred_local = tf.argmax(logits, axis=1).numpy()
    return np.array([classes[i] for i in pred_local])


def predict_matching(encoder, s_x, s_y, q_x, support_emb=None, query_emb=None):
    classes = np.unique(s_y)
    if support_emb is None or query_emb is None:
        probs = matching_probs(encoder, s_x, s_y, q_x, classes, training=False)
    else:
        support_emb = tf.math.l2_normalize(as_float_tensor(support_emb), axis=-1)
        query_emb = tf.math.l2_normalize(as_float_tensor(query_emb), axis=-1)
        sim = 10.0 * tf.matmul(query_emb, support_emb, transpose_b=True)
        attn = tf.nn.softmax(sim, axis=1)
        class_map = {cls: i for i, cls in enumerate(classes)}
        support_local = np.array([class_map[c] for c in s_y], dtype=np.int32)
        onehot_support = tf.one_hot(support_local, depth=len(classes), dtype=attn.dtype)
        probs = tf.matmul(attn, onehot_support)
    pred_local = tf.argmax(probs, axis=1).numpy()
    return np.array([classes[i] for i in pred_local])


def predict_relation(encoder, relation_head, s_x, s_y, q_x, support_emb=None, query_emb=None):
    classes = np.unique(s_y)
    if support_emb is None or query_emb is None:
        scores = relation_scores(encoder, relation_head, s_x, s_y, q_x, classes, training=False)
    else:
        support_emb = as_float_tensor(support_emb)
        query_emb = as_float_tensor(query_emb)
        pair_feat = build_relation_pair_tensor(support_emb, query_emb)
        pair_scores = relation_head(pair_feat, training=False)
        pair_scores = tf.reshape(pair_scores, [tf.shape(query_emb)[0], tf.shape(support_emb)[0]])
        scores = class_mean_scores(pair_scores, s_y, classes)
    pred_local = tf.argmax(scores, axis=1).numpy()
    return np.array([classes[i] for i in pred_local])


def predict_cosine(encoder, s_x, s_y, q_x, scale=10.0, support_emb=None, query_emb=None):
    classes = np.unique(s_y)
    if support_emb is None or query_emb is None:
        logits = cosine_logits(encoder, s_x, s_y, q_x, classes, training=False, scale=scale)
    else:
        support_emb = tf.math.l2_normalize(as_float_tensor(support_emb), axis=-1)
        query_emb = tf.math.l2_normalize(as_float_tensor(query_emb), axis=-1)
        prototypes = []
        for cls in classes:
            mask = tf.convert_to_tensor(s_y == cls)
            cls_emb = tf.boolean_mask(support_emb, mask)
            p = tf.reduce_mean(cls_emb, axis=0)
            prototypes.append(tf.math.l2_normalize(p, axis=-1))
        prototypes = tf.stack(prototypes, axis=0)
        logits = scale * tf.matmul(query_emb, prototypes, transpose_b=True)
    pred_local = tf.argmax(logits, axis=1).numpy()
    return np.array([classes[i] for i in pred_local])


def predict_covariance(encoder, s_x, s_y, q_x, reg=1e-3, support_emb=None, query_emb=None):
    if support_emb is None or query_emb is None:
        s_emb, q_emb = compute_episode_embeddings(encoder, s_x, q_x)
    else:
        s_emb = np.asarray(support_emb, dtype=np.float32)
        q_emb = np.asarray(query_emb, dtype=np.float32)

    classes = np.unique(s_y)
    scores = np.zeros((len(q_x), len(classes)), dtype=np.float64)
    emb_dim = s_emb.shape[1] if s_emb.ndim > 1 else 1

    for i, cls in enumerate(classes):
        cls_emb = s_emb[s_y == cls]
        mu = np.mean(cls_emb, axis=0)

        # In 1-shot or tiny per-class support sets, the empirical covariance is
        # undefined or collapses to a scalar. Fall back to isotropic covariance.
        if cls_emb.shape[0] < 2:
            cov = reg * np.eye(emb_dim, dtype=np.float64)
        else:
            cov = np.cov(cls_emb, rowvar=False)
            cov = np.atleast_2d(cov).astype(np.float64, copy=False)
            if cov.shape != (emb_dim, emb_dim):
                cov = reg * np.eye(emb_dim, dtype=np.float64)
            else:
                cov = cov + reg * np.eye(emb_dim, dtype=np.float64)

        inv_cov = np.linalg.pinv(cov)

        diff = q_emb - mu
        maha = np.sum((diff @ inv_cov) * diff, axis=1)
        scores[:, i] = -maha

    pred_local = np.argmax(scores, axis=1)
    return np.array([classes[i] for i in pred_local])


def predict_tpn(encoder, s_x, s_y, q_x, alpha=0.8, sigma=1.0, support_emb=None, query_emb=None):
    if support_emb is None or query_emb is None:
        s_emb, q_emb = compute_episode_embeddings(encoder, s_x, q_x)
    else:
        s_emb = np.asarray(support_emb, dtype=np.float32)
        q_emb = np.asarray(query_emb, dtype=np.float32)

    X_all = np.vstack([s_emb, q_emb])
    n_support = len(s_emb)
    n_total = len(X_all)

    sq = np.sum((X_all[:, None, :] - X_all[None, :, :]) ** 2, axis=2)
    W = np.exp(-sq / (2 * sigma * sigma))
    np.fill_diagonal(W, 0.0)

    d = np.sum(W, axis=1) + 1e-8
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
    S = D_inv_sqrt @ W @ D_inv_sqrt

    classes = np.unique(s_y)
    c2i = {c: i for i, c in enumerate(classes)}
    Y0 = np.zeros((n_total, len(classes)), dtype=np.float64)

    for i, label in enumerate(s_y):
        Y0[i, c2i[label]] = 1.0

    I = np.eye(n_total)
    F = (1 - alpha) * np.linalg.solve(I - alpha * S, Y0)
    query_scores = F[n_support:]
    pred_local = np.argmax(query_scores, axis=1)
    return np.array([classes[i] for i in pred_local])


def predict_siamese(encoder, siamese_head, s_x, s_y, q_x, support_emb=None, query_emb=None):
    classes = np.unique(s_y)
    if support_emb is None or query_emb is None:
        support_emb, query_emb = compute_episode_embeddings(encoder, s_x, q_x)
    else:
        support_emb = np.asarray(support_emb, dtype=np.float32)
        query_emb = np.asarray(query_emb, dtype=np.float32)
    support_emb_tf = as_float_tensor(support_emb)
    query_emb_tf = as_float_tensor(query_emb)
    q_exp = query_emb_tf[:, None, :]
    s_exp = support_emb_tf[None, :, :]
    pair_feat = tf.abs(q_exp - s_exp)
    flat_feat = tf.reshape(pair_feat, [-1, tf.shape(pair_feat)[-1]])
    sim_scores = siamese_head(flat_feat, training=False)
    sim_scores = tf.reshape(sim_scores, [len(query_emb), len(support_emb)]).numpy()

    class_scores = np.stack([np.mean(sim_scores[:, s_y == cls], axis=1) for cls in classes], axis=1)
    pred_local = np.argmax(class_scores, axis=1)
    return np.array([classes[i] for i in pred_local])


def visualize_episode_embeddings(seed, episode_idx, encoder, s_x, s_y, q_x, q_y, dataset_name, config):
    if not SAVE_EMBEDDINGS:
        return
    s_emb, q_emb = compute_episode_embeddings(encoder, s_x, q_x)

    all_emb = np.vstack([s_emb, q_emb])
    all_y = np.concatenate([s_y, q_y])
    all_type = np.array(["support"] * len(s_emb) + ["query"] * len(q_emb))

    pca = PCA(n_components=2)
    z = pca.fit_transform(all_emb)

    plt.figure(figsize=(8, 6))
    for cls in np.unique(all_y):
        for t, marker in [("support", "*"), ("query", "o")]:
            mask = (all_y == cls) & (all_type == t)
            plt.scatter(z[mask, 0], z[mask, 1], marker=marker, s=90, label=f"clase {cls} - {t}")

    handles, labels = plt.gca().get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    plt.legend(uniq.values(), uniq.keys(), fontsize=8, loc="best", ncol=2)
    plt.title(f"Embeddings episodicos (seed={seed}, episodio={episode_idx})")
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.tight_layout()
    # Acorta el nombre para evitar rutas largas en Windows (límite ~260 chars).
    safe_name = Path(str(dataset_name)).stem
    safe_name = "".join(ch if ch.isalnum() else "_" for ch in safe_name).strip("_")[:80]
    config_name = config_to_label(config)
    out_dir = Path("embeddings")
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"embeddings_{safe_name}_{config_name}_ep{episode_idx}_seed{seed}.png"
    plt.savefig(out, dpi=150)
    plt.close()


# Datos base
def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark few-shot con carga generica de datasets CSV."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Ruta de un CSV individual. Si no se indica, se evaluan los CSV reducidos de --reduced-root.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default=None,
        help="Nombre de la columna objetivo. Si no se indica, se usa la ultima columna.",
    )
    parser.add_argument(
        "--sep",
        type=str,
        default=",",
        help="Separador del CSV (por defecto ',').",
    )
    parser.add_argument(
        "--reduced-root",
        type=str,
        default="reduced_csv",
        help="Carpeta raiz para buscar CSV reducidos (por defecto 'reduced_csv').",
    )
    parser.add_argument(
        "--include-iris",
        action="store_true",
        help="En modo lote (sin --csv), agrega tambien la evaluacion de Iris base.",
    )
    parser.add_argument(
        "--skip-iris",
        action="store_true",
        help="Compatibilidad hacia atras. Ya no hace falta: Iris se omite por defecto en modo lote.",
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        default="fewshot_results.csv",
        help="Ruta de salida para resultados en CSV (por defecto 'fewshot_results.csv').",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Lista de seeds separadas por coma, por ejemplo '7,21,42'.",
    )
    parser.add_argument(
        "--fewshot-config",
        action="append",
        default=[],
        help="Configuracion few-shot en formato N:K:Q. Puedes repetir el argumento varias veces.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=min(4, max(1, (os.cpu_count() or 2) - 1)),
        help="Hilos para modelos clasicos (kNN/RF). Usa -1 para todos los nucleos.",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Fuerza ejecucion en CPU (sin GPU TensorFlow).",
    )
    parser.add_argument(
        "--disable-xla",
        action="store_true",
        help="Desactiva compilacion XLA de TensorFlow.",
    )
    parser.add_argument(
        "--disable-mixed-precision",
        action="store_true",
        help="Desactiva mixed precision en GPU.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Reduce episodios de tuning/evaluacion para pruebas rapidas.",
    )
    return parser.parse_args()


def load_dataset(csv_path=None, target_column=None, sep=","):
    if csv_path is None:
        X_data, y_data = load_iris(return_X_y=True)
        return X_data.astype(np.float32), y_data.astype(np.int32), "Iris"

    df = pd.read_csv(csv_path, sep=sep)
    if df.empty:
        raise make_skip_error("empty_dataset", f"El archivo CSV esta vacio: {csv_path}")

    if target_column is None:
        target_column = df.columns[-1]
    if target_column not in df.columns:
        raise make_skip_error(
            "missing_target_column",
            f"La columna objetivo '{target_column}' no existe en el CSV. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    y_series = df[target_column].fillna("missing").astype(str)
    X_df = df.drop(columns=[target_column]).copy()
    if X_df.shape[1] == 0:
        raise make_skip_error(
            "no_feature_columns",
            "El CSV debe tener al menos una columna de caracteristicas.",
        )

    for col in X_df.columns:
        if pd.api.types.is_numeric_dtype(X_df[col]):
            X_df[col] = X_df[col].fillna(X_df[col].median())
        else:
            mode = X_df[col].mode(dropna=True)
            fill_val = mode.iloc[0] if not mode.empty else "missing"
            X_df[col] = X_df[col].fillna(fill_val).astype(str)

    X_df = pd.get_dummies(X_df, drop_first=False)
    X_data = X_df.to_numpy(dtype=np.float32)
    y_data, _ = pd.factorize(y_series, sort=True)

    approx_mb = X_data.nbytes / (1024**2)
    if X_data.shape[1] >= 10000:
        print(
            f"[WARN] Dataset muy ancho tras get_dummies: {X_data.shape[1]} columnas. "
            "Esto puede elevar mucho el uso de RAM."
        )
    if approx_mb >= 512:
        print(
            f"[WARN] Matriz de entrada grande en memoria: ~{approx_mb:.1f} MiB "
            f"({X_data.shape[0]} filas x {X_data.shape[1]} columnas)."
        )

    return X_data, y_data.astype(np.int32), csv_path


def adapt_fewshot_config(y_data, config):
    classes = np.unique(y_data)
    n_classes = len(classes)
    if n_classes < 2:
        raise make_skip_error(
            "insufficient_global_classes",
            "Se requieren al menos 2 clases para evaluar few-shot.",
        )

    dataset_regime = "binary" if n_classes == 2 else "multiclass"
    requested_n_way = config.requested_n_way
    effective_n_way = choose_auto_n_way(n_classes, requested_n_way, config.explicit_config)

    class_counts = np.asarray([np.sum(y_data == cls) for cls in classes], dtype=np.int32)
    min_per_class = int(np.min(class_counts))
    counts_str = ", ".join([f"{int(cls)}:{int(cnt)}" for cls, cnt in zip(classes, class_counts)])
    # Modo robusto para datasets reducidos: al menos 1 por clase en train y 1 en test.
    min_test_required = 1
    min_train_required = 1

    if min_per_class < (min_test_required + min_train_required):
        raise make_skip_error(
            "insufficient_samples_per_class",
            f"No hay suficientes muestras por clase (min={min_per_class}). "
            f"Se requieren al menos {min_test_required + min_train_required} por clase "
            "para split estratificado few-shot. "
            f"Conteos detectados: [{counts_str}]."
        )

    test_size_min = min_test_required / float(min_per_class)
    test_size_max = 1.0 - (min_train_required / float(min_per_class))
    effective_test_size = config.test_size
    if not (test_size_min <= effective_test_size <= test_size_max):
        old_test_size = effective_test_size
        effective_test_size = float(np.clip(effective_test_size, test_size_min, test_size_max))
        print(
            f"[adapt_fewshot_config] TEST_SIZE ajustado de {old_test_size:.3f} a {effective_test_size:.3f} "
            f"para respetar minimos por clase (min_per_class={min_per_class})."
        )

    min_test_per_class = max(1, int(np.floor(min_per_class * effective_test_size)))
    if min_test_per_class >= min_per_class:
        min_test_per_class = min_per_class - 1
    min_train_per_class = min_per_class - min_test_per_class

    max_total_for_episode = max(2, min(min_per_class, min_train_per_class, min_test_per_class))
    if max_total_for_episode < 2:
        raise make_skip_error(
            "insufficient_samples_for_episode",
            "No hay suficientes muestras por clase para episodios few-shot con el split actual. "
            "Prueba aumentando datos o ajustando TEST_SIZE."
        )

    requested_k_shot = config.requested_k_shot
    requested_q_query = config.requested_q_query
    effective_k_shot, effective_q_query = choose_episode_shots(
        max_total_for_episode,
        requested_k_shot,
        requested_q_query,
        explicit_config=config.explicit_config,
    )
    inner_val_per_class = effective_k_shot + effective_q_query

    old_train_episodes = config.train_episodes
    old_eval_episodes = config.eval_episodes
    old_tune_episodes = config.tune_episodes
    old_classical_tune = config.classical_tune_episodes
    old_meta_tune_eval = config.meta_tune_eval_episodes
    old_meta_tune_train = config.meta_tune_train_episodes
    old_siamese_tune_train = config.siamese_tune_train_episodes

    train_episode_budget, eval_episode_budget, tune_episode_budget = choose_episode_budgets(len(y_data))

    train_episodes = train_episode_budget
    eval_episodes = eval_episode_budget
    tune_episodes = tune_episode_budget
    classical_tune_episodes = max(4, tune_episode_budget // 2)
    meta_tune_eval_episodes = max(4, tune_episode_budget // 2)
    meta_tune_train_episodes = max(8, train_episode_budget)
    siamese_tune_train_episodes = max(8, train_episode_budget)

    (
        class_disjoint_n_way,
        class_disjoint_test_classes,
        class_disjoint_status,
        class_disjoint_status_code,
    ) = choose_class_disjoint_setup(n_classes, effective_n_way)

    adapted_config = replace(
        config,
        test_size=effective_test_size,
        train_episodes=train_episodes,
        eval_episodes=eval_episodes,
        tune_episodes=tune_episodes,
        classical_tune_episodes=classical_tune_episodes,
        meta_tune_eval_episodes=meta_tune_eval_episodes,
        meta_tune_train_episodes=meta_tune_train_episodes,
        siamese_tune_train_episodes=siamese_tune_train_episodes,
        effective_n_way=effective_n_way,
        effective_k_shot=effective_k_shot,
        effective_q_query=effective_q_query,
        inner_val_per_class=inner_val_per_class,
        dataset_regime=dataset_regime,
        class_disjoint_n_way=class_disjoint_n_way,
        class_disjoint_test_classes=class_disjoint_test_classes,
        class_disjoint_status=class_disjoint_status,
        class_disjoint_status_code=class_disjoint_status_code,
    )
    if adapted_config.quick_mode:
        adapted_config = apply_quick_mode(adapted_config)

    print(
        "[adapt_fewshot_config] "
        f"regimen={adapted_config.dataset_regime}, clases={n_classes}, min_por_clase={min_per_class}, "
        f"K={adapted_config.k_shot}, Q={adapted_config.q_query}, inner_val={adapted_config.inner_val}, "
        f"train_ep={adapted_config.train_episodes}, eval_ep={adapted_config.eval_episodes}, "
        f"tune_ep={adapted_config.tune_episodes}"
    )
    if adapted_config.explicit_config:
        print(
            "[adapt_fewshot_config] "
            f"N-way solicitado={requested_n_way}; N-way efectivo={adapted_config.sample_n_way}."
        )
    else:
        print(
            "[adapt_fewshot_config] "
            f"N-way auto={adapted_config.sample_n_way} (cap={AUTO_N_WAY_CAP}) segun el numero de clases del dataset."
        )
    if adapted_config.class_disjoint_n_way >= 2:
        print(
            "[adapt_fewshot_config] "
            f"class-disjoint habilitado con N-way={adapted_config.class_disjoint_n_way}, "
            f"test_classes={adapted_config.class_disjoint_test_classes}, "
            f"train_classes={n_classes - adapted_config.class_disjoint_test_classes}."
        )
    else:
        print(
            "[adapt_fewshot_config] "
            f"class-disjoint desactivado automaticamente: {adapted_config.class_disjoint_status}."
        )
    if adapted_config.k_shot != requested_k_shot or adapted_config.q_query != requested_q_query:
        print(
            "[adapt_fewshot_config] "
            f"episodio ajustado por escasez de muestras: "
            f"K {requested_k_shot}->{adapted_config.k_shot}, Q {requested_q_query}->{adapted_config.q_query}."
        )
    if (
        old_train_episodes != adapted_config.train_episodes
        or old_eval_episodes != adapted_config.eval_episodes
        or old_tune_episodes != adapted_config.tune_episodes
        or old_classical_tune != adapted_config.classical_tune_episodes
        or old_meta_tune_eval != adapted_config.meta_tune_eval_episodes
        or old_meta_tune_train != adapted_config.meta_tune_train_episodes
        or old_siamese_tune_train != adapted_config.siamese_tune_train_episodes
    ):
        print(
            "[adapt_fewshot_config] episodios ajustados: "
            f"train {old_train_episodes}->{adapted_config.train_episodes}, "
            f"eval {old_eval_episodes}->{adapted_config.eval_episodes}, "
            f"tune {old_tune_episodes}->{adapted_config.tune_episodes}, "
            f"classical_tune {old_classical_tune}->{adapted_config.classical_tune_episodes}, "
            f"meta_eval {old_meta_tune_eval}->{adapted_config.meta_tune_eval_episodes}, "
            f"meta_train {old_meta_tune_train}->{adapted_config.meta_tune_train_episodes}, "
            f"siamese_train {old_siamese_tune_train}->{adapted_config.siamese_tune_train_episodes}"
        )
    return adapted_config

sample_disjoint_model_names = [
    "kNN",
    "kNN_ProtoEmb",
    "SVM_RBF",
    "SVM_RBF_ProtoEmb",
    "LogReg",
    "LogReg_ProtoEmb",
    "MLP",
    "MLP_ProtoEmb",
    "RandomForest",
    "RandomForest_ProtoEmb",
    "ProtoNet",
    "MatchingNet",
    "RelationNet",
    "CosineClassifier",
    "CovarianceNet",
    "TransductivePropagationNet",
    "SiameseNet",
]
class_disjoint_model_names = list(sample_disjoint_model_names)
meta_dependent_model_names = [
    "kNN_ProtoEmb",
    "SVM_RBF_ProtoEmb",
    "LogReg_ProtoEmb",
    "MLP_ProtoEmb",
    "RandomForest_ProtoEmb",
    "ProtoNet",
    "MatchingNet",
    "RelationNet",
    "CosineClassifier",
    "CovarianceNet",
    "TransductivePropagationNet",
    "SiameseNet",
]


def constant_prediction(support_labels, n_query):
    cls = int(np.asarray(support_labels)[0])
    return np.full(n_query, cls, dtype=np.int32)


def effective_n_jobs():
    return N_JOBS if PARALLEL_BACKEND_AVAILABLE else 1


def disable_parallel_backend(exc):
    global PARALLEL_BACKEND_AVAILABLE
    if PARALLEL_BACKEND_AVAILABLE:
        PARALLEL_BACKEND_AVAILABLE = False
        print(
            "[WARN] Paralelismo desactivado para modelos clasicos por error de permisos "
            f"({exc}). Se reintentara con n_jobs=1."
        )


def fit_classifier_with_fallback(clf, s_x, s_y, builder_fn=None, params=None):
    try:
        clf.fit(s_x, s_y)
        return clf
    except PermissionError as exc:
        disable_parallel_backend(exc)
        if builder_fn is None or params is None:
            raise
        retry_clf = builder_fn(params)
        if hasattr(retry_clf, "n_neighbors"):
            retry_clf.set_params(n_neighbors=min(int(retry_clf.n_neighbors), len(s_x)))
        retry_clf.fit(s_x, s_y)
        return retry_clf


def build_knn_classifier(n_neighbors, weights, p, n_samples):
    capped_k = max(1, min(int(n_neighbors), int(n_samples)))
    return KNeighborsClassifier(
        n_neighbors=capped_k,
        weights=weights,
        p=p,
        metric="minkowski",
        n_jobs=effective_n_jobs(),
    )


def build_logreg_classifier(C, penalty, random_state, n_samples):
    solver = "liblinear" if penalty == "l1" or n_samples <= 50 else "lbfgs"
    return LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        max_iter=400,
        tol=1e-3,
        class_weight="balanced",
        random_state=random_state,
    )


def build_svm_classifier(C, gamma, n_samples):
    return SVC(
        kernel="rbf",
        C=C,
        gamma=gamma,
        class_weight="balanced" if n_samples <= 100 else None,
    )


def build_mlp_classifier(hidden_layer_sizes, alpha, activation, random_state, n_samples):
    capped_hidden = tuple(max(4, min(int(h), max(4, n_samples))) for h in hidden_layer_sizes)
    return MLPClassifier(
        hidden_layer_sizes=capped_hidden,
        alpha=alpha,
        activation=activation,
        solver="lbfgs",
        max_iter=400,
        tol=1e-3,
        random_state=random_state,
    )


def build_rf_classifier(n_estimators, max_depth, min_samples_leaf, random_state):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=effective_n_jobs(),
    )


def add_classical_predictions(preds, prefix, s_x, s_y, q_x, seed, knn_params, svm_params, logreg_params, mlp_params, rf_params):
    knn = build_knn_classifier(
        n_neighbors=knn_params["n_neighbors"],
        weights=knn_params["weights"],
        p=knn_params["p"],
        n_samples=len(s_x),
    )
    knn = fit_classifier_with_fallback(
        knn,
        s_x,
        s_y,
        builder_fn=lambda p: build_knn_classifier(
            n_neighbors=p["n_neighbors"],
            weights=p["weights"],
            p=p["p"],
            n_samples=len(s_x),
        ),
        params=knn_params,
    )
    preds[f"kNN{prefix}"] = knn.predict(q_x)

    svm_rbf = build_svm_classifier(
        C=svm_params["C"],
        gamma=svm_params["gamma"],
        n_samples=len(s_x),
    )
    svm_rbf.fit(s_x, s_y)
    preds[f"SVM_RBF{prefix}"] = svm_rbf.predict(q_x)

    logreg = build_logreg_classifier(
        C=logreg_params["C"],
        penalty=logreg_params["penalty"],
        random_state=seed,
        n_samples=len(s_x),
    )
    logreg.fit(s_x, s_y)
    preds[f"LogReg{prefix}"] = logreg.predict(q_x)

    mlp = build_mlp_classifier(
        hidden_layer_sizes=mlp_params["hidden_layer_sizes"],
        alpha=mlp_params["alpha"],
        activation=mlp_params["activation"],
        random_state=seed,
        n_samples=len(s_x),
    )
    mlp.fit(s_x, s_y)
    preds[f"MLP{prefix}"] = mlp.predict(q_x)

    rf = build_rf_classifier(
        n_estimators=rf_params["n_estimators"],
        max_depth=rf_params["max_depth"],
        min_samples_leaf=rf_params["min_samples_leaf"],
        random_state=seed,
    )
    rf = fit_classifier_with_fallback(
        rf,
        s_x,
        s_y,
        builder_fn=lambda p: build_rf_classifier(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            min_samples_leaf=p["min_samples_leaf"],
            random_state=seed,
        ),
        params=rf_params,
    )
    preds[f"RandomForest{prefix}"] = rf.predict(q_x)


def default_tuned_params():
    return (
        {"n_neighbors": 3, "weights": "distance", "p": 2},
        0.0,
        {"C": 10.0, "gamma": "scale"},
        0.0,
        {"C": 3.0, "penalty": "l2"},
        0.0,
        {"hidden_layer_sizes": (32, 16), "alpha": 1e-4, "activation": "relu"},
        0.0,
        {"n_estimators": 240, "max_depth": None, "min_samples_leaf": 1},
        0.0,
        {"learning_rate": LEARNING_RATE},
        0.0,
        {"scale": 10.0},
        0.0,
        {"reg": 1e-3},
        0.0,
        {"alpha": 0.8, "sigma": 1.0},
        0.0,
        {"learning_rate": LEARNING_RATE},
        0.0,
        {"learning_rate": LEARNING_RATE},
        0.0,
        {"hidden_units": 64, "learning_rate": LEARNING_RATE},
        0.0,
    )


def build_protocol_model_metadata(
    *,
    with_tuning,
    knn_params,
    knn_score=np.nan,
    svm_params,
    svm_score=np.nan,
    logreg_params,
    logreg_score=np.nan,
    mlp_params,
    mlp_score=np.nan,
    rf_params,
    rf_score=np.nan,
    proto_params,
    proto_score=np.nan,
    cosine_params,
    cosine_score=np.nan,
    cov_params,
    cov_score=np.nan,
    tpn_params,
    tpn_score=np.nan,
    matching_params,
    matching_score=np.nan,
    relation_params,
    relation_score=np.nan,
    siamese_params,
    siamese_score=np.nan,
    tuning_status=None,
    tuning_fallback_reason_code="",
    tuning_fallback_reason="",
):
    def meta_entry(params, tuning_score):
        return {
            "best_params_json": model_params_to_json(params),
            "tuning_score": tuning_score,
            "with_tuning": with_tuning,
            "skip_reason_code": "",
            "skip_reason": "",
            "tuning_status": (
                "not_requested" if not with_tuning else ("ok" if tuning_status is None else str(tuning_status))
            ),
            "tuning_fallback_reason_code": str(tuning_fallback_reason_code),
            "tuning_fallback_reason": str(tuning_fallback_reason),
        }

    return {
        "kNN": meta_entry(knn_params, knn_score),
        "kNN_ProtoEmb": meta_entry(knn_params, knn_score),
        "SVM_RBF": meta_entry(svm_params, svm_score),
        "SVM_RBF_ProtoEmb": meta_entry(svm_params, svm_score),
        "LogReg": meta_entry(logreg_params, logreg_score),
        "LogReg_ProtoEmb": meta_entry(logreg_params, logreg_score),
        "MLP": meta_entry(mlp_params, mlp_score),
        "MLP_ProtoEmb": meta_entry(mlp_params, mlp_score),
        "RandomForest": meta_entry(rf_params, rf_score),
        "RandomForest_ProtoEmb": meta_entry(rf_params, rf_score),
        "ProtoNet": meta_entry(proto_params, proto_score),
        "MatchingNet": meta_entry(matching_params, matching_score),
        "RelationNet": meta_entry(relation_params, relation_score),
        "CosineClassifier": meta_entry(cosine_params, cosine_score),
        "CovarianceNet": meta_entry(cov_params, cov_score),
        "TransductivePropagationNet": meta_entry(tpn_params, tpn_score),
        "SiameseNet": meta_entry(siamese_params, siamese_score),
    }


def is_data_scarcity_error(exc):
    if isinstance(exc, SkipDatasetError):
        return get_skip_reason_code(exc) in DATA_SCARCITY_SKIP_CODES
    if not isinstance(exc, ValueError):
        return False

    text = str(exc).lower()
    scarcity_markers = [
        "expected n_neighbors <=",
        "needs samples of at least 2 classes",
        "the number of classes has to be greater than one",
        "least populated class in y has only 1 member",
        "the least populated class in y has only 1 member",
        "cannot have number of splits",
        "n_splits=",
        "train_size = 0",
        "resulting train set will be empty",
        "the resulting train set will be empty",
        "found array with 0 sample",
        "found input variables with inconsistent numbers of samples",
    ]
    return any(marker in text for marker in scarcity_markers)


def print_results(title, metrics, model_names):
    print(f"\n=== {title} ===")
    for name in model_names:
        acc_mean, acc_std, acc_lo, acc_hi = ci95(metric_summary_values(metrics, name, "acc"))
        f1_mean, f1_std, f1_lo, f1_hi = ci95(metric_summary_values(metrics, name, "f1"))
        print(
            f"{name:27s} | "
            f"Acc: {acc_mean:.4f} +/- {acc_std:.4f} | IC95% [{acc_lo:.4f}, {acc_hi:.4f}] | "
            f"F1: {f1_mean:.4f} +/- {f1_std:.4f} | IC95% [{f1_lo:.4f}, {f1_hi:.4f}]"
        )


def run_class_disjoint_no_tuning(config, X, y, dataset_name):
    metrics = init_metrics(class_disjoint_model_names)
    print("\n=== Evaluacion Few-Shot Class-Disjoint (sin tuning, clasico + meta-learning) ===")
    print(
        f"Dataset: {dataset_name} | N-way={config.class_disjoint_n_way}, "
        f"K-shot={config.k_shot}, Q-query={config.q_query} | Eval episodes: {config.eval_episodes}"
    )
    print(f"Semillas: {SEEDS}")

    classes = np.unique(y)
    if config.class_disjoint_test_classes < 2 or config.class_disjoint_n_way < 2:
        detail = config.class_disjoint_status or "se requieren al menos 4 clases globales para meta-train/test"
        print(f"[INFO] Class-disjoint omitido automaticamente: {detail}.")
        return metrics
    if len(classes) < config.class_disjoint_test_classes:
        print("[WARN] Class-disjoint omitido: no hay suficientes clases.")
        return metrics
    if config.class_disjoint_n_way > config.class_disjoint_test_classes:
        print("[WARN] Class-disjoint omitido: N-way invalido para test classes.")
        return metrics

    (
        best_knn_params,
        _,
        best_svm_params,
        _,
        best_logreg_params,
        _,
        best_mlp_params,
        _,
        best_rf_params,
        _,
        best_proto_params,
        _,
        best_cosine_params,
        _,
        best_cov_params,
        _,
        best_tpn_params,
        _,
        best_matching_params,
        _,
        best_relation_params,
        _,
        best_siamese_params,
        _,
    ) = default_tuned_params()
    protocol_model_meta = build_protocol_model_metadata(
        with_tuning=False,
        knn_params=best_knn_params,
        svm_params=best_svm_params,
        logreg_params=best_logreg_params,
        mlp_params=best_mlp_params,
        rf_params=best_rf_params,
        proto_params=best_proto_params,
        cosine_params=best_cosine_params,
        cov_params=best_cov_params,
        tpn_params=best_tpn_params,
        matching_params=best_matching_params,
        relation_params=best_relation_params,
        siamese_params=best_siamese_params,
    )

    for seed in SEEDS:
        set_seed(seed)
        rng = np.random.default_rng(seed)
        seed_model_meta = {name: dict(meta) for name, meta in protocol_model_meta.items()}
        tuning_status = "ok"
        tuning_fallback_reason_code = ""
        tuning_fallback_reason = ""

        class_perm = rng.permutation(classes)
        test_classes = class_perm[: config.class_disjoint_test_classes]
        train_classes = class_perm[config.class_disjoint_test_classes :]

        tr_mask = np.isin(y, train_classes)
        te_mask = np.isin(y, test_classes)
        X_train_pool, y_train_pool = X[tr_mask], y[tr_mask]
        X_test_pool, y_test_pool = X[te_mask], y[te_mask]

        scaler = StandardScaler()
        X_train_pool = scaler.fit_transform(X_train_pool)
        X_test_pool = scaler.transform(X_test_pool)
        train_class_indices = build_class_indices(y_train_pool, train_classes)
        test_class_indices = build_class_indices(y_test_pool, test_classes)
        seed_metrics = init_seed_metrics(class_disjoint_model_names)

        print(
            f"\n[Seed {seed}] class-train={train_classes.tolist()} | "
            f"class-test={test_classes.tolist()}"
        )

        fewshot_available = len(train_classes) >= config.class_disjoint_n_way
        fewshot_skip_code = ""
        fewshot_skip_reason = ""
        if fewshot_available:
            try:
                proto_encoder = make_encoder(X_train_pool.shape[1], n_samples=len(X_train_pool))
                matching_encoder = make_encoder(X_train_pool.shape[1], n_samples=len(X_train_pool))
                relation_encoder = make_encoder(X_train_pool.shape[1], n_samples=len(X_train_pool))
                relation_head = build_relation_head(emb_dim=encoder_embedding_dim(relation_encoder))
                siamese_encoder = make_encoder(X_train_pool.shape[1], n_samples=len(X_train_pool))
                siamese_head = build_siamese_head(
                    emb_dim=encoder_embedding_dim(siamese_encoder),
                    hidden_units=best_siamese_params["hidden_units"],
                )
                print(
                    f"[Seed {seed}] Encoder meta adaptativo: "
                    f"{describe_encoder_spec(proto_encoder.encoder_spec)}"
                )

                proto_train_specs = build_episode_specs(
                    y_train_pool,
                    train_classes,
                    config.class_disjoint_n_way,
                    config.k_shot,
                    config.q_query,
                    config.train_episodes,
                    make_rng(seed, "class_disjoint", "train_proto_specs"),
                    class_indices=train_class_indices,
                )
                matching_train_specs = build_episode_specs(
                    y_train_pool,
                    train_classes,
                    config.class_disjoint_n_way,
                    config.k_shot,
                    config.q_query,
                    config.train_episodes,
                    make_rng(seed, "class_disjoint", "train_matching_specs"),
                    class_indices=train_class_indices,
                )
                relation_train_specs = build_episode_specs(
                    y_train_pool,
                    train_classes,
                    config.class_disjoint_n_way,
                    config.k_shot,
                    config.q_query,
                    config.train_episodes,
                    make_rng(seed, "class_disjoint", "train_relation_specs"),
                    class_indices=train_class_indices,
                )
                siamese_train_specs = build_episode_specs(
                    y_train_pool,
                    train_classes,
                    config.class_disjoint_n_way,
                    config.k_shot,
                    config.q_query,
                    config.train_episodes,
                    make_rng(seed, "class_disjoint", "train_siamese_specs"),
                    class_indices=train_class_indices,
                )

                train_meta_encoder_proto(
                    proto_encoder,
                    X_train_pool,
                    y_train_pool,
                    train_classes,
                    make_rng(seed, "class_disjoint", "train_proto"),
                    config,
                    learning_rate=best_proto_params["learning_rate"],
                    episode_specs=proto_train_specs,
                    n_way=config.class_disjoint_n_way,
                )
                train_meta_encoder_matching(
                    matching_encoder,
                    X_train_pool,
                    y_train_pool,
                    train_classes,
                    make_rng(seed, "class_disjoint", "train_matching"),
                    config,
                    learning_rate=best_matching_params["learning_rate"],
                    episode_specs=matching_train_specs,
                    n_way=config.class_disjoint_n_way,
                )
                train_meta_relation(
                    relation_encoder,
                    relation_head,
                    X_train_pool,
                    y_train_pool,
                    train_classes,
                    make_rng(seed, "class_disjoint", "train_relation"),
                    config,
                    learning_rate=best_relation_params["learning_rate"],
                    episode_specs=relation_train_specs,
                    n_way=config.class_disjoint_n_way,
                )
                train_meta_siamese(
                    siamese_encoder,
                    siamese_head,
                    X_train_pool,
                    y_train_pool,
                    train_classes,
                    make_rng(seed, "class_disjoint", "train_siamese"),
                    config,
                    learning_rate=best_siamese_params["learning_rate"],
                    episode_specs=siamese_train_specs,
                    n_way=config.class_disjoint_n_way,
                )
            except Exception as exc:
                if not is_data_scarcity_error(exc) and not isinstance(exc, PermissionError):
                    raise
                fewshot_available = False
                reason = "datos escasos" if is_data_scarcity_error(exc) else "entorno/restricciones"
                fewshot_skip_code = (
                    "data_scarcity_during_meta_training"
                    if is_data_scarcity_error(exc)
                    else "environment_restriction"
                )
                fewshot_skip_reason = f"meta-learning class-disjoint omitido por {reason} ({exc})."
                tuning_status = "fallback_defaults"
                tuning_fallback_reason_code = fewshot_skip_code
                tuning_fallback_reason = fewshot_skip_reason
                print(f"[WARN] Seed {seed}: meta-learning class-disjoint omitido por {reason} ({exc}).")
        else:
            fewshot_skip_code = "insufficient_meta_train_classes"
            fewshot_skip_reason = (
                f"class-disjoint meta-learning omitido; solo hay {len(train_classes)} clases de meta-train."
            )
            tuning_status = "fallback_defaults"
            tuning_fallback_reason_code = fewshot_skip_code
            tuning_fallback_reason = fewshot_skip_reason
            print(
                f"[WARN] Seed {seed}: class-disjoint meta-learning omitido; "
                f"solo hay {len(train_classes)} clases de meta-train."
            )

        if not fewshot_available:
            for model_name in meta_dependent_model_names:
                seed_model_meta[model_name]["skip_reason_code"] = fewshot_skip_code
                seed_model_meta[model_name]["skip_reason"] = fewshot_skip_reason
                seed_model_meta[model_name]["tuning_status"] = tuning_status
                seed_model_meta[model_name]["tuning_fallback_reason_code"] = tuning_fallback_reason_code
                seed_model_meta[model_name]["tuning_fallback_reason"] = tuning_fallback_reason

        eval_episode_specs = build_episode_specs(
            y_test_pool,
            test_classes,
            config.class_disjoint_n_way,
            config.k_shot,
            config.q_query,
            config.eval_episodes,
            make_rng(seed, "class_disjoint", "eval_specs"),
            class_indices=test_class_indices,
        )
        for ep, episode_spec in enumerate(eval_episode_specs, start=1):
            s_x, s_y, q_x, q_y = materialize_episode_specs(X_test_pool, y_test_pool, episode_spec)
            preds = {}
            if np.unique(s_y).size < 2:
                const_pred = constant_prediction(s_y, len(q_y))
                for name in class_disjoint_model_names:
                    preds[name] = const_pred
            else:
                add_classical_predictions(
                    preds,
                    "",
                    s_x,
                    s_y,
                    q_x,
                    seed,
                    best_knn_params,
                    best_svm_params,
                    best_logreg_params,
                    best_mlp_params,
                    best_rf_params,
                )
                if fewshot_available:
                    support_emb_proto, query_emb_proto = compute_episode_embeddings(proto_encoder, s_x, q_x)
                    add_classical_predictions(
                        preds,
                        "_ProtoEmb",
                        support_emb_proto,
                        s_y,
                        query_emb_proto,
                        seed,
                        best_knn_params,
                        best_svm_params,
                        best_logreg_params,
                        best_mlp_params,
                        best_rf_params,
                    )
                    preds["ProtoNet"] = predict_protonet(
                        proto_encoder, s_x, s_y, q_x, support_emb=support_emb_proto, query_emb=query_emb_proto
                    )
                    preds["MatchingNet"] = predict_matching(matching_encoder, s_x, s_y, q_x)
                    preds["RelationNet"] = predict_relation(relation_encoder, relation_head, s_x, s_y, q_x)
                    preds["CosineClassifier"] = predict_cosine(
                        proto_encoder,
                        s_x,
                        s_y,
                        q_x,
                        scale=best_cosine_params["scale"],
                        support_emb=support_emb_proto,
                        query_emb=query_emb_proto,
                    )
                    preds["CovarianceNet"] = predict_covariance(
                        proto_encoder,
                        s_x,
                        s_y,
                        q_x,
                        reg=best_cov_params["reg"],
                        support_emb=support_emb_proto,
                        query_emb=query_emb_proto,
                    )
                    preds["TransductivePropagationNet"] = predict_tpn(
                        proto_encoder,
                        s_x,
                        s_y,
                        q_x,
                        alpha=best_tpn_params["alpha"],
                        sigma=best_tpn_params["sigma"],
                        support_emb=support_emb_proto,
                        query_emb=query_emb_proto,
                    )
                    preds["SiameseNet"] = predict_siamese(siamese_encoder, siamese_head, s_x, s_y, q_x)

            for name, pred in preds.items():
                record_episode_metrics(seed_metrics, name, q_y, pred)

            if ep == 1 and fewshot_available:
                visualize_episode_embeddings(seed, ep, proto_encoder, s_x, s_y, q_x, q_y, dataset_name, config)
        finalize_seed_metrics(metrics, seed_metrics, seed, seed_model_meta=seed_model_meta)
        clear_runtime_memory()
    return metrics


def run_sample_disjoint_tuning(config, X, y, dataset_name):
    metrics = init_metrics(sample_disjoint_model_names)
    print("\n=== Evaluacion Few-Shot Sample-Disjoint (con tuning) ===")
    print(f"Dataset: {dataset_name} | N-way={config.sample_n_way}, K-shot={config.k_shot}, Q-query={config.q_query}")
    print(f"Train episodes meta: {config.train_episodes} | Eval episodes: {config.eval_episodes}")
    print(f"Semillas: {SEEDS}")

    for seed in SEEDS:
        set_seed(seed)
        classes = np.unique(y)
        if len(classes) < config.sample_n_way:
            raise make_skip_error(
                "insufficient_classes_for_n_way",
                "No hay suficientes clases para el N-way configurado.",
            )

        X_train_pool, X_test_pool, y_train_pool, y_test_pool = train_test_split(
            X, y, test_size=config.test_size, stratify=y, random_state=seed
        )

        scaler = StandardScaler()
        X_train_pool = scaler.fit_transform(X_train_pool)
        X_test_pool = scaler.transform(X_test_pool)
        train_class_indices = build_class_indices(y_train_pool, classes)
        test_class_indices = build_class_indices(y_test_pool, classes)
        seed_metrics = init_seed_metrics(sample_disjoint_model_names)
        tuning_status = "ok"
        tuning_fallback_reason_code = ""
        tuning_fallback_reason = ""

        print(f"\n[Seed {seed}] clases disponibles={classes.tolist()}")

        try:
            split_rng = make_rng(seed, "sample_disjoint", "inner_split")
            X_inner_train, y_inner_train, X_inner_val, y_inner_val = split_classwise_train_val(
                X_train_pool,
                y_train_pool,
                classes,
                config.inner_val,
                split_rng,
                class_indices=train_class_indices,
            )

            best_knn_params, best_knn_score = tune_knn(
                X_inner_train,
                y_inner_train,
                X_inner_val,
                y_inner_val,
                classes,
                make_rng(seed, "sample_disjoint", "tune_knn"),
                config,
            )
            best_svm_params, best_svm_score = tune_svm_rbf(
                X_inner_train,
                y_inner_train,
                X_inner_val,
                y_inner_val,
                classes,
                make_rng(seed, "sample_disjoint", "tune_svm"),
                config,
            )
            best_logreg_params, best_logreg_score = tune_logreg(
                X_inner_train,
                y_inner_train,
                X_inner_val,
                y_inner_val,
                classes,
                make_rng(seed, "sample_disjoint", "tune_logreg"),
                config,
            )
            best_mlp_params, best_mlp_score = tune_mlp(
                X_inner_train,
                y_inner_train,
                X_inner_val,
                y_inner_val,
                classes,
                make_rng(seed, "sample_disjoint", "tune_mlp"),
                config,
            )
            best_rf_params, best_rf_score = tune_rf(
                X_inner_train,
                y_inner_train,
                X_inner_val,
                y_inner_val,
                classes,
                make_rng(seed, "sample_disjoint", "tune_rf"),
                config,
            )

            best_proto_params, best_proto_score = tune_proto_lr(
                X_inner_train,
                y_inner_train,
                X_inner_val,
                y_inner_val,
                classes,
                make_rng(seed, "sample_disjoint", "tune_proto"),
                X_train_pool.shape[1],
                config,
            )
            proto_tune_encoder = make_encoder(X_train_pool.shape[1], n_samples=len(X_inner_train))
            proto_head_train_specs = build_episode_specs(
                y_inner_train,
                classes,
                config.sample_n_way,
                config.k_shot,
                config.q_query,
                config.meta_tune_train_episodes,
                make_rng(seed, "sample_disjoint", "proto_head_train_specs"),
            )
            train_meta_encoder_proto(
                proto_tune_encoder,
                X_inner_train,
                y_inner_train,
                classes,
                make_rng(seed, "sample_disjoint", "proto_head_train"),
                config,
                learning_rate=best_proto_params["learning_rate"],
                train_episodes=config.meta_tune_train_episodes,
                episode_specs=proto_head_train_specs,
            )
            (
                best_cosine_params,
                best_cosine_score,
                best_cov_params,
                best_cov_score,
                best_tpn_params,
                best_tpn_score,
            ) = tune_proto_based_heads(
                proto_tune_encoder,
                X_inner_train,
                y_inner_train,
                X_inner_val,
                y_inner_val,
                classes,
                make_rng(seed, "sample_disjoint", "tune_proto_heads"),
                config,
            )

            best_matching_params, best_matching_score = tune_matching_lr(
                X_inner_train,
                y_inner_train,
                X_inner_val,
                y_inner_val,
                classes,
                make_rng(seed, "sample_disjoint", "tune_matching"),
                X_train_pool.shape[1],
                config,
            )
            best_relation_params, best_relation_score = tune_relation_lr(
                X_inner_train,
                y_inner_train,
                X_inner_val,
                y_inner_val,
                classes,
                make_rng(seed, "sample_disjoint", "tune_relation"),
                X_train_pool.shape[1],
                config,
            )
            best_siamese_params, best_siamese_score = tune_siamese_hparams(
                X_inner_train,
                y_inner_train,
                X_inner_val,
                y_inner_val,
                classes,
                make_rng(seed, "sample_disjoint", "tune_siamese"),
                X_train_pool.shape[1],
                config,
            )
            del proto_tune_encoder
            clear_runtime_memory()
        except Exception as exc:
            if not is_data_scarcity_error(exc) and not isinstance(exc, PermissionError):
                raise
            reason = "datos escasos" if is_data_scarcity_error(exc) else "entorno/restricciones"
            tuning_status = "fallback_defaults"
            tuning_fallback_reason_code = (
                "data_scarcity_during_tuning"
                if is_data_scarcity_error(exc)
                else "environment_restriction"
            )
            tuning_fallback_reason = f"tuning interno omitido por {reason} ({exc})."
            print(f"[WARN] Seed {seed}: tuning interno omitido por {reason} ({exc}).")
            (
                best_knn_params,
                best_knn_score,
                best_svm_params,
                best_svm_score,
                best_logreg_params,
                best_logreg_score,
                best_mlp_params,
                best_mlp_score,
                best_rf_params,
                best_rf_score,
                best_proto_params,
                best_proto_score,
                best_cosine_params,
                best_cosine_score,
                best_cov_params,
                best_cov_score,
                best_tpn_params,
                best_tpn_score,
                best_matching_params,
                best_matching_score,
                best_relation_params,
                best_relation_score,
                best_siamese_params,
                best_siamese_score,
            ) = default_tuned_params()
            best_knn_score = np.nan
            best_svm_score = np.nan
            best_logreg_score = np.nan
            best_mlp_score = np.nan
            best_rf_score = np.nan
            best_proto_score = np.nan
            best_cosine_score = np.nan
            best_cov_score = np.nan
            best_tpn_score = np.nan
            best_matching_score = np.nan
            best_relation_score = np.nan
            best_siamese_score = np.nan

        protocol_model_meta = build_protocol_model_metadata(
            with_tuning=True,
            knn_params=best_knn_params,
            knn_score=best_knn_score,
            svm_params=best_svm_params,
            svm_score=best_svm_score,
            logreg_params=best_logreg_params,
            logreg_score=best_logreg_score,
            mlp_params=best_mlp_params,
            mlp_score=best_mlp_score,
            rf_params=best_rf_params,
            rf_score=best_rf_score,
            proto_params=best_proto_params,
            proto_score=best_proto_score,
            cosine_params=best_cosine_params,
            cosine_score=best_cosine_score,
            cov_params=best_cov_params,
            cov_score=best_cov_score,
            tpn_params=best_tpn_params,
            tpn_score=best_tpn_score,
            matching_params=best_matching_params,
            matching_score=best_matching_score,
            relation_params=best_relation_params,
            relation_score=best_relation_score,
            siamese_params=best_siamese_params,
            siamese_score=best_siamese_score,
            tuning_status=tuning_status,
            tuning_fallback_reason_code=tuning_fallback_reason_code,
            tuning_fallback_reason=tuning_fallback_reason,
        )

        print(
            f"[Seed {seed}] Mejor kNN: k={best_knn_params['n_neighbors']}, "
            f"weights={best_knn_params['weights']}, p={best_knn_params['p']} "
            f"(acc_tune={best_knn_score:.4f})"
        )
        print(
            f"[Seed {seed}] Mejor SVM_RBF: C={best_svm_params['C']}, gamma={best_svm_params['gamma']} "
            f"(acc_tune={best_svm_score:.4f})"
        )
        print(
            f"[Seed {seed}] Mejor LogReg: C={best_logreg_params['C']}, "
            f"penalty={best_logreg_params['penalty']} "
            f"(acc_tune={best_logreg_score:.4f})"
        )
        print(
            f"[Seed {seed}] Mejor MLP: hidden={best_mlp_params['hidden_layer_sizes']}, "
            f"alpha={best_mlp_params['alpha']:.4g}, activation={best_mlp_params['activation']} "
            f"(acc_tune={best_mlp_score:.4f})"
        )
        print(
            f"[Seed {seed}] Mejor RandomForest: n_estimators={best_rf_params['n_estimators']}, "
            f"max_depth={best_rf_params['max_depth']}, "
            f"min_samples_leaf={best_rf_params['min_samples_leaf']} "
            f"(acc_tune={best_rf_score:.4f})"
        )
        print(
            f"[Seed {seed}] Mejor ProtoNet: lr={best_proto_params['learning_rate']:.4g} "
            f"(acc_tune={best_proto_score:.4f})"
        )
        print(
            f"[Seed {seed}] Mejor MatchingNet: lr={best_matching_params['learning_rate']:.4g} "
            f"(acc_tune={best_matching_score:.4f})"
        )
        print(
            f"[Seed {seed}] Mejor RelationNet: lr={best_relation_params['learning_rate']:.4g} "
            f"(acc_tune={best_relation_score:.4f})"
        )
        print(
            f"[Seed {seed}] Mejor CosineClassifier: scale={best_cosine_params['scale']} "
            f"(acc_tune={best_cosine_score:.4f})"
        )
        print(
            f"[Seed {seed}] Mejor CovarianceNet: reg={best_cov_params['reg']:.4g} "
            f"(acc_tune={best_cov_score:.4f})"
        )
        print(
            f"[Seed {seed}] Mejor TPN: alpha={best_tpn_params['alpha']}, sigma={best_tpn_params['sigma']} "
            f"(acc_tune={best_tpn_score:.4f})"
        )
        print(
            f"[Seed {seed}] Mejor SiameseNet: hidden={best_siamese_params['hidden_units']}, "
            f"lr={best_siamese_params['learning_rate']:.4g} (acc_tune={best_siamese_score:.4f})"
        )

        # Meta-training
        proto_encoder = make_encoder(X_train_pool.shape[1], n_samples=len(X_train_pool))
        matching_encoder = make_encoder(X_train_pool.shape[1], n_samples=len(X_train_pool))
        relation_encoder = make_encoder(X_train_pool.shape[1], n_samples=len(X_train_pool))
        relation_head = build_relation_head(emb_dim=encoder_embedding_dim(relation_encoder))
        siamese_encoder = make_encoder(X_train_pool.shape[1], n_samples=len(X_train_pool))
        siamese_head = build_siamese_head(
            emb_dim=encoder_embedding_dim(siamese_encoder),
            hidden_units=best_siamese_params["hidden_units"],
        )
        print(
            f"[Seed {seed}] Encoder meta adaptativo: "
            f"{describe_encoder_spec(proto_encoder.encoder_spec)}"
        )
        proto_train_specs = build_episode_specs(
            y_train_pool,
            classes,
            config.sample_n_way,
            config.k_shot,
            config.q_query,
            config.train_episodes,
            make_rng(seed, "sample_disjoint", "train_proto_specs"),
            class_indices=train_class_indices,
        )
        matching_train_specs = build_episode_specs(
            y_train_pool,
            classes,
            config.sample_n_way,
            config.k_shot,
            config.q_query,
            config.train_episodes,
            make_rng(seed, "sample_disjoint", "train_matching_specs"),
            class_indices=train_class_indices,
        )
        relation_train_specs = build_episode_specs(
            y_train_pool,
            classes,
            config.sample_n_way,
            config.k_shot,
            config.q_query,
            config.train_episodes,
            make_rng(seed, "sample_disjoint", "train_relation_specs"),
            class_indices=train_class_indices,
        )
        siamese_train_specs = build_episode_specs(
            y_train_pool,
            classes,
            config.sample_n_way,
            config.k_shot,
            config.q_query,
            config.train_episodes,
            make_rng(seed, "sample_disjoint", "train_siamese_specs"),
            class_indices=train_class_indices,
        )

        train_meta_encoder_proto(
            proto_encoder,
            X_train_pool,
            y_train_pool,
            classes,
            make_rng(seed, "sample_disjoint", "train_proto"),
            config,
            learning_rate=best_proto_params["learning_rate"],
            episode_specs=proto_train_specs,
        )
        train_meta_encoder_matching(
            matching_encoder,
            X_train_pool,
            y_train_pool,
            classes,
            make_rng(seed, "sample_disjoint", "train_matching"),
            config,
            learning_rate=best_matching_params["learning_rate"],
            episode_specs=matching_train_specs,
        )
        train_meta_relation(
            relation_encoder,
            relation_head,
            X_train_pool,
            y_train_pool,
            classes,
            make_rng(seed, "sample_disjoint", "train_relation"),
            config,
            learning_rate=best_relation_params["learning_rate"],
            episode_specs=relation_train_specs,
        )
        train_meta_siamese(
            siamese_encoder,
            siamese_head,
            X_train_pool,
            y_train_pool,
            classes,
            make_rng(seed, "sample_disjoint", "train_siamese"),
            config,
            learning_rate=best_siamese_params["learning_rate"],
            episode_specs=siamese_train_specs,
        )

        # Evaluacion episodica en test
        eval_episode_specs = build_episode_specs(
            y_test_pool,
            classes,
            config.sample_n_way,
            config.k_shot,
            config.q_query,
            config.eval_episodes,
            make_rng(seed, "sample_disjoint", "eval_specs"),
            class_indices=test_class_indices,
        )
        for ep, episode_spec in enumerate(eval_episode_specs, start=1):
            s_x, s_y, q_x, q_y = materialize_episode_specs(X_test_pool, y_test_pool, episode_spec)

            preds = {}
            if np.unique(s_y).size < 2:
                const_pred = constant_prediction(s_y, len(q_y))
                for name in sample_disjoint_model_names:
                    preds[name] = const_pred
            else:
                add_classical_predictions(
                    preds,
                    "",
                    s_x,
                    s_y,
                    q_x,
                    seed,
                    best_knn_params,
                    best_svm_params,
                    best_logreg_params,
                    best_mlp_params,
                    best_rf_params,
                )
                support_emb_proto, query_emb_proto = compute_episode_embeddings(proto_encoder, s_x, q_x)
                add_classical_predictions(
                    preds,
                    "_ProtoEmb",
                    support_emb_proto,
                    s_y,
                    query_emb_proto,
                    seed,
                    best_knn_params,
                    best_svm_params,
                    best_logreg_params,
                    best_mlp_params,
                    best_rf_params,
                )
                preds["ProtoNet"] = predict_protonet(
                    proto_encoder, s_x, s_y, q_x, support_emb=support_emb_proto, query_emb=query_emb_proto
                )
                preds["MatchingNet"] = predict_matching(matching_encoder, s_x, s_y, q_x)
                preds["RelationNet"] = predict_relation(relation_encoder, relation_head, s_x, s_y, q_x)
                preds["CosineClassifier"] = predict_cosine(
                    proto_encoder,
                    s_x,
                    s_y,
                    q_x,
                    scale=best_cosine_params["scale"],
                    support_emb=support_emb_proto,
                    query_emb=query_emb_proto,
                )
                preds["CovarianceNet"] = predict_covariance(
                    proto_encoder,
                    s_x,
                    s_y,
                    q_x,
                    reg=best_cov_params["reg"],
                    support_emb=support_emb_proto,
                    query_emb=query_emb_proto,
                )
                preds["TransductivePropagationNet"] = predict_tpn(
                    proto_encoder,
                    s_x,
                    s_y,
                    q_x,
                    alpha=best_tpn_params["alpha"],
                    sigma=best_tpn_params["sigma"],
                    support_emb=support_emb_proto,
                    query_emb=query_emb_proto,
                )
                preds["SiameseNet"] = predict_siamese(siamese_encoder, siamese_head, s_x, s_y, q_x)

            for name, pred in preds.items():
                record_episode_metrics(seed_metrics, name, q_y, pred)

            if ep == 1:
                visualize_episode_embeddings(seed, ep, proto_encoder, s_x, s_y, q_x, q_y, dataset_name, config)
        finalize_seed_metrics(metrics, seed_metrics, seed, seed_model_meta=protocol_model_meta)
        clear_runtime_memory()
    return metrics


def best_model_by_accuracy(metrics, model_names):
    best_name = None
    best_acc = -1.0
    for name in model_names:
        mean_acc, _, _, _ = ci95(metric_summary_values(metrics, name, "acc"))
        if np.isnan(mean_acc):
            continue
        if mean_acc > best_acc:
            best_name = name
            best_acc = mean_acc
    if best_name is None:
        return "N/A", float("nan")
    return best_name, best_acc


def run_single_dataset(
    csv_path=None,
    target_column=None,
    sep=",",
    fewshot_config=None,
    quick=False,
    explicit_config=False,
):
    config = reset_fewshot_config()
    config = apply_fewshot_config_override(config, fewshot_config)
    if quick:
        config = replace(config, quick_mode=True)

    X, y, dataset_name = load_dataset(
        csv_path=csv_path,
        target_column=target_column,
        sep=sep,
    )
    config = replace(config, explicit_config=bool(explicit_config))
    config = adapt_fewshot_config(y, config)
    class_counts = np.asarray([np.sum(y == cls) for cls in np.unique(y)], dtype=np.int32)
    dataset_path = str(resolve_input_path(csv_path).resolve()) if csv_path is not None else "builtin:iris"
    config_label = config_to_label(config)
    dataset_label = f"{dataset_name} | {config_label}"
    dataset_meta = {
        "dataset": dataset_name,
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "dataset_label": dataset_label,
        "seed_count": len(SEEDS),
        "dataset_regime": config.dataset_regime,
        "DATASET_REGIME": config.dataset_regime,
        "REQUESTED_N_WAY": int(config.requested_n_way),
        "REQUESTED_K_SHOT": int(config.requested_k_shot),
        "REQUESTED_Q_QUERY": int(config.requested_q_query),
        "n_classes": int(len(np.unique(y))),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "min_class_count": int(np.min(class_counts)),
        "max_class_count": int(np.max(class_counts)),
        "inner_val_per_class": int(config.inner_val),
        "train_episodes": int(config.train_episodes),
        "eval_episodes": int(config.eval_episodes),
        "TRAIN_EPISODES": int(config.train_episodes),
        "EVAL_EPISODES": int(config.eval_episodes),
        "TUNE_EPISODES": int(config.tune_episodes),
        "CLASSICAL_TUNE_EPISODES": int(config.classical_tune_episodes),
        "META_TUNE_EVAL_EPISODES": int(config.meta_tune_eval_episodes),
        "META_TUNE_TRAIN_EPISODES": int(config.meta_tune_train_episodes),
        "SIAMESE_TUNE_TRAIN_EPISODES": int(config.siamese_tune_train_episodes),
        "TEST_SIZE": float(config.test_size),
        "CLASS_DISJOINT_N_WAY": int(config.class_disjoint_n_way),
        "CLASS_DISJOINT_TEST_CLASSES": int(config.class_disjoint_test_classes),
        "CLASS_DISJOINT_STATUS": config.class_disjoint_status,
        "CLASS_DISJOINT_STATUS_CODE": config.class_disjoint_status_code,
        "allow_replacement": bool(config.allow_replacement),
        "quick_mode": bool(config.quick_mode),
        "explicit_config": bool(config.explicit_config),
        "adaptation_mode": infer_adaptation_mode(config),
        "n_episodes_eval": int(config.eval_episodes),
        "run_config_json": fewshot_config_to_json(config),
    }
    class_disjoint_meta = {
        **dataset_meta,
        "protocol": "class_disjoint",
        "with_tuning": False,
        "n_way": int(config.class_disjoint_n_way),
        "k_shot": int(config.k_shot),
        "q_query": int(config.q_query),
        "N_WAY": int(config.class_disjoint_n_way),
        "K_SHOT": int(config.k_shot),
        "Q_QUERY": int(config.q_query),
        "fewshot_config": config_to_label(config.as_label_dict(config.class_disjoint_n_way))
        if config.class_disjoint_n_way >= 1
        else config_label,
        "test_size": float(config.test_size),
        "skip_reason": config.class_disjoint_status,
        "skip_reason_code": config.class_disjoint_status_code,
    }
    sample_disjoint_meta = {
        **dataset_meta,
        "protocol": "sample_disjoint",
        "with_tuning": True,
        "n_way": int(config.sample_n_way),
        "k_shot": int(config.k_shot),
        "q_query": int(config.q_query),
        "N_WAY": int(config.sample_n_way),
        "K_SHOT": int(config.k_shot),
        "Q_QUERY": int(config.q_query),
        "fewshot_config": config_label,
        "test_size": float(config.test_size),
        "skip_reason": "",
        "skip_reason_code": "",
    }

    print(f"\n\n########## DATASET: {dataset_name} ##########")
    print(f"[config] {config_label} | regimen={config.dataset_regime} | seeds={SEEDS}")
    class_disjoint_metrics = run_class_disjoint_no_tuning(config, X, y, dataset_name)
    sample_disjoint_metrics = run_sample_disjoint_tuning(config, X, y, dataset_name)

    print_results(
        "Resultados Globales Class-Disjoint (sin tuning, few-shot clasico)",
        class_disjoint_metrics,
        class_disjoint_model_names,
    )
    print_results(
        "Resultados Globales Sample-Disjoint (con tuning)",
        sample_disjoint_metrics,
        sample_disjoint_model_names,
    )

    class_best_name, class_best_acc = best_model_by_accuracy(
        class_disjoint_metrics, class_disjoint_model_names
    )
    sample_best_name, sample_best_acc = best_model_by_accuracy(
        sample_disjoint_metrics, sample_disjoint_model_names
    )
    if SAVE_EMBEDDINGS:
        print("\nEmbeddings guardados: embeddings/<dataset>_episode_seed<seed>.png")
    print(
        f"Resumen dataset {dataset_label}: "
        f"mejor class-disjoint={class_best_name} ({class_best_acc:.4f}), "
        f"mejor sample-disjoint={sample_best_name} ({sample_best_acc:.4f})"
    )
    metric_rows = []
    metric_rows.extend(
        metrics_to_rows(
            dataset_label,
            "class_disjoint",
            class_disjoint_metrics,
            class_disjoint_model_names,
            class_disjoint_meta,
        )
    )
    metric_rows.extend(
        metrics_to_rows(
            dataset_label,
            "sample_disjoint",
            sample_disjoint_metrics,
            sample_disjoint_model_names,
            sample_disjoint_meta,
        )
    )
    clear_runtime_memory()
    return {
        "dataset": dataset_label,
        "class_best_name": class_best_name,
        "class_best_acc": class_best_acc,
        "sample_best_name": sample_best_name,
        "sample_best_acc": sample_best_acc,
        "metric_rows": metric_rows,
        "dataset_meta": dataset_meta,
    }


def discover_reduced_csvs(root_dir):
    root_path = resolve_input_path(root_dir)
    if not root_path.exists():
        raise ValueError(f"La carpeta indicada no existe: {root_dir} -> {root_path}")
    csv_paths = sorted(root_path.rglob("*_reduced.csv"))
    if not csv_paths:
        raise ValueError(f"No se encontraron archivos '*_reduced.csv' en: {root_dir}")
    return [str(p) for p in csv_paths]


def discover_reduced_csv_groups(root_dir):
    root_path = resolve_input_path(root_dir)
    if not root_path.exists():
        raise ValueError(f"La carpeta indicada no existe: {root_dir} -> {root_path}")

    subdirs = sorted([p for p in root_path.iterdir() if p.is_dir()])
    if not subdirs:
        csv_paths = sorted(root_path.rglob("*_reduced.csv"))
        if not csv_paths:
            raise ValueError(f"No se encontraron archivos '*_reduced.csv' en: {root_dir}")
        return [{"name": root_path.name, "dir": root_path, "csv_paths": [str(p) for p in csv_paths]}]

    groups = []
    for subdir in subdirs:
        csv_paths = sorted(subdir.rglob("*_reduced.csv"))
        if not csv_paths:
            continue
        groups.append({"name": subdir.name, "dir": subdir, "csv_paths": [str(p) for p in csv_paths]})

    if not groups:
        raise ValueError(f"No se encontraron archivos '*_reduced.csv' dentro de subcarpetas en: {root_dir}")
    return groups


def group_results_csv_path(base_results_csv, group_dir):
    base_path = Path(base_results_csv)
    return str(Path(group_dir) / base_path.name)


def print_batch_summary(summaries):
    print("\n\n================ RESUMEN GLOBAL POR DATASET ================")
    for item in summaries:
        seed_label = f"{item.get('dataset_meta', {}).get('seed_count', len(SEEDS))} seeds"
        print(
            f"{item['dataset']:45s} | "
            f"Class-Disjoint: {item['class_best_name']:20s} ({item['class_best_acc']:.4f}) | "
            f"Sample-Disjoint: {item['sample_best_name']:25s} ({item['sample_best_acc']:.4f}) | "
            f"{seed_label}"
        )


def print_skipped_summary(skipped):
    if not skipped:
        return
    print("\n\n================ DATASETS OMITIDOS ================")
    for item in skipped:
        print(f"{item['dataset']:45s} | Motivo: {item['reason']}")


def metrics_to_rows(dataset, split_name, metrics, model_names, dataset_meta):
    rows = []
    for name in model_names:
        seed_runs = metrics[name].get("seed_runs", [])
        effective_seed_count = len([sr for sr in seed_runs if sr.get("status") == "ok"])
        for seed_run in seed_runs:
            seed_skip_reason = seed_run.get("skip_reason") or dataset_meta.get(
                "skip_reason",
                "Modelo sin observaciones: no se generaron episodios evaluables para este caso.",
            )
            seed_skip_reason_code = seed_run.get("skip_reason_code") or dataset_meta.get(
                "skip_reason_code",
                "",
            )
            seed_tuning_status = seed_run.get("tuning_status", "")
            seed_tuning_fallback_reason_code = seed_run.get("tuning_fallback_reason_code", "")
            seed_tuning_fallback_reason = seed_run.get("tuning_fallback_reason", "")
            rows.append(
                {
                    "status": seed_run["status"],
                    "dataset": dataset_meta["dataset"],
                    "dataset_name": dataset_meta["dataset_name"],
                    "dataset_path": dataset_meta["dataset_path"],
                    "dataset_label": dataset_meta["dataset_label"],
                    "protocol": dataset_meta.get("protocol", split_name),
                    "split": split_name,
                    "model": name,
                    "model_name": name,
                    "seed": seed_run["seed"],
                    "acc_mean": seed_run["acc_mean"],
                    "acc_std": np.nan,
                    "acc_ci_low": np.nan,
                    "acc_ci_high": np.nan,
                    "acc_ci95_low": np.nan,
                    "acc_ci95_high": np.nan,
                    "f1_mean": seed_run["f1_mean"],
                    "f1_std": np.nan,
                    "f1_ci_low": np.nan,
                    "f1_ci_high": np.nan,
                    "f1_ci95_low": np.nan,
                    "f1_ci95_high": np.nan,
                    "n_way": dataset_meta["n_way"],
                    "k_shot": dataset_meta["k_shot"],
                    "q_query": dataset_meta["q_query"],
                    "inner_val_per_class": dataset_meta["inner_val_per_class"],
                    "REQUESTED_N_WAY": dataset_meta["REQUESTED_N_WAY"],
                    "REQUESTED_K_SHOT": dataset_meta["REQUESTED_K_SHOT"],
                    "REQUESTED_Q_QUERY": dataset_meta["REQUESTED_Q_QUERY"],
                    "N_WAY": dataset_meta["N_WAY"],
                    "K_SHOT": dataset_meta["K_SHOT"],
                    "Q_QUERY": dataset_meta["Q_QUERY"],
                    "TEST_SIZE": dataset_meta["TEST_SIZE"],
                    "TRAIN_EPISODES": dataset_meta["TRAIN_EPISODES"],
                    "EVAL_EPISODES": dataset_meta["EVAL_EPISODES"],
                    "TUNE_EPISODES": dataset_meta["TUNE_EPISODES"],
                    "CLASSICAL_TUNE_EPISODES": dataset_meta["CLASSICAL_TUNE_EPISODES"],
                    "META_TUNE_EVAL_EPISODES": dataset_meta["META_TUNE_EVAL_EPISODES"],
                    "META_TUNE_TRAIN_EPISODES": dataset_meta["META_TUNE_TRAIN_EPISODES"],
                    "SIAMESE_TUNE_TRAIN_EPISODES": dataset_meta["SIAMESE_TUNE_TRAIN_EPISODES"],
                    "CLASS_DISJOINT_N_WAY": dataset_meta["CLASS_DISJOINT_N_WAY"],
                    "CLASS_DISJOINT_TEST_CLASSES": dataset_meta["CLASS_DISJOINT_TEST_CLASSES"],
                    "DATASET_REGIME": dataset_meta["DATASET_REGIME"],
                    "CLASS_DISJOINT_STATUS": dataset_meta["CLASS_DISJOINT_STATUS"],
                    "fewshot_config": dataset_meta["fewshot_config"],
                    "seed_count": 1,
                    "n_classes": dataset_meta["n_classes"],
                    "n_samples": dataset_meta["n_samples"],
                    "n_features": dataset_meta["n_features"],
                    "min_class_count": dataset_meta["min_class_count"],
                    "max_class_count": dataset_meta["max_class_count"],
                    "n_episodes_eval": dataset_meta["n_episodes_eval"],
                    "test_size": dataset_meta["test_size"],
                    "allow_replacement": dataset_meta["allow_replacement"],
                    "with_tuning": seed_run.get("with_tuning", dataset_meta.get("with_tuning", np.nan)),
                    "quick_mode": dataset_meta["quick_mode"],
                    "explicit_config": dataset_meta["explicit_config"],
                    "adaptation_mode": dataset_meta["adaptation_mode"],
                    "best_params_json": seed_run.get("best_params_json", ""),
                    "tuning_score": seed_run.get("tuning_score", np.nan),
                    "run_config_json": dataset_meta["run_config_json"],
                    "aggregation_level": "run",
                    "n_observations": seed_run["n_observations"],
                    "tuning_status": seed_tuning_status,
                    "tuning_fallback_reason_code": seed_tuning_fallback_reason_code,
                    "tuning_fallback_reason": seed_tuning_fallback_reason,
                    "skip_reason_code": ""
                    if seed_run["status"] == "ok"
                    else seed_skip_reason_code,
                    "reason": ""
                    if seed_run["status"] == "ok"
                    else seed_skip_reason,
                }
            )
        acc_values = metric_summary_values(metrics, name, "acc")
        f1_values = metric_summary_values(metrics, name, "f1")
        model_has_observations = len(acc_values) > 0 or len(f1_values) > 0
        acc_mean, acc_std, acc_lo, acc_hi = ci95(acc_values)
        f1_mean, f1_std, f1_lo, f1_hi = ci95(f1_values)
        tuning_score = aggregate_tuning_score(seed_runs)
        best_params_json = aggregate_best_params(seed_runs)
        summary_skip_reason = aggregate_skip_reason_field(
            seed_runs,
            "skip_reason",
            dataset_meta.get(
                "skip_reason",
                "Modelo sin observaciones: no se generaron episodios evaluables para este caso.",
            ),
        )
        summary_skip_reason_code = aggregate_skip_reason_field(
            seed_runs,
            "skip_reason_code",
            dataset_meta.get("skip_reason_code", ""),
        )
        summary_tuning_status = aggregate_skip_reason_field(seed_runs, "tuning_status", "")
        summary_tuning_fallback_reason_code = aggregate_skip_reason_field(
            seed_runs,
            "tuning_fallback_reason_code",
            "",
        )
        summary_tuning_fallback_reason = aggregate_skip_reason_field(
            seed_runs,
            "tuning_fallback_reason",
            "",
        )
        rows.append(
            {
                "status": "ok" if model_has_observations else "skipped",
                "dataset": dataset_meta["dataset"],
                "dataset_name": dataset_meta["dataset_name"],
                "dataset_path": dataset_meta["dataset_path"],
                "dataset_label": dataset,
                "protocol": dataset_meta.get("protocol", split_name),
                "split": split_name,
                "model": name,
                "model_name": name,
                "seed": np.nan,
                "acc_mean": acc_mean,
                "acc_std": acc_std,
                "acc_ci_low": acc_lo,
                "acc_ci_high": acc_hi,
                "acc_ci95_low": acc_lo,
                "acc_ci95_high": acc_hi,
                "f1_mean": f1_mean,
                "f1_std": f1_std,
                "f1_ci_low": f1_lo,
                "f1_ci_high": f1_hi,
                "f1_ci95_low": f1_lo,
                "f1_ci95_high": f1_hi,
                "n_way": dataset_meta["n_way"],
                "k_shot": dataset_meta["k_shot"],
                "q_query": dataset_meta["q_query"],
                "inner_val_per_class": dataset_meta["inner_val_per_class"],
                "REQUESTED_N_WAY": dataset_meta["REQUESTED_N_WAY"],
                "REQUESTED_K_SHOT": dataset_meta["REQUESTED_K_SHOT"],
                "REQUESTED_Q_QUERY": dataset_meta["REQUESTED_Q_QUERY"],
                "N_WAY": dataset_meta["N_WAY"],
                "K_SHOT": dataset_meta["K_SHOT"],
                "Q_QUERY": dataset_meta["Q_QUERY"],
                "TEST_SIZE": dataset_meta["TEST_SIZE"],
                "TRAIN_EPISODES": dataset_meta["TRAIN_EPISODES"],
                "EVAL_EPISODES": dataset_meta["EVAL_EPISODES"],
                "TUNE_EPISODES": dataset_meta["TUNE_EPISODES"],
                "CLASSICAL_TUNE_EPISODES": dataset_meta["CLASSICAL_TUNE_EPISODES"],
                "META_TUNE_EVAL_EPISODES": dataset_meta["META_TUNE_EVAL_EPISODES"],
                "META_TUNE_TRAIN_EPISODES": dataset_meta["META_TUNE_TRAIN_EPISODES"],
                "SIAMESE_TUNE_TRAIN_EPISODES": dataset_meta["SIAMESE_TUNE_TRAIN_EPISODES"],
                "CLASS_DISJOINT_N_WAY": dataset_meta["CLASS_DISJOINT_N_WAY"],
                "CLASS_DISJOINT_TEST_CLASSES": dataset_meta["CLASS_DISJOINT_TEST_CLASSES"],
                "DATASET_REGIME": dataset_meta["DATASET_REGIME"],
                "CLASS_DISJOINT_STATUS": dataset_meta["CLASS_DISJOINT_STATUS"],
                "fewshot_config": dataset_meta["fewshot_config"],
                "seed_count": effective_seed_count if effective_seed_count else dataset_meta["seed_count"],
                "n_classes": dataset_meta["n_classes"],
                "n_samples": dataset_meta["n_samples"],
                "n_features": dataset_meta["n_features"],
                "min_class_count": dataset_meta["min_class_count"],
                "max_class_count": dataset_meta["max_class_count"],
                "n_episodes_eval": dataset_meta["n_episodes_eval"],
                "test_size": dataset_meta["test_size"],
                "allow_replacement": dataset_meta["allow_replacement"],
                "with_tuning": dataset_meta.get("with_tuning", np.nan),
                "quick_mode": dataset_meta["quick_mode"],
                "explicit_config": dataset_meta["explicit_config"],
                "adaptation_mode": dataset_meta["adaptation_mode"],
                "best_params_json": best_params_json,
                "tuning_score": tuning_score,
                "run_config_json": dataset_meta["run_config_json"],
                "aggregation_level": "seed_summary"
                if len(acc_values) == len(metrics[name].get("acc_by_seed", [])) and len(acc_values) > 0
                else "episode_summary",
                "n_observations": len(acc_values),
                "tuning_status": summary_tuning_status,
                "tuning_fallback_reason_code": summary_tuning_fallback_reason_code,
                "tuning_fallback_reason": summary_tuning_fallback_reason,
                "skip_reason_code": "" if model_has_observations else summary_skip_reason_code,
                "reason": ""
                if model_has_observations
                else summary_skip_reason,
            }
        )
    return rows


def is_dataset_summary_row(row):
    aggregation_level = row.get("aggregation_level", "")
    if aggregation_level not in {"seed_summary", "episode_summary"}:
        return False
    if row.get("status") != "ok":
        return False
    if pd.isna(row.get("seed")) is False:
        return False
    return bool(row.get("model"))


def benchmark_group_key(row):
    return (
        row.get("protocol", ""),
        row.get("split", ""),
        row.get("fewshot_config", ""),
        bool(row.get("quick_mode", False)),
        bool(row.get("explicit_config", False)),
        row.get("with_tuning", np.nan),
    )


def benchmark_group_label(group_key):
    protocol, split_name, fewshot_config, quick_mode, explicit_config, with_tuning = group_key
    tuning_label = "with_tuning" if bool(with_tuning) else "no_tuning"
    quick_label = "quick" if quick_mode else "full"
    explicit_label = "explicit" if explicit_config else "auto"
    return (
        f"benchmark::{protocol}::{split_name}::{fewshot_config}::"
        f"{tuning_label}::{quick_label}::{explicit_label}"
    )


def build_benchmark_base_row(first_row, group_key):
    protocol, split_name, fewshot_config, quick_mode, explicit_config, with_tuning = group_key
    benchmark_label = benchmark_group_label(group_key)
    return {
        "status": "ok",
        "dataset": "__benchmark__",
        "dataset_name": "__benchmark__",
        "dataset_path": "",
        "dataset_label": benchmark_label,
        "protocol": protocol,
        "split": split_name,
        "model": "",
        "model_name": "",
        "seed": np.nan,
        "acc_mean": np.nan,
        "acc_std": np.nan,
        "acc_ci_low": np.nan,
        "acc_ci_high": np.nan,
        "acc_ci95_low": np.nan,
        "acc_ci95_high": np.nan,
        "f1_mean": np.nan,
        "f1_std": np.nan,
        "f1_ci_low": np.nan,
        "f1_ci_high": np.nan,
        "f1_ci95_low": np.nan,
        "f1_ci95_high": np.nan,
        "n_way": np.nan,
        "k_shot": np.nan,
        "q_query": np.nan,
        "inner_val_per_class": np.nan,
        "REQUESTED_N_WAY": np.nan,
        "REQUESTED_K_SHOT": np.nan,
        "REQUESTED_Q_QUERY": np.nan,
        "N_WAY": np.nan,
        "K_SHOT": np.nan,
        "Q_QUERY": np.nan,
        "TEST_SIZE": np.nan,
        "TRAIN_EPISODES": np.nan,
        "EVAL_EPISODES": np.nan,
        "TUNE_EPISODES": np.nan,
        "CLASSICAL_TUNE_EPISODES": np.nan,
        "META_TUNE_EVAL_EPISODES": np.nan,
        "META_TUNE_TRAIN_EPISODES": np.nan,
        "SIAMESE_TUNE_TRAIN_EPISODES": np.nan,
        "CLASS_DISJOINT_N_WAY": np.nan,
        "CLASS_DISJOINT_TEST_CLASSES": np.nan,
        "DATASET_REGIME": "",
        "CLASS_DISJOINT_STATUS": "",
        "fewshot_config": fewshot_config,
        "seed_count": np.nan,
        "n_classes": np.nan,
        "n_samples": np.nan,
        "n_features": np.nan,
        "min_class_count": np.nan,
        "max_class_count": np.nan,
        "n_episodes_eval": np.nan,
        "test_size": np.nan,
        "allow_replacement": np.nan,
        "with_tuning": with_tuning,
        "quick_mode": quick_mode,
        "explicit_config": explicit_config,
        "adaptation_mode": first_row.get("adaptation_mode", ""),
        "best_params_json": "",
        "tuning_score": np.nan,
        "run_config_json": "",
        "aggregation_level": "",
        "n_observations": np.nan,
        "tuning_status": "",
        "tuning_fallback_reason_code": "",
        "tuning_fallback_reason": "",
        "skip_reason_code": "",
        "reason": "",
        "benchmark_avg_rank": np.nan,
        "benchmark_rank_std": np.nan,
        "benchmark_dataset_count": np.nan,
        "benchmark_metric": "acc_mean",
        "benchmark_reference_model": "",
        "benchmark_wilcoxon_stat": np.nan,
        "benchmark_wilcoxon_pvalue": np.nan,
        "benchmark_wilcoxon_pvalue_bonferroni": np.nan,
        "benchmark_wilcoxon_common_datasets": np.nan,
        "benchmark_wins": np.nan,
        "benchmark_losses": np.nan,
        "benchmark_ties": np.nan,
        "benchmark_mean_delta_acc": np.nan,
        "benchmark_median_delta_acc": np.nan,
    }


def build_benchmark_analysis_rows(metric_rows):
    summary_rows = [row for row in metric_rows if is_dataset_summary_row(row)]
    if not summary_rows:
        return []

    grouped_rows = {}
    for row in summary_rows:
        grouped_rows.setdefault(benchmark_group_key(row), []).append(row)

    benchmark_rows = []
    for group_key, rows in grouped_rows.items():
        group_df = pd.DataFrame(rows)
        if group_df.empty:
            continue

        rank_records = []
        for dataset_label, dataset_df in group_df.groupby("dataset_label"):
            dataset_df = dataset_df[["model", "acc_mean"]].dropna(subset=["acc_mean"])
            if dataset_df.empty:
                continue
            scores = dataset_df["acc_mean"].to_numpy(dtype=np.float64)
            ranks = rankdata(-scores, method="average")
            for model_name, score, rank_value in zip(dataset_df["model"], scores, ranks):
                rank_records.append(
                    {
                        "dataset_label": dataset_label,
                        "model": model_name,
                        "acc_mean": float(score),
                        "rank": float(rank_value),
                    }
                )

        if not rank_records:
            continue

        rank_df = pd.DataFrame(rank_records)
        rank_summary = (
            rank_df.groupby("model", as_index=False)
            .agg(
                benchmark_avg_rank=("rank", "mean"),
                benchmark_rank_std=("rank", "std"),
                benchmark_dataset_count=("dataset_label", "nunique"),
                acc_mean=("acc_mean", "mean"),
            )
            .sort_values(["benchmark_avg_rank", "acc_mean", "model"], ascending=[True, False, True])
            .reset_index(drop=True)
        )

        first_row = rows[0]
        leader_model = str(rank_summary.iloc[0]["model"])

        for _, item in rank_summary.iterrows():
            row = build_benchmark_base_row(first_row, group_key)
            row.update(
                {
                    "model": item["model"],
                    "model_name": item["model"],
                    "acc_mean": float(item["acc_mean"]),
                    "aggregation_level": "benchmark_rank",
                    "n_observations": int(item["benchmark_dataset_count"]),
                    "benchmark_avg_rank": float(item["benchmark_avg_rank"]),
                    "benchmark_rank_std": float(item["benchmark_rank_std"])
                    if not pd.isna(item["benchmark_rank_std"])
                    else 0.0,
                    "benchmark_dataset_count": int(item["benchmark_dataset_count"]),
                    "benchmark_reference_model": leader_model,
                }
            )
            benchmark_rows.append(row)

        pivot_df = (
            group_df.pivot_table(
                index="dataset_label",
                columns="model",
                values="acc_mean",
                aggfunc="first",
            )
            .sort_index()
        )
        comparison_count = max(0, len(rank_summary) - 1)
        leader_scores = pivot_df[leader_model] if leader_model in pivot_df.columns else pd.Series(dtype=float)

        for _, item in rank_summary.iloc[1:].iterrows():
            challenger_model = str(item["model"])
            compare_df = pd.concat(
                [leader_scores.rename("leader"), pivot_df[challenger_model].rename("challenger")],
                axis=1,
                join="inner",
            ).dropna()

            wins = losses = ties = 0
            stat = pvalue = mean_delta = median_delta = np.nan
            if not compare_df.empty:
                diffs = compare_df["challenger"] - compare_df["leader"]
                wins = int(np.sum(diffs > 0))
                losses = int(np.sum(diffs < 0))
                ties = int(np.sum(np.isclose(diffs, 0.0)))
                mean_delta = float(np.mean(diffs))
                median_delta = float(np.median(diffs))
                if len(compare_df) >= 2:
                    try:
                        stat, pvalue = wilcoxon(
                            compare_df["challenger"],
                            compare_df["leader"],
                            zero_method="wilcox",
                            alternative="two-sided",
                        )
                        stat = float(stat)
                        pvalue = float(pvalue)
                    except ValueError:
                        stat = 0.0
                        pvalue = 1.0 if ties == len(compare_df) else np.nan

            row = build_benchmark_base_row(first_row, group_key)
            row.update(
                {
                    "model": challenger_model,
                    "model_name": challenger_model,
                    "acc_mean": float(item["acc_mean"]),
                    "aggregation_level": "benchmark_wilcoxon_vs_leader",
                    "n_observations": int(len(compare_df)),
                    "benchmark_avg_rank": float(item["benchmark_avg_rank"]),
                    "benchmark_rank_std": float(item["benchmark_rank_std"])
                    if not pd.isna(item["benchmark_rank_std"])
                    else 0.0,
                    "benchmark_dataset_count": int(item["benchmark_dataset_count"]),
                    "benchmark_reference_model": leader_model,
                    "benchmark_wilcoxon_stat": stat,
                    "benchmark_wilcoxon_pvalue": pvalue,
                    "benchmark_wilcoxon_pvalue_bonferroni": min(1.0, pvalue * comparison_count)
                    if not pd.isna(pvalue)
                    else np.nan,
                    "benchmark_wilcoxon_common_datasets": int(len(compare_df)),
                    "benchmark_wins": wins,
                    "benchmark_losses": losses,
                    "benchmark_ties": ties,
                    "benchmark_mean_delta_acc": mean_delta,
                    "benchmark_median_delta_acc": median_delta,
                }
            )
            benchmark_rows.append(row)

    return benchmark_rows


def save_results_csv(out_csv, metric_rows, skipped):
    skipped_rows = [
        {
            "status": "skipped",
            "dataset": item["dataset"],
            "dataset_name": item.get("dataset_name", item["dataset"]),
            "dataset_path": item.get("dataset_path", item["dataset"]),
            "dataset_label": item.get("dataset_label", item["dataset"]),
            "protocol": item.get("protocol", ""),
            "split": "",
            "model": "",
            "model_name": "",
            "seed": np.nan,
            "acc_mean": np.nan,
            "acc_std": np.nan,
            "acc_ci_low": np.nan,
            "acc_ci_high": np.nan,
            "acc_ci95_low": np.nan,
            "acc_ci95_high": np.nan,
            "f1_mean": np.nan,
            "f1_std": np.nan,
            "f1_ci_low": np.nan,
            "f1_ci_high": np.nan,
            "f1_ci95_low": np.nan,
            "f1_ci95_high": np.nan,
            "n_way": np.nan,
            "k_shot": np.nan,
            "q_query": np.nan,
            "inner_val_per_class": np.nan,
            "REQUESTED_N_WAY": item.get("N_WAY", np.nan),
            "REQUESTED_K_SHOT": item.get("K_SHOT", np.nan),
            "REQUESTED_Q_QUERY": item.get("Q_QUERY", np.nan),
            "N_WAY": item.get("N_WAY", np.nan),
            "K_SHOT": item.get("K_SHOT", np.nan),
            "Q_QUERY": item.get("Q_QUERY", np.nan),
            "TEST_SIZE": np.nan,
            "TRAIN_EPISODES": np.nan,
            "EVAL_EPISODES": np.nan,
            "TUNE_EPISODES": np.nan,
            "CLASSICAL_TUNE_EPISODES": np.nan,
            "META_TUNE_EVAL_EPISODES": np.nan,
            "META_TUNE_TRAIN_EPISODES": np.nan,
            "SIAMESE_TUNE_TRAIN_EPISODES": np.nan,
            "CLASS_DISJOINT_N_WAY": np.nan,
            "CLASS_DISJOINT_TEST_CLASSES": np.nan,
            "DATASET_REGIME": "",
            "CLASS_DISJOINT_STATUS": "",
            "fewshot_config": "",
            "seed_count": np.nan,
            "n_classes": np.nan,
            "n_samples": np.nan,
            "n_features": np.nan,
            "min_class_count": np.nan,
            "max_class_count": np.nan,
            "n_episodes_eval": np.nan,
            "test_size": np.nan,
            "allow_replacement": np.nan,
            "with_tuning": item.get("with_tuning", np.nan),
            "quick_mode": item.get("quick_mode", np.nan),
            "explicit_config": item.get("explicit_config", np.nan),
            "adaptation_mode": item.get("adaptation_mode", ""),
            "best_params_json": "",
            "tuning_score": np.nan,
            "run_config_json": item.get("run_config_json", ""),
            "aggregation_level": "",
            "n_observations": np.nan,
            "tuning_status": item.get("tuning_status", ""),
            "tuning_fallback_reason_code": item.get("tuning_fallback_reason_code", ""),
            "tuning_fallback_reason": item.get("tuning_fallback_reason", ""),
            "skip_reason_code": item.get("skip_reason_code", ""),
            "reason": item["reason"],
        }
        for item in skipped
    ]
    benchmark_rows = build_benchmark_analysis_rows(metric_rows)
    rows = metric_rows + benchmark_rows + skipped_rows
    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n[OK] Resultados guardados en CSV: {out_csv}")


def evaluate_csv_group(group_name, csv_paths, args, fewshot_configs):
    summaries = []
    all_metric_rows = []
    skipped = []
    explicit_config = bool(args.fewshot_config)

    print(
        f"\n[batch] Carpeta {group_name}: se evaluaran {len(csv_paths)} archivos reducidos."
    )
    for csv_path in csv_paths:
        for config in fewshot_configs:
            try:
                result = run_single_dataset(
                    csv_path=csv_path,
                    target_column=args.target_column,
                    sep=args.sep,
                    fewshot_config=config,
                    quick=args.quick,
                    explicit_config=explicit_config,
                )
                summaries.append(result)
                all_metric_rows.extend(result["metric_rows"])
            except ValueError as exc:
                dataset_label = f"{csv_path} | {config_to_label(config)}"
                requested_config = build_requested_fewshot_config(
                    config_override=config,
                    quick=args.quick,
                    explicit_config=explicit_config,
                )
                skipped.append(
                    {
                        "dataset": csv_path,
                        "dataset_name": csv_path,
                        "dataset_path": str(resolve_input_path(csv_path).resolve()),
                        "dataset_label": dataset_label,
                        "N_WAY": config["N_WAY"],
                        "K_SHOT": config["K_SHOT"],
                        "Q_QUERY": config["Q_QUERY"],
                        "quick_mode": bool(args.quick),
                        "explicit_config": explicit_config,
                        "adaptation_mode": infer_adaptation_mode(requested_config),
                        "run_config_json": fewshot_config_to_json(requested_config),
                        "skip_reason_code": get_skip_reason_code(exc),
                        "reason": str(exc),
                    }
                )
                print(f"\n[WARN] Se omite dataset {dataset_label}: {exc}")

    return summaries, all_metric_rows, skipped


def main():
    args = parse_args()
    global SEEDS
    if args.seeds:
        SEEDS = parse_int_list(args.seeds)
    explicit_config = bool(args.fewshot_config)
    fewshot_configs = [parse_fewshot_config(text) for text in args.fewshot_config]
    if not fewshot_configs:
        fewshot_configs = [BASE_FEWSHOT_CONFIG.as_label_dict(BASE_FEWSHOT_CONFIG.requested_n_way)]
    if args.quick:
        quick_base = BASE_FEWSHOT_CONFIG.with_quick_mode()
        print(
            "[quick] Se reduciran episodios despues de adaptar cada dataset: "
            f"train={quick_base.train_episodes}, eval={quick_base.eval_episodes}, "
            f"tune={quick_base.tune_episodes}, classical_tune={quick_base.classical_tune_episodes}, "
            f"meta_eval={quick_base.meta_tune_eval_episodes}, "
            f"meta_train={quick_base.meta_tune_train_episodes}, "
            f"siamese_train={quick_base.siamese_tune_train_episodes}"
        )
    configure_runtime(
        cpu_only=args.cpu_only,
        disable_xla=args.disable_xla,
        disable_mixed_precision=args.disable_mixed_precision,
        n_jobs=args.n_jobs,
    )

    if args.csv:
        all_metric_rows = []
        for config in fewshot_configs:
            try:
                result = run_single_dataset(
                    csv_path=args.csv,
                    target_column=args.target_column,
                    sep=args.sep,
                    fewshot_config=config,
                    quick=args.quick,
                    explicit_config=explicit_config,
                )
                all_metric_rows.extend(result["metric_rows"])
            except SkipDatasetError as exc:
                requested_config = build_requested_fewshot_config(
                    config_override=config,
                    quick=args.quick,
                    explicit_config=explicit_config,
                )
                skipped_row = {
                    "dataset": args.csv,
                    "dataset_name": args.csv,
                    "dataset_path": str(resolve_input_path(args.csv).resolve()),
                    "dataset_label": f"{args.csv} | {config_to_label(config)}",
                    "N_WAY": config["N_WAY"],
                    "K_SHOT": config["K_SHOT"],
                    "Q_QUERY": config["Q_QUERY"],
                    "quick_mode": bool(args.quick),
                    "explicit_config": explicit_config,
                    "adaptation_mode": infer_adaptation_mode(requested_config),
                    "run_config_json": fewshot_config_to_json(requested_config),
                    "skip_reason_code": get_skip_reason_code(exc),
                    "reason": str(exc),
                }
                print(f"\n[WARN] Se omite dataset {skipped_row['dataset_label']}: {exc}")
                save_results_csv(args.results_csv, all_metric_rows, skipped=[skipped_row])
                return
        save_results_csv(args.results_csv, all_metric_rows, skipped=[])
        return

    include_iris = args.include_iris and not args.skip_iris
    if include_iris:
        iris_rows = []
        iris_skipped = []
        for config in fewshot_configs:
            try:
                result = run_single_dataset(
                    csv_path=None,
                    target_column=None,
                    sep=args.sep,
                    fewshot_config=config,
                    quick=args.quick,
                    explicit_config=explicit_config,
                )
                iris_rows.extend(result["metric_rows"])
            except ValueError as exc:
                requested_config = build_requested_fewshot_config(
                    config_override=config,
                    quick=args.quick,
                    explicit_config=explicit_config,
                )
                iris_skipped.append(
                    {
                        "dataset": "Iris",
                        "dataset_name": "Iris",
                        "dataset_path": "builtin:iris",
                        "dataset_label": f"Iris | {config_to_label(config)}",
                        "N_WAY": config["N_WAY"],
                        "K_SHOT": config["K_SHOT"],
                        "Q_QUERY": config["Q_QUERY"],
                        "quick_mode": bool(args.quick),
                        "explicit_config": explicit_config,
                        "adaptation_mode": infer_adaptation_mode(requested_config),
                        "run_config_json": fewshot_config_to_json(requested_config),
                        "skip_reason_code": get_skip_reason_code(exc),
                        "reason": str(exc),
                    }
                )
                print(f"\n[WARN] Se omite dataset Iris ({config_to_label(config)}): {exc}")
        iris_out_csv = str(resolve_input_path(args.reduced_root) / f"iris_{Path(args.results_csv).name}")
        save_results_csv(iris_out_csv, iris_rows, iris_skipped)

    csv_groups = discover_reduced_csv_groups(args.reduced_root)
    print(
        f"\n[batch] Se evaluaran {len(csv_groups)} carpetas desde: "
        f"{resolve_input_path(args.reduced_root).resolve()}"
    )
    for group in csv_groups:
        summaries, all_metric_rows, skipped = evaluate_csv_group(
            group["name"], group["csv_paths"], args, fewshot_configs
        )
        print_batch_summary(summaries)
        print_skipped_summary(skipped)
        save_results_csv(
            group_results_csv_path(args.results_csv, group["dir"]),
            all_metric_rows,
            skipped,
        )
        clear_runtime_memory()


if __name__ == "__main__":
    main()
