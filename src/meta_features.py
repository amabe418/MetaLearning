from amltk.metalearning import MetaFeature, DatasetStatistic, compute_metafeatures
import openml
import pandas as pd
import numpy as np

from sklearn.cross_decomposition import CCA

dataset = openml.datasets.get_dataset(
    2,
    download_data=True,
    download_features_meta_data=False,
    download_qualities=False,
)
X, y, _, _ = dataset.get_data(
    dataset_format="dataframe",
    target=dataset.default_target_attribute,
)

class NAValues(DatasetStatistic):
    """A mask of all NA values in a dataset"""

    @classmethod
    def compute(
        cls,
        x: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        dependancy_values: dict,
    ) -> pd.DataFrame:
        return x.isna()


class PercentageNA(MetaFeature):
    """The percentage of values missing"""

    dependencies = (NAValues,)

    @classmethod
    def compute(
        cls,
        x: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        dependancy_values: dict,
    ) -> int:
        na_values = dependancy_values[NAValues]
        n_na = na_values.sum().sum()
        n_values = int(x.shape[0] * x.shape[1])
        return float(n_na / n_values)

# mfs = compute_metafeatures(X, y, features=[PercentageNA])
# print(mfs)

def compute_metafeatures_safe(
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    features: list[type[MetaFeature]] | None = None,
) -> dict:
    """
    Calcula metafeatures una por una y pone NaN si alguna falla.
    """
    features = features or list(MetaFeature.iter())
    values: dict[str, float] = {}
    for mf in features:
        name = mf.name()
        try:
            series = compute_metafeatures(X, y, features=[mf])
            values[name] = float(series.get(name, np.nan))
        except Exception:
            values[name] = np.nan
    return values


def _safe_value(name: str, func) -> dict:
    try:
        return {name: float(func())}
    except Exception:
        return {name: np.nan}


def compute_statistical_metafeatures(X: pd.DataFrame) -> dict:
    X_num = X.select_dtypes(include=[np.number])
    if X_num.shape[1] == 0:
        return {
            "num_mean_mean": np.nan,
            "num_std_mean": np.nan,
            "num_var_mean": np.nan,
            "num_min_mean": np.nan,
            "num_max_mean": np.nan,
            "num_zero_frac": np.nan,
        }

    num_mean = X_num.mean(numeric_only=True)
    num_std = X_num.std(numeric_only=True)
    num_var = X_num.var(numeric_only=True)
    num_min = X_num.min(numeric_only=True)
    num_max = X_num.max(numeric_only=True)
    zero_frac = (X_num == 0).sum().sum() / max(1, X_num.size)

    return {
        "num_mean_mean": float(np.nanmean(num_mean.values)),
        "num_std_mean": float(np.nanmean(num_std.values)),
        "num_var_mean": float(np.nanmean(num_var.values)),
        "num_min_mean": float(np.nanmean(num_min.values)),
        "num_max_mean": float(np.nanmean(num_max.values)),
        "num_zero_frac": float(zero_frac),
    }


def compute_info_theory_metafeatures(y: pd.Series | pd.DataFrame) -> dict:
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    y = y.dropna()

    if len(y) == 0:
        return {
            "class_entropy": np.nan,
            "class_entropy_norm": np.nan,
            "class_max_prob": np.nan,
            "class_min_prob": np.nan,
        }

    counts = y.value_counts()
    probs = counts / counts.sum()
    entropy = float(-np.sum(probs * np.log2(np.maximum(probs, 1e-12))))
    n_classes = len(probs)
    entropy_norm = entropy / np.log2(n_classes) if n_classes > 1 else 0.0

    return {
        "class_entropy": float(entropy),
        "class_entropy_norm": float(entropy_norm),
        "class_max_prob": float(probs.max()),
        "class_min_prob": float(probs.min()),
    }


def compute_outlier_metafeatures(X: pd.DataFrame) -> dict:
    X_num = X.select_dtypes(include=[np.number])
    if X_num.shape[1] == 0:
        return {"ratio_outlier_features": np.nan}

    def has_outlier(col: pd.Series) -> bool:
        q1 = col.quantile(0.25)
        q3 = col.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or np.isnan(iqr):
            return False
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return bool(((col < lower) | (col > upper)).any())

    outlier_flags = X_num.apply(has_outlier, axis=0)
    return {"ratio_outlier_features": float(outlier_flags.mean())}


def compute_canonical_corr_metafeatures(X: pd.DataFrame, y: pd.Series | pd.DataFrame) -> dict:
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    X_num = X.select_dtypes(include=[np.number])
    if X_num.shape[1] == 0:
        return {"canonical_corr_attr_class": np.nan}

    y = y.dropna()
    if len(y) == 0:
        return {"canonical_corr_attr_class": np.nan}

    y_dummies = pd.get_dummies(y)
    if y_dummies.shape[1] < 2:
        return {"canonical_corr_attr_class": np.nan}

    X_num = X_num.loc[y.index]
    X_num = X_num.fillna(X_num.median(numeric_only=True))

    X_vals = X_num.values
    y_vals = y_dummies.values

    def _corr() -> float:
        cca = CCA(n_components=1)
        X_c, y_c = cca.fit_transform(X_vals, y_vals)
        corr = np.corrcoef(X_c[:, 0], y_c[:, 0])[0, 1]
        return float(corr)

    return _safe_value("canonical_corr_attr_class", _corr)

def extract_meta_features(X: pd.DataFrame, y: pd.Series | pd.DataFrame) -> dict:
    """
    Extrae metacaracterísticas del dataset usando amltk.
    
    Args:
        X: DataFrame con las características
        y: Serie o DataFrame con la variable objetivo
    """

    base = compute_metafeatures_safe(X, y)
    base.update(compute_statistical_metafeatures(X))
    base.update(compute_info_theory_metafeatures(y))
    base.update(compute_outlier_metafeatures(X))
    base.update(compute_canonical_corr_metafeatures(X, y))
    return base


def extract_meta_features_batch(datasets: pd.DataFrame) -> pd.DataFrame:
    """
    Extrae metacaracterísticas para un lote de datasets.
    
    Args:
        datasets: DataFrame con información de datasets (debe incluir 'dataset_id')
    
    Returns:
        DataFrame con metacaracterísticas extraídas
    """
    meta_features_list = []

    for _, row in datasets.iterrows():
        dataset_id = int(row["dataset_id"])
        dataset = openml.datasets.get_dataset(dataset_id, download_data=True)
        X, y, _, _ = dataset.get_data(
            dataset_format="dataframe",
            target=dataset.default_target_attribute,
        )
        try:
            mfs = extract_meta_features(X, y)
        except Exception as exc:
            print(f"[WARN] metafeatures fallaron para dataset_id={dataset_id}: {exc}")
            mfs = {"error": str(exc)}
        mfs["dataset_id"] = dataset_id
        meta_features_list.append(mfs)

    return pd.DataFrame(meta_features_list)


def extract_meta_features_from_tasks(task_ids: list[int]) -> pd.DataFrame:
    """
    Extrae metacaracteristicas a partir de OpenML task_ids.

    Args:
        task_ids: lista de ids de tareas OpenML

    Returns:
        DataFrame con metacaracteristicas y dataset_id
    """
    dataset_ids = []
    for task_id in task_ids:
        task = openml.tasks.get_task(task_id)
        dataset_ids.append(int(task.dataset_id))

    datasets_df = pd.DataFrame({"dataset_id": dataset_ids})
    return extract_meta_features_batch(datasets_df)
    
