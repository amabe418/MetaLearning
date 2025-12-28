from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetaFeatureSchema:
    """
    Esquema estable: define qué meta-features vamos a usar SIEMPRE.
    Si falta una en el cálculo, se rellena con NaN y luego se imputa.
    """
    features: List[str]

    @staticmethod
    def default() -> "MetaFeatureSchema":
        # Vector meta siepre tiene estas columnas.
        return MetaFeatureSchema(features=[
            "n_rows",
            "n_cols",
            "n_classes",
            "missing_frac",
            "class_entropy",
            "mean_abs_corr",
            "mean_skewness",
            "mean_kurtosis",
        ])

    def enforce(self, raw: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k in self.features:
            v = raw.get(k, np.nan)
            try:
                out[k] = float(v) if v is not None else np.nan
            except Exception:
                out[k] = np.nan
        return out


def compute_basic_metafeatures(X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """
    Meta-features simples/estadísticas/info-theoretic básicas (Sección 2).
    Costo bajo, robustas y suficientes para arrancar.
    """
    # Asegurar DataFrame/Series
    X = X.copy()
    y = y.copy()

    n_rows = int(X.shape[0])
    n_cols = int(X.shape[1])

    # Missing fraction
    missing = X.isna().sum().sum()
    missing_frac = float(missing / max(1, (n_rows * max(1, n_cols))))

    # n_classes
    y_no_na = y.dropna()
    classes = y_no_na.unique()
    n_classes = int(len(classes)) if len(y_no_na) else 0

    # class entropy
    if n_classes <= 1:
        class_entropy = 0.0
    else:
        probs = y_no_na.value_counts(normalize=True).values
        class_entropy = float(-np.sum(probs * np.log2(np.maximum(probs, 1e-12))))

    # Solo columnas numéricas para stats/corr
    X_num = X.select_dtypes(include=[np.number]).copy()

    # mean_abs_corr
    if X_num.shape[1] >= 2:
        corr = X_num.corr(numeric_only=True).abs()
        # excluir diagonal
        vals = corr.values
        mask = ~np.eye(vals.shape[0], dtype=bool)
        mean_abs_corr = float(np.nanmean(vals[mask]))
        if np.isnan(mean_abs_corr):
            mean_abs_corr = 0.0
    else:
        mean_abs_corr = 0.0

    # skewness / kurtosis promedio
    if X_num.shape[1] >= 1:
        skew = X_num.skew(numeric_only=True)
        kurt = X_num.kurt(numeric_only=True)
        mean_skewness = float(np.nanmean(skew.values)) if len(skew) else 0.0
        mean_kurtosis = float(np.nanmean(kurt.values)) if len(kurt) else 0.0
        if np.isnan(mean_skewness): mean_skewness = 0.0
        if np.isnan(mean_kurtosis): mean_kurtosis = 0.0
    else:
        mean_skewness = 0.0
        mean_kurtosis = 0.0

    return {
        "n_rows": float(n_rows),
        "n_cols": float(n_cols),
        "n_classes": float(n_classes),
        "missing_frac": float(missing_frac),
        "class_entropy": float(class_entropy),
        "mean_abs_corr": float(mean_abs_corr),
        "mean_skewness": float(mean_skewness),
        "mean_kurtosis": float(mean_kurtosis),
    }


def as_dataframe(
    rows: List[Dict[str, Any]],
    schema: Optional[MetaFeatureSchema] = None,
    index: Optional[List[Any]] = None
) -> pd.DataFrame:
    schema = schema or MetaFeatureSchema.default()
    enforced = [schema.enforce(r) for r in rows]
    df = pd.DataFrame(enforced)
    if index is not None:
        df.index = index
    return df
