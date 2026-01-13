#!/usr/bin/env python3
import openml
import pandas as pd
from pymfe.mfe import MFE
import numpy as np
import warnings
import argparse

warnings.filterwarnings("ignore")
np.seterr(all="ignore")


class MetaFeatureExtractor:
    """Clase para extraer meta-features de tareas de OpenML."""
    
    def __init__(self, groups=["all"], drop_features=None):
        self.groups = groups
        self.drop_features = drop_features if drop_features else []

    def extract_from_task(self, task_id):
        """Extrae meta-features de un task dado su ID."""
        task = openml.tasks.get_task(task_id)
        X, y = task.get_X_and_y()
        return self._extract(X, y, task_id)

    def _extract(self, X, y, task_id):
        """Calcula meta-features y maneja arrays como t1, f1v, etc."""
        mfe = MFE(groups=self.groups)
        mfe.fit(X, y)
        ft_names, ft_values = mfe.extract()

        df = pd.DataFrame([ft_values], columns=ft_names)
        df["task_id"] = task_id

        # Solo calcular t1.mean y t1.sd
        if "t1" in df.columns:
            t1_values = df.loc[0, "t1"]
            t1_array = np.array(t1_values, dtype=float)
            df["t1.mean"] = t1_array.mean()
            df["t1.sd"] = t1_array.std()
        

        # Filtrar columnas no deseadas
        for col in self.drop_features:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        return df


def read_ids_from_file(file_path):
    """Lee IDs de task desde un archivo (uno por línea o separados por coma)."""
    with open(file_path, "r") as f:
        content = f.read()
    ids = []
    for part in content.replace(",", "\n").splitlines():
        part = part.strip()
        if part:
            ids.append(int(part))
    return ids


def main():
    parser = argparse.ArgumentParser(description="Extractor de meta-features de OpenML")
    parser.add_argument(
        "--ids_file", type=str, required=True,
        help="Archivo con task IDs (uno por línea o separados por coma)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Archivo CSV de salida"
    )
    args = parser.parse_args()

    task_ids = read_ids_from_file(args.ids_file)

    drop_cols = [
        'worst_node.mean.relative', 'best_node.mean.relative', 'linear_discr.mean.relative',
        'elite_nn.mean.relative', 'naive_bayes.mean.relative', 't1',
        'one_nn.mean.relative', 'random_node.mean.relative', 'random_node.sd.relative',
        'one_nn.sd.relative', 'worst_node.sd.relative', 'elite_nn.sd.relative',
        'best_node.sd.relative', 'naive_bayes.sd.relative', 'linear_discr.sd.relative'
    ]

    extractor = MetaFeatureExtractor(drop_features=drop_cols)
    all_dfs = []

    for tid in task_ids:
        print(f"Extrayendo task {tid}...")
        df = extractor.extract_from_task(tid)
        all_dfs.append(df)

    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.to_csv(args.output, index=False)
    print(f"Meta-features guardadas en {args.output}")


if __name__ == "__main__":
    main()
