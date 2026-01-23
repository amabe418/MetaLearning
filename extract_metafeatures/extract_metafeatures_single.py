"""
Script para extraer metafeatures de un dataset CSV usando pymfe.

Uso:
    python extract_metafeatures_single.py --dataset <dataset.csv> [--metafeatures-base <metafeatures_base.csv>]

Argumentos:
    --dataset            Ruta al archivo CSV del dataset (la última columna es la clase). (Obligatorio)
    --metafeatures-base  (Opcional) CSV con metafeatures de referencia para imputar valores faltantes por la media global de cada metafeature.

Salida:
    Genera un archivo CSV con los metafeatures extraídos, llamado <nombre_dataset>_metafeatures.csv en el mismo directorio donde se ejecuta el script.
    Si se pasa el archivo de metafeatures base, los valores faltantes se imputan con la media de esa columna en el total de metafeatures.
"""
import os
import pandas as pd
import argparse
from pymfe.mfe import MFE

parser = argparse.ArgumentParser(description="Extrae metafeatures de un dataset CSV usando pymfe.")
parser.add_argument('--dataset', required=True, help='Ruta al archivo CSV del dataset (la última columna es la clase)')
parser.add_argument('--metafeatures-base', required=False, help='CSV con metafeatures de referencia para imputar valores faltantes por la media global')
args = parser.parse_args()

input_csv = args.dataset
base_name = os.path.splitext(os.path.basename(input_csv))[0]
output_csv = f"{base_name}_metafeatures.csv"
metafeatures_base = pd.read_csv(args.metafeatures_base) if args.metafeatures_base else None

try:
    df = pd.read_csv(input_csv)
    # Imputar valores faltantes en el dataset
    for col in df.columns[:-1]:  # No imputar la columna de la clase
        if df[col].dtype == object:
            df[col] = df[col].fillna('missing')
        else:
            df[col] = df[col].fillna(df[col].mean())
    # Separar X e y (asume que la última columna es la clase)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    mfe = MFE()
    mfe.fit(X, y)
    ft_names, ft_values = mfe.extract()
    row = {'dataset': os.path.basename(input_csv)}
    row.update(dict(zip(ft_names, ft_values)))
    result_df = pd.DataFrame([row])

    # Si se pasa un csv de metafeatures base, imputar con la media global
    if metafeatures_base is not None:
        # Para cada columna numérica, imputar NaN con la media de la base
        for col in result_df.columns:
            if col != 'dataset' and col in metafeatures_base.columns:
                if result_df[col].isnull().any():
                    mean_val = metafeatures_base[col].mean()
                    result_df[col] = result_df[col].fillna(mean_val)

    result_df.to_csv(output_csv, index=False)
    print(f"Metafeatures guardados en {output_csv}")
except Exception as e:
    print(f"Error procesando {input_csv}: {e}")
    try:
        df = pd.read_csv(input_csv)
        # Imputar valores faltantes en el dataset
        for col in df.columns[:-1]:  # No imputar la columna de la clase
            if df[col].dtype == object:
                df[col] = df[col].fillna('missing')
            else:
                df[col] = df[col].fillna(df[col].mean())
        # Separar X e y (asume que la última columna es la clase)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        mfe = MFE()
        mfe.fit(X, y)
        ft_names, ft_values = mfe.extract()
        row = {'dataset': os.path.basename(input_csv)}
        row.update(dict(zip(ft_names, ft_values)))
        result_df = pd.DataFrame([row])

        # Si se pasa un csv de metafeatures base, imputar con la media global
        if metafeatures_base is not None:
            for col in result_df.columns:
                if col != 'dataset' and col in metafeatures_base.columns:
                    if result_df[col].isnull().any():
                        mean_val = metafeatures_base[col].mean()
                        result_df[col] = result_df[col].fillna(mean_val)

        result_df.to_csv(output_csv, index=False)
        print(f"Metafeatures guardados en {output_csv}")
    except Exception as e:
        print(f"Error procesando {input_csv}: {e}")
        exit(1)
