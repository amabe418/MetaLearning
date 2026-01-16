import openml
import pandas as pd
import numpy as np
from pymfe.mfe import MFE
import warnings
from sklearn.preprocessing import LabelEncoder
import argparse


def extract_bootstrap_metafeatures_streaming(dataset_id, k_samples=1000, batch_size=50, verbose= True ):
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    
    df = pd.DataFrame(X)
    df['target'] = y

    output_csv = f"metabu_augmented_task_{dataset_id}.csv"
    first_write = True
    
    print(f"Iniciando bootstrap para dataset {dataset_id}...")

    for start in range(0, k_samples, batch_size):
        batch_end = min(start + batch_size, k_samples)
        batch_meta_features = []

        for i in range(start, batch_end):
            boot_df = df.sample(n=len(df), replace=True, random_state=i)
            X_df = boot_df.drop(columns=['target']).values
            y_df = boot_df['target'].values

            if y_df.dtype.name == 'category' or y_df.dtype == 'O':
                le = LabelEncoder()
                y_df = le.fit_transform(y_df)
            else:
                y_df = y_df.astype(np.float64)

            mfe = MFE(groups=["all"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mfe.fit(X_df, y_df)
                ft_names, ft_values = mfe.extract()

            res = dict(zip(ft_names, ft_values))
            res['bootstrap_id'] = i
            res['original_task_id'] = dataset_id
            batch_meta_features.append(res)

        batch_df = pd.DataFrame(batch_meta_features)

        # Guardar al CSV (append) y liberar memoria
        batch_df.to_csv(output_csv, index=False, mode='w' if first_write else 'a', header=first_write)
        first_write = False

        if verbose:
            print(f"Guardado batch: {start+1}-{batch_end} de {k_samples}")

    print("¡Bootstrap completado y guardado!")



def main():
    
    parser = argparse.ArgumentParser(description="Extractor de vecinos MetaFeatX CLI-like")
    parser.add_argument("--task", type=int, default=3, help="Task específica")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Cantidad de archivos bootstrap")
    parser.add_argument("--batch", type=int, default=50, help="Número de batches a guardar para no saturar la memoria")
    parser.add_argument("--verbose", action="store_true", help="Mostrar progreso")
    
    args = parser.parse_args()

    extract_bootstrap_metafeatures_streaming(args.task, args.bootstrap,args.batch, args.verbose)



if __name__ == "__main__":
    main()
