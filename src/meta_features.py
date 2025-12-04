from amltk.metalearning import MetaFeature, DatasetStatistic, compute_metafeatures
import openml
import pandas as pd

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

def extract_meta_features(X: pd.DataFrame, y: pd.Series | pd.DataFrame) -> dict:
    """
    Extrae metacaracterísticas del dataset usando amltk.
    
    Args:
        X: DataFrame con las características
        y: Serie o DataFrame con la variable objetivo
    """

    return compute_metafeatures(X,y)


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
        dataset_id = row['dataset_id']
        dataset = openml.datasets.get_dataset(dataset_id, download_data=True)
        X, y, _, _ = dataset.get_data(
            dataset_format="dataframe",
            target=dataset.default_target_attribute,
        )
        mfs = extract_meta_features(X, y)
        mfs['dataset_id'] = dataset_id
        meta_features_list.append(mfs)

    return pd.DataFrame(meta_features_list)
    