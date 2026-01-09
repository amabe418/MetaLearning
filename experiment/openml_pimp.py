import sklearn
from sklearn.impute import SimpleImputer

from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


import openml
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.utils import check_array


def set_up_pipeline_for_task(task_id, classifier):
    """
    Configura un pipeline de ML para una tarea específica de OpenML.

    Incluye pasos de:
    - Imputación condicional de valores faltantes
    - Codificación OneHot para variables categóricas
    - Escalado estándar de características
    - Eliminación de características con baja varianza
    - Clasificador final

    Args:
        task_id (int): ID de la tarea OpenML.
        classifier (str o clase de sklearn): Tipo de clasificador a usar 
                                             (p.ej. 'random_forest', 'libsvm_svc').

    Returns:
        Pipeline: Pipeline completo listo para entrenar y evaluar.
    """

    task = openml.tasks.get_task(task_id)
    datasets = task.get_dataset()
    base, _ = modeltype_to_classifier(classifier)
    _, _, categorical_indicator, attribute_names = datasets.get_data(dataset_format="array",
                                                                     target=datasets.default_target_attribute)
    

    # Obtener índices de columnas categóricas
    cat = [index for index, value in enumerate(categorical_indicator) if value == True]
    
    # Definir pasos del pipeline
    steps = [('imputation', ConditionalImputer(strategy='median',
                                               fill_empty=0,
                                               categorical_features=cat,
                                               strategy_nominal='most_frequent')),
             ('hotencoding',
              ColumnTransformer(transformers=[('enc', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat)],
                                remainder='passthrough')),
             ('scaling', sklearn.preprocessing.StandardScaler(with_mean=False)),
             ('variencethreshold', sklearn.feature_selection.VarianceThreshold()),
             ('classifier', base)]

    # Si es RandomForest o AdaBoost no aplicamos escalado
    if isinstance(base, RandomForestClassifier) or isinstance(base, AdaBoostClassifier):
        del steps[2]

    pipe = Pipeline(steps=steps)
    return pipe


def modeltype_to_classifier(model_type, params={}):
    """
    Convierte un nombre de modelo (string) a un clasificador de scikit-learn.

    Soporta varios tipos de modelos y aplica parámetros específicos según el tipo.
    
    Parameters
    ----------
    model_type : str
        Tipo de clasificador, por ejemplo: 'adaboost', 'decision_tree', 'libsvm_svc',
        'sgd', 'random_forest'.
    params : dict, optional
        Diccionario de hiperparámetros a pasar al clasificador (default={}).

    Returns
    -------
    classifier : sklearn classifier
        El objeto clasificador inicializado con los parámetros.
    required_params : dict
        Diccionario con parámetros adicionales requeridos para compatibilidad.
    """

    # Diccionario para parámetros obligatorios según el modelo
    required_params = dict()

    if model_type == 'adaboost':
        # Separar parámetros del estimador base (DecisionTree)
        base_estimator_params = {}
        for param in list(params.keys()):
            if param.startswith('base_estimator__'):
                base_estimator_params[param[16:]] = params.pop(param)

        classifier = AdaBoostClassifier(base_estimator=sklearn.tree.DecisionTreeClassifier(**base_estimator_params),
                                        **params)
    elif model_type == 'decision_tree':
        classifier = sklearn.tree.DecisionTreeClassifier(**params)
    elif model_type == 'libsvm_svc':
        classifier = SVC(**params)
        required_params['classifier__probability'] = True
    elif model_type == 'sgd':
        classifier = sklearn.linear_model.SGDClassifier(**params)
    elif model_type == 'random_forest':
        classifier = RandomForestClassifier(**params)
    else:
        raise ValueError('Unknown classifier: %s' % model_type)
    return classifier, required_params


class ConditionalImputer(BaseEstimator, TransformerMixin):
    """
    Imputer personalizado que maneja de forma distinta columnas numéricas y categóricas.
    
    Características:
    - Para numéricas: reemplaza NaN por la media o mediana.
    - Para categóricas: reemplaza NaN por la moda.
    - Permite un valor fijo para columnas totalmente vacías (fill_empty).
    """
    def __init__(self, categorical_features=None, strategy='mean',
                 strategy_nominal='most_frequent', fill_empty=None, copy=True):
        self.categorical_features = categorical_features
        self.strategy = strategy
        self.strategy_nominal = strategy_nominal
        self.fill_empty = fill_empty
        self.copy = copy

    def fit(self, X, y=None):
        """
        Aprende los valores a imputar para cada columna.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Datos de entrada.
        y : Ignorado
            Por compatibilidad con sklearn.

        Returns
        -------
        self
        """
        X = np.array(X, copy=self.copy)
        n_features = X.shape[1]
        self.statistics_ = np.zeros(n_features, dtype=object)

        # Separar índices
        if self.categorical_features is None:
            self.categorical_features = []

        numeric_features = [i for i in range(n_features) if i not in self.categorical_features]

        # Imputer para numéricos
        if numeric_features:
            self.num_imputer_ = SimpleImputer(strategy=self.strategy)
            self.num_imputer_.fit(X[:, numeric_features])
            for idx, col in enumerate(numeric_features):
                self.statistics_[col] = self.num_imputer_.statistics_[idx]

        # Imputer para categóricos
        if self.categorical_features:
            self.cat_imputer_ = SimpleImputer(strategy=self.strategy_nominal)
            self.cat_imputer_.fit(X[:, self.categorical_features])
            for idx, col in enumerate(self.categorical_features):
                self.statistics_[col] = self.cat_imputer_.statistics_[idx]

        # Fill empty si hay columnas solo NaN
        if self.fill_empty is not None:
            for i in range(n_features):
                if self.statistics_[i] is None or (isinstance(self.statistics_[i], float) and np.isnan(self.statistics_[i])):
                    self.statistics_[i] = self.fill_empty

        return self

    def transform(self, X):
        """
        Imputa los valores aprendidos en los datos de entrada.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Datos a transformar.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            Datos con valores imputados.
        """
        X = np.array(X, copy=self.copy)
        X_transformed = X.copy()

        # Transformar columnas numéricas
        if hasattr(self, 'num_imputer_'):
            X_transformed[:, [i for i in range(X.shape[1]) if i not in self.categorical_features]] = self.num_imputer_.transform(
                X[:, [i for i in range(X.shape[1]) if i not in self.categorical_features]])

        # Transformar columnas categóricas
        if hasattr(self, 'cat_imputer_'):
            X_transformed[:, self.categorical_features] = self.cat_imputer_.transform(
                X[:, self.categorical_features])

        return X_transformed



class MemoryEfficientVarianceThreshold(VarianceThreshold):
    """
    Selector de características que elimina columnas de baja varianza.
    
    Diferencia con sklearn VarianceThreshold:
    - Calcula la varianza columna por columna.
    - No acepta matrices dispersas.
    - Evita problemas de memoria con datasets grandes.
    """

    def fit(self, X, y=None):
        """
        Aprende la varianza de cada columna y decide cuáles eliminar.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Datos de entrada.
        y : Ignorado
            Por compatibilidad con sklearn.

        Returns
        -------
        self
        """
        X = check_array(X)

        self.variances_ = []
        for i in range(X.shape[1]):
            self.variances_.append(np.var(check_array(X[:, i].reshape((-1, 1)),
                                                      dtype=np.float64)))
        self.variances_ = np.array(self.variances_)

        if np.all(self.variances_ <= self.threshold):
            msg = "No feature in X meets the variance threshold {0:.5f}"
            if X.shape[0] == 1:
                msg += " (X contains only one sample)"
            raise ValueError(msg.format(self.threshold))

        return self

