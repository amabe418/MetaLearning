# main.py

import pandas as pd
from model.metafeatx import MetaFeatX
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(basic_path, target_path):
    """
    Carga y normaliza las representaciones básicas y las representaciones objetivo.
    """
    # Cargar datasets
    basic_representations = pd.read_csv(basic_path).fillna(0)
    target_representations = pd.read_csv(target_path)
    
    # Filtrar tareas que existen en el target
    basic_representations = basic_representations[
        basic_representations.task_id.isin(target_representations.task_id.unique())
    ]
    
    # Normalizar meta-features (excepto la columna task_id)
    cols = basic_representations.columns.drop("task_id")
    scaler = StandardScaler()
    basic_representations[cols] = scaler.fit_transform(basic_representations[cols])
    
    return basic_representations, target_representations, scaler, cols

def main():
    # ========================
    # Cargar y preparar datasets
    # ========================
    basic_representations, target_representations, scaler, cols = load_and_preprocess_data(
        basic_path="data/basic_representations.csv",
        target_path="data/adaboost_target_representations.csv"
    )
    
    meta = MetaFeatX()
    meta.train(
        basic_reprs=basic_representations,
        target_reprs=target_representations,
        column_id="task_id"
    )
    
    # ========================
    # Predecir representaciones y obtener importancias
    # ========================
    predict = meta.predict(basic_reprs=basic_representations)
    importance, labels = meta.get_features_importances()
    
    print("Predicciones sobre dataset original:")
    print(predict)
    print("Importancias de features básicas:")
    print(importance)
    print("Nombres de las features básicas:")
    print(labels)
    
    # ========================
    # Nuevo dataset
    # ========================
    new_representations = pd.read_csv("data/meta_features_task2.csv").fillna(0)
    
    # Asegurarse de que tenga las mismas columnas
    new_representations = new_representations[basic_representations.columns]
    
    # Normalizar con el mismo scaler
    new_representations[cols] = scaler.transform(new_representations[cols])
    
    # Predecir nuevas representaciones 
    predict_new = meta.predict(basic_reprs=new_representations)
    print("Predicciones sobre nuevo dataset:")
    print(predict_new)

if __name__ == "__main__":
    main()
