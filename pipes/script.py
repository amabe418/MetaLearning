import pandas as pd
from pathlib import Path

def encode_csv_file(input_path: Path, output_dir: Path, categorical_columns, suffix="_encoded"):
    """
    Lee un CSV, aplica one-hot encoding a las columnas categóricas y guarda
    un nuevo CSV en la carpeta de salida con el mismo nombre más un sufijo.
    """
    # Leer CSV
    df = pd.read_csv(input_path)


    df= df.drop(columns=['source_csv','Classifier'])

    # One-hot encoding (0/1)
    df_encoded = pd.get_dummies(
        df,
        columns=categorical_columns,
        dtype=int
    )

    # Crear carpeta de salida si no existe
    output_dir.mkdir(parents=True, exist_ok=True)

    # Nombre del archivo de salida
    output_path = output_dir / (input_path.stem + suffix + input_path.suffix)

    # Guardar CSV codificado
    df_encoded.to_csv(output_path, index=False)
    print(f"✅ Guardado: {output_path}")


if __name__ == "__main__":
    # Carpeta actual del script
    current_dir = Path(__file__).parent

    # Carpeta donde guardar los CSV codificados
    encoded_dir = current_dir / "encoded"

    # Columnas categóricas a codificar
    categorical_columns = [
        "Imputer Strategy",
        "Categorical Strategy",
        "Feature Selection",
        "Scaler"
    ]

    # Iterar todos los CSV de la carpeta
    for csv_file in current_dir.glob("*.csv"):
        encode_csv_file(csv_file, encoded_dir, categorical_columns)
