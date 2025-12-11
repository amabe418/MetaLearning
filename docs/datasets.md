# Datasets descargados (OpenML)

Resumen de los datasets guardados en `data/raw/` y sus metadatos principales (meta-features calculados y qualities de OpenML).

## 15 - breast-w (`data/raw/15_breast-w.csv`)
- Dominio: salud, diagnostico de cancer de mama (Wisconsin).
- Tarea: clasificacion binaria.
- Tamano: 699 filas, 9 atributos + `target`.
- Tipos: 9 numericos; qualities reporta 1 simbolico (el resto numericos).
- Clases: 2.
- Faltantes: 16 valores (~0.23%); ~2.3% de instancias con NA.
- Notas: medidas de caracteristicas celulares.

## 54 - vehicle (`data/raw/54_vehicle.csv`)
- Dominio: reconocimiento de vehiculos por caracteristicas fisicas.
- Tarea: clasificacion multiclase.
- Tamano: 846 filas, 18 atributos + `target`.
- Tipos: 18 numericos; 1 simbolico (clase).
- Clases: 4.
- Faltantes: 0.

## 195 - auto_price (`data/raw/195_auto_price.csv`)
- Dominio: precios de automoviles.
- Tarea: regresion.
- Tamano: 159 filas, 15 atributos + `target`.
- Tipos: 15 numericos; 1 simbolico.
- Faltantes: 0.

## 520 - analcatdata_wildcat (`data/raw/520_analcatdata_wildcat.csv`)
- Dominio: laboral/industrial (quejas sindicales).
- Tarea: regresion.
- Tamano: 163 filas, 5 atributos + `target`.
- Tipos: mezcla numerica/categorica (qualities: 4 numericos, 2 simbolicos).
- Clases: 0 (indicativo de regresion).
- Faltantes: 0.
- Notas: incluye variables enteras y log-transform.

## 626 - fri_c2_500_50 (`data/raw/626_fri_c2_500_50.csv`)
- Dominio: sintetico (familia Friedman).
- Tarea: regresion.
- Tamano: 500 filas, 50 atributos + `target`.
- Tipos: 51 numericos; 0 simbolicos (qualities).
- Clases: 0.
- Faltantes: 0.

## 704 - disclosure_x_noise (`data/raw/704_disclosure_x_noise.csv`)
- Dominio: socioeconomico / ingresos.
- Tarea: regresion.
- Tamano: 662 filas, 3 atributos + `target`.
- Tipos: 3 numericos; 0 simbolicos.
- Clases: 0.
- Faltantes: 0.

## 1426 - a5a (`data/raw/1426_a5a.csv`)
- Dominio: censo/ingresos (version del Adult de LIBSVM).
- Tarea: clasificacion binaria (qualities reporta 0 clases; tipicamente 2 clases en Adult).
- Tamano: 32,561 filas, 123 atributos + `target`.
- Tipos: 124 numericos; 0 simbolicos.
- Faltantes: 0.

## 1427 - a6a (`data/raw/1427_a6a.csv`)
- Dominio: censo/ingresos (version del Adult de LIBSVM).
- Tarea: clasificacion binaria (qualities reporta 0 clases; tipicamente 2 clases en Adult).
- Tamano: 32,561 filas, 123 atributos + `target`.
- Tipos: 124 numericos; 0 simbolicos.
- Faltantes: 0.

## 1489 - phoneme (`data/raw/1489_phoneme.csv`)
- Dominio: reconocimiento de habla.
- Tarea: clasificacion binaria.
- Tamano: 5,404 filas, 5 atributos numericos + `target`.
- Tipos: 5 numericos; 1 simbolico (clase).
- Clases: 2.
- Faltantes: 0.
- Notas: datos estandarizados.

## 3046 - QSAR-TID-10878 (`data/raw/3046_QSAR-TID-10878.csv`)
- Dominio: quimica computacional (QSAR).
- Tarea: regresion.
- Tamano: 427 filas, 1,024 descriptores + `target`.
- Tipos: 1,025 numericos; 1 simbolico.
- Clases: 0.
- Faltantes: 0.

## 3383 - QSAR-TID-12718 (`data/raw/3383_QSAR-TID-12718.csv`)
- Dominio: quimica computacional (QSAR).
- Tarea: regresion.
- Tamano: 134 filas, 1,024 descriptores + `target`.
- Tipos: 1,025 numericos; 1 simbolico.
- Clases: 0.
- Faltantes: 0.
- Notas: muy alto numero de atributos con pocos ejemplos.

## 40693 - xd6 (`data/raw/40693_xd6.csv`)
- Dominio: no especificado (nombre abreviado en OpenML).
- Tarea: clasificacion binaria.
- Tamano: 973 filas, 9 atributos + `target`.
- Tipos: qualities reporta 0 numericos y 10 simbolicos (mayormente categorico).
- Clases: 2.
- Faltantes: 0.

## 41146 - sylvine (`data/raw/41146_sylvine.csv`)
- Dominio: no especificado; dataset tabular de referencia en OpenML.
- Tarea: clasificacion binaria.
- Tamano: 5,124 filas, 20 atributos + `target`.
- Tipos: 20 numericos; 1 simbolico.
- Clases: 2.
- Faltantes: 0.

## 41164 - fabert (`data/raw/41164_fabert.csv`)
- Dominio: vision/quimica (descriptor de imagen/quimico, segun OpenML).
- Tarea: clasificacion multiclase.
- Tamano: 8,237 filas, 800 atributos + `target`.
- Tipos: 800 numericos; 1 simbolico.
- Clases: 7.
- Faltantes: 0.

## 41754 - FOREX_eurjpy-day-Close (`data/raw/41754_FOREX_eurjpy-day-Close.csv`)
- Dominio: series financieras (Forex EUR/JPY).
- Tarea: clasificacion binaria (qualities: 2 clases).
- Tamano: 1,832 filas, 11 atributos + `target`.
- Tipos: 11 numericos; 1 simbolico.
- Clases: 2.
- Faltantes: 0.

## 42256 - Asteroid_Dataset (`data/raw/42256_Asteroid_Dataset.csv`)
- Dominio: astronomia/asteroides.
- Tarea: clasificacion binaria (qualities: 2 clases).
- Tamano: 126,131 filas, 33 atributos + `target`.
- Tipos: 32 numericos; 1 simbolico.
- Clases: 2.
- Faltantes: 99 valores (~0.0023%).

## 44019 - house_sales (`data/raw/44019_house_sales.csv`)
- Dominio: inmuebles (precios de casas).
- Tarea: regresion.
- Tamano: 21,613 filas, 15 atributos + `target`.
- Tipos: 15 numericos; 1 simbolico.
- Clases: 0.
- Faltantes: 0.

## 44565 - philippine_seed_2_nrows_2000_nclasses_10_ncols_100_stratify_True (`data/raw/44565_philippine_seed_2_nrows_2000_nclasses_10_ncols_100_stratify_True.csv`)
- Dominio: sintetico/tabular (nombre indica 10 clases).
- Tarea: qualities reporta 2 clases (posible discrepancia con el nombre).
- Tamano: 2,000 filas, 100 atributos + `target`.
- Tipos: 100 numericos; 1 simbolico.
- Faltantes: 0.
- Notas: verificar numero real de clases por la discrepancia nombre/qualities.

## 44573 - arcene_seed_0_nrows_2000_nclasses_10_ncols_100_stratify_True (`data/raw/44573_arcene_seed_0_nrows_2000_nclasses_10_ncols_100_stratify_True.csv`)
- Dominio: bioinformatica/quimica (Arcene es un benchmark de cancer vs control).
- Tarea: qualities reporta 2 clases (el nombre indica 10; posible discrepancia).
- Tamano: 100 filas, 100 atributos + `target`.
- Tipos: 100 numericos; 1 simbolico.
- Faltantes: 0.
- Notas: verificar numero real de clases.

## 46653 - wine_reviews (`data/raw/46653_wine_reviews.csv`)
- Dominio: opiniones sobre vinos (texto + numerico).
- Tarea: clasificacion multiclase (variedad de uva).
- Tamano: 84,123 filas, 5 columnas (`country`, `description`, `points`, `price`, `province`, mas `target`).
- Tipos: mezcla texto/categorico y numerico; qualities: 2 numericos, 0 simbolicos.
- Clases: 30.
- Faltantes: 5,644 valores (~1.12%); ~6.7% de instancias con NA.

### Meta-features y qualities
- `data/meta_features/meta_features.csv`: meta-features calculados con `amltk` (ej. `instance_count`, `number_of_features`, `percentage_missing_values`, etc.) mas qualities prefijadas `quality_*`.
- `data/meta_features/openml_qualities.csv`: qualities originales de OpenML sin prefijo (numero de clases, porcentajes de faltantes, landmarks simples, etc.).
