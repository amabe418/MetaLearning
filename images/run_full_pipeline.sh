#!/bin/bash

################################################################################
# PIPELINE COMPLETO: Datasets → Metafeatures → Ejecuciones → CSVs para Metabu
################################################################################

set -e  # Salir si hay algún error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================================"
echo "PIPELINE COMPLETO PARA METABU CON DATASETS DE IMÁGENES"
echo "================================================================================"
echo "Directorio de trabajo: $SCRIPT_DIR"
echo ""

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

################################################################################
# PASO 1: DESCARGAR DATASETS
################################################################################
echo ""
echo "${BLUE}================================================================================${NC}"
echo "${BLUE}PASO 1: DESCARGAR DATASETS DE IMÁGENES${NC}"
echo "${BLUE}================================================================================${NC}"
echo ""

if [ -f "download_all_datasets.py" ]; then
    echo "${GREEN}→ Ejecutando download_all_datasets.py...${NC}"
    python3 download_all_datasets.py
    echo "${GREEN}✓ Datasets descargados${NC}"
else
    echo "${YELLOW}⚠ Script download_all_datasets.py no encontrado, saltando...${NC}"
fi

################################################################################
# PASO 2: EXTRAER META-FEATURES
################################################################################
echo ""
echo "${BLUE}================================================================================${NC}"
echo "${BLUE}PASO 2: EXTRAER META-FEATURES DE DATASETS${NC}"
echo "${BLUE}================================================================================${NC}"
echo ""

if [ -f "extract_metafeatures_descargados.py" ]; then
    echo "${GREEN}→ Ejecutando extract_metafeatures_descargados.py...${NC}"
    python3 extract_metafeatures_descargados.py
    
    if [ -f "metafeatures_consistent.csv" ]; then
        echo "${GREEN}✓ Meta-features extraídas → metafeatures_consistent.csv${NC}"
        echo "  Datasets: $(tail -n +2 metafeatures_consistent.csv | wc -l)"
    else
        echo "${RED}✗ Error: metafeatures_consistent.csv no generado${NC}"
        exit 1
    fi
else
    echo "${YELLOW}⚠ Script extract_metafeatures_descargados.py no encontrado${NC}"
    
    if [ ! -f "metafeatures_consistent.csv" ]; then
        echo "${RED}✗ Error: metafeatures_consistent.csv no existe y no se puede generar${NC}"
        exit 1
    else
        echo "${YELLOW}→ Usando metafeatures_consistent.csv existente${NC}"
    fi
fi

################################################################################
# PASO 3: EJECUTAR REDES NEURONALES
################################################################################
echo ""
echo "${BLUE}================================================================================${NC}"
echo "${BLUE}PASO 3: ENTRENAR REDES NEURONALES Y GENERAR TARGET REPRESENTATIONS${NC}"
echo "${BLUE}================================================================================${NC}"
echo ""

if [ -f "generate_target_representations.py" ]; then
    echo "${GREEN}→ Ejecutando generate_target_representations.py...${NC}"
    echo "${YELLOW}  (Esto puede tomar mucho tiempo dependiendo de GPU/CPU)${NC}"
    python3 generate_target_representations.py
    
    if [ -f "target_representations.csv" ]; then
        echo "${GREEN}✓ Ejecuciones completadas → target_representations.csv${NC}"
        echo "  Configuraciones: $(tail -n +2 target_representations.csv | wc -l)"
    else
        echo "${RED}✗ Error: target_representations.csv no generado${NC}"
        exit 1
    fi
elif [ -f "generate_target_representations_VOL2.py" ]; then
    echo "${GREEN}→ Ejecutando generate_target_representations_VOL2.py...${NC}"
    echo "${YELLOW}  (Esto puede tomar mucho tiempo dependiendo de GPU/CPU)${NC}"
    python3 generate_target_representations_VOL2.py
    
    if [ -f "target_representations.csv" ]; then
        echo "${GREEN}✓ Ejecuciones completadas → target_representations.csv${NC}"
        echo "  Configuraciones: $(tail -n +2 target_representations.csv | wc -l)"
    else
        echo "${RED}✗ Error: target_representations.csv no generado${NC}"
        exit 1
    fi
else
    echo "${YELLOW}⚠ Scripts generate_target_representations*.py no encontrados${NC}"
    
    if [ ! -f "target_representations.csv" ]; then
        echo "${RED}✗ Error: target_representations.csv no existe y no se puede generar${NC}"
        exit 1
    else
        echo "${YELLOW}→ Usando target_representations.csv existente${NC}"
    fi
fi

################################################################################
# PASO 4: PREPARAR CSVs PARA METABU
################################################################################
echo ""
echo "${BLUE}================================================================================${NC}"
echo "${BLUE}PASO 4: PREPARAR CSVs PARA METABU${NC}"
echo "${BLUE}================================================================================${NC}"
echo ""

if [ ! -f "prepare_data_for_metabu.py" ]; then
    echo "${RED}✗ Error: prepare_data_for_metabu.py no encontrado${NC}"
    exit 1
fi

echo "${GREEN}→ Ejecutando prepare_data_for_metabu.py...${NC}"
python3 prepare_data_for_metabu.py

# Verificar archivos generados
echo ""
echo "${BLUE}================================================================================${NC}"
echo "${BLUE}VERIFICANDO ARCHIVOS GENERADOS${NC}"
echo "${BLUE}================================================================================${NC}"
echo ""

files_ok=true

# Verificar basic_representations.csv
if [ -f "basic_representations.csv" ]; then
    rows=$(tail -n +2 basic_representations.csv | wc -l)
    cols=$(head -1 basic_representations.csv | awk -F',' '{print NF}')
    echo "${GREEN}✓ basic_representations.csv${NC}"
    echo "  → $rows datasets, $cols columnas"
else
    echo "${RED}✗ basic_representations.csv NO ENCONTRADO${NC}"
    files_ok=false
fi

# Verificar archivos target por arquitectura
for arch in ResNet18 EfficientNetB0 MobileNetV2; do
    file="target_representations_${arch}.csv"
    if [ -f "$file" ]; then
        rows=$(tail -n +2 "$file" | wc -l)
        cols=$(head -1 "$file" | awk -F',' '{print NF}')
        echo "${GREEN}✓ $file${NC}"
        echo "  → $rows configuraciones, $cols columnas"
    else
        echo "${RED}✗ $file NO ENCONTRADO${NC}"
        files_ok=false
    fi
done

################################################################################
# RESUMEN FINAL
################################################################################
echo ""
echo "${BLUE}================================================================================${NC}"
echo "${BLUE}RESUMEN FINAL${NC}"
echo "${BLUE}================================================================================${NC}"
echo ""

if [ "$files_ok" = true ]; then
    echo "${GREEN}✓✓✓ PIPELINE COMPLETADO EXITOSAMENTE ✓✓✓${NC}"
    echo ""
    echo "Archivos listos para Metabu:"
    echo "  - basic_representations.csv"
    echo "  - target_representations_ResNet18.csv"
    echo "  - target_representations_EfficientNetB0.csv"
    echo "  - target_representations_MobileNetV2.csv"
    echo ""
    echo "Uso con Metabu:"
    echo "${YELLOW}─────────────────────────────────────────────────────────────────────────────${NC}"
    echo "from metabu import Metabu"
    echo "import pandas as pd"
    echo ""
    echo "# Cargar datos"
    echo "basic = pd.read_csv('images/basic_representations.csv')"
    echo "target = pd.read_csv('images/target_representations_ResNet18.csv')"
    echo ""
    echo "# Filtrar datasets comunes"
    echo "common = set(basic['task_id']) & set(target['task_id'])"
    echo "basic_filtered = basic[basic['task_id'].isin(common)]"
    echo "target_filtered = target[target['task_id'].isin(common)]"
    echo ""
    echo "# Entrenar Metabu"
    echo "metabu = Metabu()"
    echo "metabu.train(basic_reprs=basic_filtered,"
    echo "             target_reprs=target_filtered,"
    echo "             column_id='task_id')"
    echo "${YELLOW}─────────────────────────────────────────────────────────────────────────────${NC}"
else
    echo "${RED}✗✗✗ ERRORES EN EL PIPELINE ✗✗✗${NC}"
    echo "Algunos archivos no fueron generados correctamente."
    exit 1
fi

echo ""
echo "${GREEN}Pipeline finalizado en: $SCRIPT_DIR${NC}"
echo ""
