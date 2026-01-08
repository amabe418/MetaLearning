import csv
import os
import numpy as np
import torch
from PIL import Image
import torchvision.datasets as datasets
from skimage.feature import graycomatrix, graycoprops, hog
from skimage.filters import sobel
from skimage.measure import shannon_entropy
from scipy.stats import skew, kurtosis, entropy as scipy_entropy, gmean
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN AVANZADA OPTIMIZADA
# ============================================================================
SAMPLE_CONFIG = {
    'pixel_stats': 500,           # Para estadísticas robustas
    'texture_features': 100,      # Para texturas representativas
    'landmarkers': 2000,          # Para accuracy de clasificadores
    'color_features': 200,        # Para histogramas de color
    'edge_features': 100,         # Para detección de bordes
    'hog_features': 50            # Para HOG (costoso)
}

# Parámetros para normalización
FEATURE_SCALES = {
    'texture': {'min': 0, 'max': 10000},      # Texturas GLCM
    'landmarkers': {'min': 0, 'max': 1},      # Accuracies
    'entropy': {'min': 0, 'max': 10},         # Entropías
    'complexity': {'min': 0, 'max': 5},       # Índices de complejidad
}

# Mapeo de nombres de datasets a nombres de carpetas
folder_names = {
    'EMNIST': 'EMNIST-Letters',
    'FashionMNIST': 'Fashion-MNIST',
    'CIFAR10': 'CIFAR-10',
    'CIFAR100': 'CIFAR-100',
    'GTSRB': 'GTSRB',
    'EuroSAT': 'EuroSAT',
    'Flowers102': 'Flowers102',
    'DTD': 'DTD'
}

# Lista de datasets
datasets_list = [
    ('MNIST', datasets.MNIST, {'train': True}, {'train': False}),
    ('FashionMNIST', datasets.FashionMNIST, {'train': True}, {'train': False}),
    ('EMNIST', datasets.EMNIST, {'split': 'letters', 'train': True}, {'split': 'letters', 'train': False}),
    ('CIFAR10', datasets.CIFAR10, {'train': True}, {'train': False}),
    ('CIFAR100', datasets.CIFAR100, {'train': True}, {'train': False}),
    ('SVHN', datasets.SVHN, {'split': 'train'}, {'split': 'test'}),
    ('USPS', datasets.USPS, {'train': True}, {'train': False}),
    ('Omniglot', datasets.Omniglot, {'background': True}, {'background': False}),
    ('EuroSAT', datasets.EuroSAT, {}, {}),
    ('Flowers102', datasets.Flowers102, {'split': 'train'}, {'split': 'test'}),
    ('DTD', datasets.DTD, {'split': 'train'}, {'split': 'test'}),
    ('GTSRB', datasets.GTSRB, {'split': 'train'}, {'split': 'test'}),
    ('OxfordIIITPet', datasets.OxfordIIITPet, {'split': 'trainval'}, {'split': 'test'}),
]

# ============================================================================
# FUNCIONES DE EXTRACCIÓN CORREGIDAS Y OPTIMIZADAS
# ============================================================================

def _safe_get_targets(dataset, dataset_name=""):
    """Extrae targets de cualquier dataset robustamente."""
    try:
        # Método 1: Atributo directo
        if hasattr(dataset, 'targets') and dataset.targets is not None:
            targets = np.array(dataset.targets)
            if len(targets) > 0:
                return targets
        
        # Método 2: Para SVHN
        if hasattr(dataset, 'labels') and dataset.labels is not None:
            targets = np.array(dataset.labels)
            if len(targets) > 0:
                return targets
        
        # Método 3: Para GTSRB y similares
        if hasattr(dataset, '_samples') and dataset._samples:
            targets = [label for _, label in dataset._samples]
            if len(targets) > 0:
                return np.array(targets)
        
        # Método 4: Para Omniglot
        if dataset_name == 'Omniglot':
            targets = []
            for i in range(min(2000, len(dataset))):
                _, label = dataset[i]
                targets.append(label)
            return np.array(targets)
        
        # Método 5: Extracción manual genérica
        max_samples = min(2000, len(dataset))
        targets = []
        
        for i in range(max_samples):
            try:
                _, label = dataset[i]
                targets.append(int(label))
            except:
                continue
        
        if len(targets) > 0:
            return np.array(targets)
        
        return None
        
    except Exception as e:
        print(f"  [WARN] Error extrayendo targets de {dataset_name}: {str(e)[:100]}")
        return None

def _compute_robust_pixel_stats(dataset, sample_size=500):
    """Estadísticas de píxeles robustas y normalizadas."""
    if len(dataset) == 0:
        return {'mean': 0, 'std': 0, 'median': 0, 'iqr': 0}
    
    pixel_samples = []
    count = min(sample_size, len(dataset))
    
    # Muestreo estratificado para datasets grandes
    if len(dataset) > 10000:
        indices = np.linspace(0, len(dataset)-1, count, dtype=int)
    else:
        indices = np.random.choice(len(dataset), count, replace=False)
    
    for idx in indices:
        try:
            img, _ = dataset[idx]
            if isinstance(img, Image.Image):
                img_array = np.array(img).astype(np.float32) / 255.0
            elif isinstance(img, torch.Tensor):
                img_array = img.numpy().astype(np.float32)
            else:
                img_array = np.array(img).astype(np.float32)
            
            # Para imágenes a color, usar luminosidad (Y de YUV)
            if img_array.ndim > 2:
                if img_array.shape[-1] == 3:  # RGB
                    # Convertir a luminosidad: Y = 0.299R + 0.587G + 0.114B
                    img_array = 0.299 * img_array[..., 0] + 0.587 * img_array[..., 1] + 0.114 * img_array[..., 2]
                else:
                    img_array = img_array.mean(axis=-1)
            
            pixel_samples.append(img_array.flatten())
        except:
            continue
    
    if not pixel_samples:
        return {'mean': 0, 'std': 0, 'median': 0, 'iqr': 0}
    
    # Combinar todas las muestras
    all_pixels = np.concatenate(pixel_samples)
    
    # Estadísticas robustas
    q1, median, q3 = np.percentile(all_pixels, [25, 50, 75])
    
    return {
        'mean': float(np.mean(all_pixels)),
        'std': float(np.std(all_pixels)),
        'median': float(median),
        'iqr': float(q3 - q1),
        'skewness': float(skew(all_pixels)),
        'kurtosis': float(kurtosis(all_pixels)),
        'gmean': float(gmean(all_pixels + 1e-10)),  # Media geométrica
        'entropy': float(shannon_entropy(all_pixels))  # Entropía de Shannon
    }

def _compute_advanced_color_features(dataset, sample_size=200):
    """Características de color avanzadas para metalearning."""
    if len(dataset) == 0:
        return {
            'color_entropy_mean': 0,
            'color_std_mean': 0,
            'color_diversity': 0,
            'color_saturation': 0
        }
    
    color_stats = []
    count = min(sample_size, len(dataset))
    indices = np.random.choice(len(dataset), count, replace=False)
    
    for idx in indices:
        try:
            img, _ = dataset[idx]
            if isinstance(img, Image.Image):
                img_array = np.array(img).astype(np.float32) / 255.0
            elif isinstance(img, torch.Tensor):
                img_array = img.numpy().astype(np.float32)
            else:
                img_array = np.array(img).astype(np.float32)
            
            if img_array.ndim == 2:  # Grayscale
                # Para escala de grises, crear canales dummy
                stats = {'entropy': 0, 'std': 0, 'diversity': 0, 'saturation': 0}
                color_stats.append(stats)
                continue
            
            if img_array.ndim == 3 and img_array.shape[-1] >= 3:
                # Calcular características por canal RGB
                r_channel = img_array[..., 0]
                g_channel = img_array[..., 1]
                b_channel = img_array[..., 2]
                
                # Histograma conjunto para diversidad de color
                combined = img_array.reshape(-1, 3)
                
                # Entropía de color por canal
                hist_r, _ = np.histogram(r_channel.flatten(), bins=32, range=(0, 1))
                hist_g, _ = np.histogram(g_channel.flatten(), bins=32, range=(0, 1))
                hist_b, _ = np.histogram(b_channel.flatten(), bins=32, range=(0, 1))
                
                hist_r = hist_r / hist_r.sum()
                hist_g = hist_g / hist_g.sum()
                hist_b = hist_b / hist_b.sum()
                
                entropy_r = scipy_entropy(hist_r + 1e-10)
                entropy_g = scipy_entropy(hist_g + 1e-10)
                entropy_b = scipy_entropy(hist_b + 1e-10)
                
                # Saturación aproximada
                saturation = np.std([r_channel.std(), g_channel.std(), b_channel.std()])
                
                # Diversidad de color (número de colores únicos)
                unique_colors = len(np.unique(combined[:1000], axis=0)) if len(combined) > 1000 else len(np.unique(combined, axis=0))
                color_diversity = unique_colors / 1000  # Normalizado
                
                color_stats.append({
                    'entropy': (entropy_r + entropy_g + entropy_b) / 3,
                    'std': np.mean([r_channel.std(), g_channel.std(), b_channel.std()]),
                    'diversity': min(color_diversity, 1.0),
                    'saturation': saturation
                })
        except:
            continue
    
    if len(color_stats) == 0:
        return {
            'color_entropy_mean': 0,
            'color_std_mean': 0,
            'color_diversity': 0,
            'color_saturation': 0
        }
    
    # Promediar sobre muestras
    return {
        'color_entropy_mean': float(np.mean([s['entropy'] for s in color_stats])),
        'color_std_mean': float(np.mean([s['std'] for s in color_stats])),
        'color_diversity': float(np.mean([s['diversity'] for s in color_stats])),
        'color_saturation': float(np.mean([s['saturation'] for s in color_stats]))
    }

def _compute_normalized_texture_features(dataset, sample_size=100):
    """Características de textura normalizadas."""
    if len(dataset) == 0:
        return {
            'texture_contrast_norm': 0,
            'texture_homogeneity_norm': 0,
            'texture_energy_norm': 0,
            'texture_correlation_norm': 0,
            'texture_complexity': 0
        }
    
    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    features = {k: [] for k in ['contrast', 'homogeneity', 'energy', 'correlation']}
    count = min(sample_size, len(dataset))
    indices = np.random.choice(len(dataset), count, replace=False)
    
    for idx in indices:
        try:
            img, _ = dataset[idx]
            if isinstance(img, Image.Image):
                img_gray = np.array(img.convert('L')).astype(np.uint8)
            elif isinstance(img, torch.Tensor):
                img_gray = img.mean(dim=0).numpy().astype(np.uint8)
            else:
                img_gray = np.array(img).mean(axis=0).astype(np.uint8) if img.ndim > 2 else img.astype(np.uint8)
            
            # GLCM con múltiples configuraciones
            glcm = graycomatrix(img_gray, distances=distances, angles=angles, 
                               levels=256, symmetric=True, normed=True)
            
            for prop in ['contrast', 'homogeneity', 'energy', 'correlation']:
                value = graycoprops(glcm, prop).mean()
                features[prop].append(value)
                
        except:
            continue
    
    # Normalizar características de textura
    texture_features = {}
    if features['contrast']:
        # Log transform para contrast (valores muy grandes)
        contrast_log = np.log1p(np.mean(features['contrast']))
        texture_features['texture_contrast_norm'] = float(min(contrast_log / 10, 1.0))
    else:
        texture_features['texture_contrast_norm'] = 0.0
    
    # Las otras características ya están en rangos razonables
    for prop in ['homogeneity', 'energy', 'correlation']:
        key = f'texture_{prop}_norm'
        if features[prop]:
            texture_features[key] = float(np.clip(np.mean(features[prop]), 0, 1))
        else:
            texture_features[key] = 0.0
    
    # Complejidad de textura (inversa de homogeneidad)
    if 'texture_homogeneity_norm' in texture_features:
        texture_features['texture_complexity'] = 1.0 - texture_features['texture_homogeneity_norm']
    
    return texture_features

def _compute_edge_and_shape_features(dataset, sample_size=100):
    """Características de bordes y forma para metalearning."""
    if len(dataset) == 0:
        return {
            'edge_density': 0,
            'edge_std': 0,
            'shape_complexity': 0,
            'object_size_ratio': 0
        }
    
    edge_stats = []
    count = min(sample_size, len(dataset))
    indices = np.random.choice(len(dataset), count, replace=False)
    
    for idx in indices:
        try:
            img, _ = dataset[idx]
            if isinstance(img, Image.Image):
                img_gray = np.array(img.convert('L')).astype(np.float32) / 255.0
            elif isinstance(img, torch.Tensor):
                img_gray = img.mean(dim=0).numpy().astype(np.float32)
            else:
                img_gray = np.array(img).mean(axis=0).astype(np.float32) if img.ndim > 2 else img.astype(np.float32)
            
            # Detección de bordes
            edges = sobel(img_gray)
            
            # Densidad de bordes (umbral adaptativo)
            edge_threshold = np.percentile(np.abs(edges.flatten()), 90)  # Percentil 90
            edge_density = np.mean(np.abs(edges) > edge_threshold)
            
            # Variabilidad de bordes
            edge_std = np.std(edges)
            
            # Complejidad de forma (entropía de bordes)
            shape_complexity = shannon_entropy(np.abs(edges) + 1e-10) / 10  # Normalizado
            
            # Ratio objeto/fondo (simplificado)
            object_ratio = np.mean(img_gray > 0.5)  # Asumiendo fondo claro
            
            edge_stats.append({
                'density': edge_density,
                'std': edge_std,
                'complexity': shape_complexity,
                'object_ratio': object_ratio
            })
        except:
            continue
    
    if len(edge_stats) == 0:
        return {
            'edge_density': 0,
            'edge_std': 0,
            'shape_complexity': 0,
            'object_size_ratio': 0
        }
    
    return {
        'edge_density': float(np.mean([s['density'] for s in edge_stats])),
        'edge_std': float(np.mean([s['std'] for s in edge_stats])),
        'shape_complexity': float(np.mean([s['complexity'] for s in edge_stats])),
        'object_size_ratio': float(np.mean([s['object_ratio'] for s in edge_stats]))
    }

def _compute_advanced_landmarkers(dataset, sample_size=2000):
    """Landmarkers avanzados con validación cruzada."""
    try:
        # Extraer subset representativo
        if len(dataset) < 100:
            sample_size = min(sample_size, len(dataset))
        
        # Para datasets grandes, muestrear estratégicamente
        if len(dataset) > 5000:
            # Muestrear uniformemente
            indices = np.linspace(0, len(dataset)-1, sample_size, dtype=int)
        else:
            indices = np.random.choice(len(dataset), sample_size, replace=False)
        
        X_samples = []
        y_samples = []
        
        for idx in indices:
            try:
                img, label = dataset[idx]
                if isinstance(img, Image.Image):
                    img_array = np.array(img.convert('L')).astype(np.float32).flatten() / 255.0
                elif isinstance(img, torch.Tensor):
                    img_array = img.mean(dim=0).numpy().flatten().astype(np.float32)
                else:
                    img_array = np.array(img).mean(axis=0).flatten().astype(np.float32) if img.ndim > 2 else img.flatten().astype(np.float32)
                
                # Reducción inteligente de dimensionalidad
                if len(img_array) > 784:  # Más grande que 28x28
                    # Muestrear píxeles estratégicamente
                    step = len(img_array) // 784
                    img_array = img_array[::step][:784]
                elif len(img_array) < 100:
                    # Padding si es muy pequeño
                    img_array = np.pad(img_array, (0, 100 - len(img_array)), mode='constant')
                
                X_samples.append(img_array)
                y_samples.append(int(label))
            except:
                continue
        
        if len(X_samples) < 50 or len(set(y_samples)) < 2:
            print(f"  [WARN] Insuficientes muestras para landmarkers")
            return {
                'landmarker_1nn': 0.5,
                'landmarker_lda': 0.5,
                'landmarker_tree': 0.5,
                'landmarker_svm': 0.5,
                'landmarker_nb': 0.5
            }
        
        X = np.array(X_samples)
        y = np.array(y_samples)
        
        # Para datasets con muchas clases, limitar
        if len(np.unique(y)) > 50:
            # Tomar solo las clases más frecuentes
            unique, counts = np.unique(y, return_counts=True)
            top_classes = unique[np.argsort(-counts)[:50]]
            mask = np.isin(y, top_classes)
            X = X[mask]
            y = y[mask]
        
        # Validación cruzada estratificada (3 folds)
        cv = StratifiedKFold(n_splits=min(3, len(np.unique(y))), shuffle=True, random_state=42)
        
        landmarkers = {}
        
        # 1. 1-Nearest Neighbor
        try:
            knn = KNeighborsClassifier(n_neighbors=1)
            scores = cross_val_score(knn, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
            landmarkers['landmarker_1nn'] = float(np.mean(scores))
        except:
            landmarkers['landmarker_1nn'] = 0.5
        
        # 2. Linear Discriminant Analysis
        try:
            lda = LinearDiscriminantAnalysis()
            scores = cross_val_score(lda, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
            landmarkers['landmarker_lda'] = float(np.mean(scores))
        except:
            landmarkers['landmarker_lda'] = 0.5
        
        # 3. Decision Tree (profundidad limitada)
        try:
            tree = DecisionTreeClassifier(max_depth=5, random_state=42)
            scores = cross_val_score(tree, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
            landmarkers['landmarker_tree'] = float(np.mean(scores))
        except:
            landmarkers['landmarker_tree'] = 0.5
        
        # 4. Linear SVM
        try:
            svm = LinearSVC(max_iter=1000, random_state=42)
            scores = cross_val_score(svm, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
            landmarkers['landmarker_svm'] = float(np.mean(scores))
        except:
            landmarkers['landmarker_svm'] = 0.5
        
        # 5. Naive Bayes
        try:
            nb = GaussianNB()
            scores = cross_val_score(nb, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
            landmarkers['landmarker_nb'] = float(np.mean(scores))
        except:
            landmarkers['landmarker_nb'] = 0.5
        
        return landmarkers
        
    except Exception as e:
        print(f"  [ERROR] en landmarkers: {str(e)[:100]}")
        return {
            'landmarker_1nn': 0.5,
            'landmarker_lda': 0.5,
            'landmarker_tree': 0.5,
            'landmarker_svm': 0.5,
            'landmarker_nb': 0.5
        }

def _compute_dataset_complexity_metrics(dataset):
    """Métricas de complejidad avanzadas para metalearning."""
    targets = _safe_get_targets(dataset)
    
    if targets is None or len(targets) == 0:
        return {
            'class_entropy': 0.5,
            'imbalance_ratio': 1.0,
            'n_classes': 0,
            'class_overlap_index': 0.5,
            'fisher_discriminant_ratio': 0.5
        }
    
    unique, counts = np.unique(targets, return_counts=True)
    n_classes = len(unique)
    
    if n_classes <= 1:
        return {
            'class_entropy': 0,
            'imbalance_ratio': 1.0,
            'n_classes': n_classes,
            'class_overlap_index': 0,
            'fisher_discriminant_ratio': 0
        }
    
    # Entropía de clases normalizada
    probs = counts / counts.sum()
    class_entropy = -np.sum(probs * np.log2(probs + 1e-10))
    normalized_entropy = class_entropy / np.log2(n_classes) if n_classes > 1 else 0
    
    # Ratio de desbalanceo (log scale para normalizar)
    imbalance_ratio = counts.max() / counts.min()
    log_imbalance = np.log1p(imbalance_ratio - 1)  # 1 -> 0, >1 -> positivo
    
    # Índice de superposición de clases (simplificado)
    # Más clases = más potencial superposición
    overlap_index = min(n_classes / 20, 1.0)  # Normalizado a 20 clases
    
    # Ratio discriminante de Fisher (simplificado)
    # Asume que más clases = más difícil discriminación
    fisher_ratio = 1.0 / (1.0 + n_classes/10)
    
    return {
        'class_entropy': float(normalized_entropy),
        'imbalance_ratio': float(log_imbalance),
        'n_classes': int(n_classes),
        'class_overlap_index': float(overlap_index),
        'fisher_discriminant_ratio': float(fisher_ratio)
    }

def _compute_spatial_features(dataset, sample_size=100):
    """Características espaciales y estructurales."""
    if len(dataset) == 0:
        return {
            'spatial_compactness': 0.5,
            'aspect_ratio_variation': 0.5,
            'centroid_variation': 0.5
        }
    
    spatial_stats = []
    count = min(sample_size, len(dataset))
    indices = np.random.choice(len(dataset), count, replace=False)
    
    for idx in indices:
        try:
            img, _ = dataset[idx]
            if isinstance(img, Image.Image):
                img_array = np.array(img.convert('L')).astype(np.float32) / 255.0
            elif isinstance(img, torch.Tensor):
                img_array = img.mean(dim=0).numpy().astype(np.float32)
            else:
                img_array = np.array(img).mean(axis=0).astype(np.float32) if img.ndim > 2 else img.astype(np.float32)
            
            # Binarizar para análisis de forma
            binary = img_array > 0.5
            
            if np.any(binary):
                # Encontrar regiones activas
                rows = np.any(binary, axis=1)
                cols = np.any(binary, axis=0)
                
                if np.any(rows) and np.any(cols):
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]
                    
                    height = rmax - rmin + 1
                    width = cmax - cmin + 1
                    
                    # Compacidad espacial (área objeto / área bounding box)
                    object_area = np.sum(binary)
                    bbox_area = height * width
                    compactness = object_area / bbox_area if bbox_area > 0 else 0
                    
                    # Ratio de aspecto
                    aspect_ratio = width / height if height > 0 else 1
                    
                    # Variación del centroide
                    y_indices, x_indices = np.where(binary)
                    centroid_x = np.mean(x_indices) / img_array.shape[1]
                    centroid_y = np.mean(y_indices) / img_array.shape[0]
                    centroid_variation = np.std([centroid_x, centroid_y])
                    
                    spatial_stats.append({
                        'compactness': compactness,
                        'aspect_ratio': aspect_ratio,
                        'centroid_variation': centroid_variation
                    })
        except:
            continue
    
    if len(spatial_stats) == 0:
        return {
            'spatial_compactness': 0.5,
            'aspect_ratio_variation': 0.5,
            'centroid_variation': 0.5
        }
    
    # Calcular variaciones
    compactness_vals = [s['compactness'] for s in spatial_stats]
    aspect_ratios = [s['aspect_ratio'] for s in spatial_stats]
    centroid_vars = [s['centroid_variation'] for s in spatial_stats]
    
    return {
        'spatial_compactness': float(np.mean(compactness_vals)),
        'aspect_ratio_variation': float(np.std(aspect_ratios) / (np.mean(aspect_ratios) + 1e-10)),
        'centroid_variation': float(np.mean(centroid_vars))
    }

def _normalize_features(features_dict):
    """Normaliza todas las características a rangos consistentes."""
    normalized = features_dict.copy()
    
    # Normalizar landmarkers (ya deberían estar 0-1)
    for key in ['landmarker_1nn', 'landmarker_lda', 'landmarker_tree', 'landmarker_svm', 'landmarker_nb']:
        if key in normalized:
            normalized[key] = float(np.clip(normalized[key], 0, 1))
    
    # Normalizar características de textura
    for key in ['texture_contrast_norm', 'texture_homogeneity_norm', 
                'texture_energy_norm', 'texture_correlation_norm', 'texture_complexity']:
        if key in normalized:
            normalized[key] = float(np.clip(normalized[key], 0, 1))
    
    # Normalizar características de color
    for key in ['color_entropy_mean', 'color_diversity', 'color_saturation']:
        if key in normalized:
            normalized[key] = float(np.clip(normalized[key], 0, 1))
    
    # Normalizar características de bordes
    for key in ['edge_density', 'edge_std', 'shape_complexity', 'object_size_ratio']:
        if key in normalized:
            normalized[key] = float(np.clip(normalized[key], 0, 1))
    
    # Normalizar características espaciales
    for key in ['spatial_compactness', 'aspect_ratio_variation', 'centroid_variation']:
        if key in normalized:
            normalized[key] = float(np.clip(normalized[key], 0, 1))
    
    # Normalizar complejidad
    if 'class_overlap_index' in normalized:
        normalized['class_overlap_index'] = float(np.clip(normalized['class_overlap_index'], 0, 1))
    
    if 'fisher_discriminant_ratio' in normalized:
        normalized['fisher_discriminant_ratio'] = float(np.clip(normalized['fisher_discriminant_ratio'], 0, 1))
    
    return normalized

def _compute_meta_learning_features(features_dict):
    """Calcula características derivadas específicas para metalearning."""
    meta_features = {}
    
    # 1. Score de dificultad del dataset
    if all(k in features_dict for k in ['landmarker_1nn', 'landmarker_lda', 'landmarker_svm']):
        avg_accuracy = (features_dict['landmarker_1nn'] + 
                       features_dict['landmarker_lda'] + 
                       features_dict['landmarker_svm']) / 3
        meta_features['dataset_difficulty'] = 1.0 - avg_accuracy
    
    # 2. Score de variabilidad visual
    if all(k in features_dict for k in ['color_entropy_mean', 'texture_complexity', 'edge_density']):
        meta_features['visual_variability'] = (
            features_dict['color_entropy_mean'] * 0.4 +
            features_dict['texture_complexity'] * 0.3 +
            features_dict['edge_density'] * 0.3
        )
    
    # 3. Score de regularidad estructural
    if all(k in features_dict for k in ['spatial_compactness', 'aspect_ratio_variation']):
        meta_features['structural_regularity'] = (
            features_dict['spatial_compactness'] * 0.6 +
            (1.0 - features_dict['aspect_ratio_variation']) * 0.4
        )
    
    # 4. Tipo de dataset (clasificación simplificada)
    if 'channels' in features_dict and 'n_classes' in features_dict:
        if features_dict['channels'] == 1:
            meta_features['dataset_type'] = 'grayscale'
        elif features_dict['n_classes'] <= 10:
            meta_features['dataset_type'] = 'few_classes_color'
        elif features_dict['n_classes'] <= 100:
            meta_features['dataset_type'] = 'medium_classes_color'
        else:
            meta_features['dataset_type'] = 'many_classes_color'
    
    return meta_features

# ============================================================================
# FUNCIÓN PRINCIPAL DE EXTRACCIÓN OPTIMIZADA
# ============================================================================

def extract_metafeatures_for_metalearning(name, dataset_class, train_kwargs, test_kwargs):
    """Extracción optimizada de metafeatures para metalearning."""
    try:
        print(f"\n{'='*60}")
        print(f"Procesando {name}")
        print(f"{'='*60}")
        
        folder = folder_names.get(name, name)
        root = f'./data/{folder}'
        
        # Cargar dataset
        train_kwargs_full = train_kwargs.copy()
        train_kwargs_full['root'] = root
        train_kwargs_full['download'] = False
        
        print("  [1/10] Cargando dataset...")
        train_dataset = dataset_class(**train_kwargs_full)
        
        # Información básica
        num_train = len(train_dataset)
        
        if num_train == 0:
            raise ValueError(f"Dataset {name} vacío")
        
        # Obtener información de resolución
        print("  [2/10] Obteniendo información básica...")
        if len(train_dataset) > 0:
            img, _ = train_dataset[0]
            if isinstance(img, Image.Image):
                resolution = f"{img.size[0]}x{img.size[1]}"
                channels = len(img.getbands()) if hasattr(img, 'getbands') else 1
            elif isinstance(img, torch.Tensor):
                resolution = f"{img.shape[1]}x{img.shape[2]}" if img.ndim > 2 else f"{img.shape[0]}x{img.shape[1]}"
                channels = img.shape[0] if img.ndim > 2 else 1
            else:
                resolution = str(img.shape[:2]) if img.ndim > 2 else str(img.shape)
                channels = img.shape[2] if img.ndim > 2 else 1
        else:
            resolution = 'N/A'
            channels = 'N/A'
        
        # Calcular tamaño estimado
        img_sample, _ = train_dataset[0]
        if isinstance(img_sample, Image.Image):
            img_size = img_sample.size[0] * img_sample.size[1] * channels
        elif isinstance(img_sample, torch.Tensor):
            img_size = img_sample.numel()
        else:
            img_size = np.array(img_sample).size
        
        total_size_mb = (img_size * 4 * num_train) / (1024 ** 2)
        
        # Calcular todos los metafeatures
        print("  [3/10] Calculando estadísticas de píxeles...")
        pixel_stats = _compute_robust_pixel_stats(train_dataset, SAMPLE_CONFIG['pixel_stats'])
        
        print("  [4/10] Calculando métricas de clase...")
        class_metrics = _compute_dataset_complexity_metrics(train_dataset)
        
        print("  [5/10] Calculando características de color...")
        color_features = _compute_advanced_color_features(train_dataset, SAMPLE_CONFIG['color_features'])
        
        print("  [6/10] Calculando características de textura...")
        texture_features = _compute_normalized_texture_features(train_dataset, SAMPLE_CONFIG['texture_features'])
        
        print("  [7/10] Calculando características de bordes...")
        edge_features = _compute_edge_and_shape_features(train_dataset, SAMPLE_CONFIG['edge_features'])
        
        print("  [8/10] Calculando landmarkers (MÁS IMPORTANTE)...")
        landmarkers = _compute_advanced_landmarkers(train_dataset, SAMPLE_CONFIG['landmarkers'])
        
        print("  [9/10] Calculando características espaciales...")
        spatial_features = _compute_spatial_features(train_dataset, SAMPLE_CONFIG['edge_features'])
        
        # Compilar resultados iniciales
        print("  [10/10] Compilando y normalizando resultados...")
        all_features = {
            # Identificación
            'dataset': name,
            
            # Básicos
            'num_train': num_train,
            'resolution': resolution,
            'channels': channels,
            'total_size_mb': round(total_size_mb, 2),
            
            # Estadísticas de píxeles
            'pixel_mean': pixel_stats['mean'],
            'pixel_std': pixel_stats['std'],
            'pixel_median': pixel_stats['median'],
            'pixel_iqr': pixel_stats['iqr'],
            'pixel_skewness': pixel_stats['skewness'],
            'pixel_kurtosis': pixel_stats['kurtosis'],
            'pixel_gmean': pixel_stats['gmean'],
            'pixel_entropy': pixel_stats['entropy'],
            
            # Métricas de clase y complejidad
            **class_metrics,
            
            # Características de color
            **color_features,
            
            # Características de textura
            **texture_features,
            
            # Características de bordes y forma
            **edge_features,
            
            # Landmarkers (MÁS IMPORTANTES)
            **landmarkers,
            
            # Características espaciales
            **spatial_features,
        }
        
        # Calcular características derivadas para metalearning
        meta_features = _compute_meta_learning_features(all_features)
        all_features.update(meta_features)
        
        # Normalizar todas las características
        all_features = _normalize_features(all_features)
        
        print(f"  ✓ Completado: {len(all_features)} metafeatures extraídos")
        return all_features
        
    except Exception as e:
        print(f"  ✗ Error: {str(e)[:200]}")
        import traceback
        traceback.print_exc()
        return {'dataset': name, 'error': str(e)}

# ============================================================================
# EJECUCIÓN PRINCIPAL OPTIMIZADA
# ============================================================================

def main():
    """Función principal de ejecución."""
    print("=" * 70)
    print("EXTRACCIÓN DE METAFEATURES PARA METALEARNING")
    print("=" * 70)
    
    # Filtrar datasets disponibles
    available_datasets = []
    for name, dataset_class, train_kwargs, test_kwargs in datasets_list:
        folder = folder_names.get(name, name)
        root = f'./data/{folder}'
        if os.path.exists(root):
            available_datasets.append((name, dataset_class, train_kwargs, test_kwargs))
    
    print(f"Encontrados {len(available_datasets)} datasets disponibles")
    print("=" * 70)
    
    # Procesar todos los datasets
    results = []
    for name, dataset_class, train_kwargs, test_kwargs in available_datasets:
        meta = extract_metafeatures_for_metalearning(name, dataset_class, train_kwargs, test_kwargs)
        results.append(meta)
    
    # Guardar resultados
    if results:
        # Ordenar columnas por importancia para metalearning
        field_order = [
            # Identificación
            'dataset',
            
            # Características CRÍTICAS para metalearning (Landmarkers)
            'landmarker_1nn', 'landmarker_lda', 'landmarker_svm', 
            'landmarker_tree', 'landmarker_nb',
            
            # Scores derivados para metalearning
            'dataset_difficulty', 'visual_variability', 'structural_regularity',
            'dataset_type',
            
            # Complejidad del dataset
            'class_entropy', 'imbalance_ratio', 'class_overlap_index',
            'fisher_discriminant_ratio', 'n_classes',
            
            # Características visuales
            'color_entropy_mean', 'color_diversity', 'color_saturation',
            'texture_complexity', 'texture_homogeneity_norm',
            'edge_density', 'shape_complexity',
            
            # Características espaciales
            'spatial_compactness', 'aspect_ratio_variation', 'centroid_variation',
            
            # Estadísticas básicas
            'num_train', 'channels', 'resolution', 'total_size_mb',
            'pixel_mean', 'pixel_std', 'pixel_entropy',
            'pixel_skewness', 'pixel_kurtosis',
            
            # Error
            'error'
        ]
        
        # Filtrar solo las que existen en los resultados
        existing_fields = []
        for field in field_order:
            if any(field in result for result in results):
                existing_fields.append(field)
        
        # Añadir cualquier campo adicional que no esté en field_order
        all_keys = set()
        for result in results:
            all_keys.update(result.keys())
        
        additional_fields = sorted(list(all_keys - set(existing_fields)))
        fieldnames = existing_fields + additional_fields
        
        # Guardar CSV principal
        with open('metafeatures_metalearning_optimizado.csv', 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                row = {field: result.get(field, 'N/A') for field in fieldnames}
                writer.writerow(row)
        
        # Guardar CSV simplificado para análisis rápido
        key_features = [
            'dataset', 'dataset_difficulty', 'visual_variability', 
            'structural_regularity', 'landmarker_1nn', 'landmarker_lda',
            'landmarker_svm', 'n_classes', 'channels', 'class_entropy'
        ]
        
        with open('metafeatures_key_features.csv', 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=key_features)
            writer.writeheader()
            for result in results:
                row = {field: result.get(field, 'N/A') for field in key_features}
                writer.writerow(row)
        
        # Mostrar resumen estadístico
        print(f"\n{'='*70}")
        print("RESUMEN ESTADÍSTICO")
        print("=" * 70)
        
        # Calcular estadísticas de los landmarkers
        landmarker_cols = [c for c in fieldnames if c.startswith('landmarker_')]
        if landmarker_cols:
            print("\nAccuracy de Landmarkers (más alto = más fácil):")
            for col in landmarker_cols:
                values = [r.get(col, 0) for r in results if isinstance(r.get(col), (int, float))]
                if values:
                    print(f"  {col}: min={min(values):.3f}, avg={np.mean(values):.3f}, max={max(values):.3f}")
        
        # Mostrar los datasets más difíciles y fáciles
        if 'dataset_difficulty' in fieldnames:
            difficulties = [(r['dataset'], r.get('dataset_difficulty', 0)) for r in results 
                          if isinstance(r.get('dataset_difficulty'), (int, float))]
            
            if difficulties:
                difficulties.sort(key=lambda x: x[1])
                print("\nTop 5 datasets más FÁCILES (menor dificultad):")
                for name, diff in difficulties[:5]:
                    print(f"  {name}: {diff:.3f}")
                
                print("\nTop 5 datasets más DIFÍCILES (mayor dificultad):")
                for name, diff in difficulties[-5:]:
                    print(f"  {name}: {diff:.3f}")
        
        print(f"\n{'='*70}")
        print(f"¡PROCESO COMPLETADO EXITOSAMENTE!")
        print(f"Metafeatures principales guardados en: metafeatures_metalearning_optimizado.csv")
        print(f"Características clave guardadas en: metafeatures_key_features.csv")
        print(f"Total de datasets procesados: {len(results)}")
        print(f"Total de metafeatures por dataset: {len(fieldnames)}")
        print("=" * 70)
    
    else:
        print("No se encontraron datasets para procesar")

if __name__ == "__main__":
    main()