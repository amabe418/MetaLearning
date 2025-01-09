import csv
import os
import numpy as np
import torch
from PIL import Image
import torchvision.datasets as datasets
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import sobel
from skimage.measure import shannon_entropy
from scipy.stats import skew, kurtosis, entropy as scipy_entropy, gmean
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN AVANZADA OPTIMIZADA
# ============================================================================
SAMPLE_CONFIG = {
    'pixel_stats': 300,           # Reducido para consistencia
    'texture_features': 80,       # Reducido
    'landmarkers': 1000,          # Reducido pero más consistente
    'color_features': 150,        # Reducido
    'edge_features': 80,          # Reducido
    'hog_features': 0            # Desactivado (causa inconsistencia)
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
# FUNCIONES DE EXTRACCIÓN MEJORADAS PARA CONSISTENCIA
# ============================================================================
import joblib
import hashlib

LANDMARKER_CACHE_DIR = "./landmarker_cache"
os.makedirs(LANDMARKER_CACHE_DIR, exist_ok=True)

def _compute_cached_landmarkers(dataset, dataset_name, sample_size=500):
    """Landmarkers con caché y mejor consistencia."""
    
    # Crear identificador único más específico
    cache_id = f"{dataset_name}_{len(dataset)}_{sample_size}_{hash(str(dataset.__class__))}"
    cache_hash = hashlib.md5(cache_id.encode()).hexdigest()[:16]
    cache_file = os.path.join(LANDMARKER_CACHE_DIR, f"{cache_hash}.pkl")
    
    # Verificar caché
    if os.path.exists(cache_file):
        try:
            print(f"  [CACHE] Cargando landmarkers desde caché...")
            cached_data = joblib.load(cache_file)
            # Verificar que los datos cacheados tengan valores válidos
            if all(0 <= v <= 1 for v in cached_data.values() if isinstance(v, (int, float))):
                return cached_data
            else:
                print(f"  [CACHE] Datos cacheados inválidos, recalculando...")
        except:
            print(f"  [CACHE] Error cargando, recalculando...")
    
    # Calcular nuevos landmarkers
    print(f"  [CACHE] Calculando nuevos landmarkers...")
    results = _compute_robust_landmarkers(dataset, sample_size)
    
    # Guardar en caché
    try:
        joblib.dump(results, cache_file)
        print(f"  [CACHE] Guardados en {cache_file}")
    except:
        print(f"  [CACHE] Error guardando en caché")
    
    return results

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

def _compute_robust_pixel_stats(dataset, sample_size=300):
    """Estadísticas de píxeles robustas y normalizadas."""
    if len(dataset) == 0:
        return {'mean': 0, 'std': 0, 'median': 0, 'iqr': 0, 'skewness': 0, 
                'kurtosis': 0, 'gmean': 0, 'entropy': 0}
    
    pixel_samples = []
    count = min(sample_size, len(dataset))
    
    # Muestreo más consistente
    if len(dataset) > 10000:
        step = max(1, len(dataset) // count)
        indices = list(range(0, len(dataset), step))[:count]
    else:
        np.random.seed(42)  # Fijar semilla para consistencia
        indices = np.random.choice(len(dataset), count, replace=False)
    
    for idx in indices:
        try:
            img, _ = dataset[idx]
            if isinstance(img, Image.Image):
                img_array = np.array(img).astype(np.float32) / 255.0
            elif isinstance(img, torch.Tensor):
                img_array = img.numpy().astype(np.float32)
                if img_array.ndim > 2 and img_array.shape[0] in [1, 3, 4]:
                    # Reordenar a (H, W, C) si es necesario
                    img_array = np.transpose(img_array, (1, 2, 0))
            else:
                img_array = np.array(img).astype(np.float32)
            
            # Para imágenes a color, usar luminosidad
            if img_array.ndim > 2 and img_array.shape[-1] >= 3:
                # Convertir a luminosidad: Y = 0.299R + 0.587G + 0.114B
                img_array = 0.299 * img_array[..., 0] + 0.587 * img_array[..., 1] + 0.114 * img_array[..., 2]
            elif img_array.ndim > 2:
                img_array = img_array.mean(axis=-1)
            
            pixel_samples.append(img_array.flatten())
        except Exception as e:
            print(f"    [WARN] Error procesando imagen {idx}: {str(e)[:50]}")
            continue
    
    if not pixel_samples:
        return {'mean': 0, 'std': 0, 'median': 0, 'iqr': 0, 'skewness': 0, 
                'kurtosis': 0, 'gmean': 0, 'entropy': 0}
    
    # Combinar todas las muestras
    all_pixels = np.concatenate(pixel_samples)
    
    # Filtrar valores extremos para mayor robustez
    q1, q99 = np.percentile(all_pixels, [1, 99])
    filtered_pixels = all_pixels[(all_pixels >= q1) & (all_pixels <= q99)]
    
    if len(filtered_pixels) < 100:
        filtered_pixels = all_pixels
    
    # Estadísticas robustas
    q25, median, q75 = np.percentile(filtered_pixels, [25, 50, 75])
    
    return {
        'mean': float(np.mean(filtered_pixels)),
        'std': float(np.std(filtered_pixels)),
        'median': float(median),
        'iqr': float(q75 - q25),
        'skewness': float(skew(filtered_pixels)),
        'kurtosis': float(kurtosis(filtered_pixels)),
        'gmean': float(gmean(filtered_pixels + 1e-10)),
        'entropy': float(shannon_entropy(filtered_pixels))
    }

def _compute_consistent_color_features(dataset, sample_size=150):
    """Características de color consistentes."""
    if len(dataset) == 0:
        return {
            'color_entropy_mean': 0,
            'color_diversity': 0,
            'color_saturation': 0,
            'color_contrast': 0
        }
    
    color_stats = []
    count = min(sample_size, len(dataset))
    
    np.random.seed(42)
    indices = np.random.choice(len(dataset), count, replace=False)
    
    for idx in indices:
        try:
            img, _ = dataset[idx]
            if isinstance(img, Image.Image):
                img_array = np.array(img).astype(np.float32) / 255.0
            elif isinstance(img, torch.Tensor):
                img_array = img.numpy().astype(np.float32)
                if img_array.ndim > 2 and img_array.shape[0] in [1, 3, 4]:
                    img_array = np.transpose(img_array, (1, 2, 0))
            else:
                img_array = np.array(img).astype(np.float32)
            
            if img_array.ndim != 3 or img_array.shape[-1] < 3:
                # Para escala de grises, valores por defecto
                color_stats.append({
                    'entropy': 0.0,
                    'diversity': 0.0,
                    'saturation': 0.0,
                    'contrast': 0.0
                })
                continue
            
            # Asegurar 3 canales RGB
            if img_array.shape[-1] > 3:
                img_array = img_array[..., :3]
            
            # Calcular características por canal
            r_channel = img_array[..., 0]
            g_channel = img_array[..., 1]
            b_channel = img_array[..., 2]
            
            # Entropía de color
            hist_r, _ = np.histogram(r_channel.flatten(), bins=16, range=(0, 1))
            hist_g, _ = np.histogram(g_channel.flatten(), bins=16, range=(0, 1))
            hist_b, _ = np.histogram(b_channel.flatten(), bins=16, range=(0, 1))
            
            hist_r = hist_r / (hist_r.sum() + 1e-10)
            hist_g = hist_g / (hist_g.sum() + 1e-10)
            hist_b = hist_b / (hist_b.sum() + 1e-10)
            
            entropy_r = scipy_entropy(hist_r + 1e-10)
            entropy_g = scipy_entropy(hist_g + 1e-10)
            entropy_b = scipy_entropy(hist_b + 1e-10)
            
            # Diversidad de color (normalizada)
            unique_colors = len(np.unique(
                (img_array.reshape(-1, 3) * 16).astype(np.uint8), 
                axis=0
            ))
            color_diversity = min(unique_colors / 256, 1.0)
            
            # Saturación (variabilidad entre canales)
            channel_stds = [r_channel.std(), g_channel.std(), b_channel.std()]
            saturation = np.std(channel_stds) / (np.mean(channel_stds) + 1e-10)
            
            # Contraste de color
            color_contrast = np.std([r_channel.mean(), g_channel.mean(), b_channel.mean()])
            
            color_stats.append({
                'entropy': (entropy_r + entropy_g + entropy_b) / 3 / 4,  # Normalizado
                'diversity': color_diversity,
                'saturation': min(saturation, 1.0),
                'contrast': min(color_contrast, 1.0)
            })
        except Exception as e:
            print(f"    [WARN] Error en color features: {str(e)[:50]}")
            continue
    
    if len(color_stats) == 0:
        return {
            'color_entropy_mean': 0,
            'color_diversity': 0,
            'color_saturation': 0,
            'color_contrast': 0
        }
    
    # Promediar y normalizar
    return {
        'color_entropy_mean': float(np.mean([s['entropy'] for s in color_stats])),
        'color_diversity': float(np.mean([s['diversity'] for s in color_stats])),
        'color_saturation': float(np.mean([s['saturation'] for s in color_stats])),
        'color_contrast': float(np.mean([s['contrast'] for s in color_stats]))
    }

def _compute_consistent_texture_features(dataset, sample_size=80):
    """Características de textura consistentes."""
    if len(dataset) == 0:
        return {
            'texture_contrast': 0.5,
            'texture_homogeneity': 0.5,
            'texture_energy': 0.5,
            'texture_correlation': 0.5,
            'texture_complexity': 0.5
        }
    
    distances = [1]
    angles = [0, np.pi/4]
    
    features = {k: [] for k in ['contrast', 'homogeneity', 'energy', 'correlation']}
    count = min(sample_size, len(dataset))
    
    np.random.seed(42)
    indices = np.random.choice(len(dataset), count, replace=False)
    
    for idx in indices:
        try:
            img, _ = dataset[idx]
            
            # Convertir a escala de grises
            if isinstance(img, Image.Image):
                img_gray = np.array(img.convert('L'))
            elif isinstance(img, torch.Tensor):
                img_np = img.numpy()
                if img_np.ndim > 2:
                    img_gray = img_np.mean(axis=0)
                else:
                    img_gray = img_np
                img_gray = (img_gray * 255).astype(np.uint8)
            else:
                img_array = np.array(img)
                if img_array.ndim > 2:
                    img_gray = img_array.mean(axis=2)
                else:
                    img_gray = img_array
                img_gray = (img_gray * 255).astype(np.uint8)
            
            # Reducir niveles para consistencia
            img_gray = (img_gray // 32)  # Reducir a 8 niveles
            
            # GLCM
            glcm = graycomatrix(img_gray, distances=distances, angles=angles, 
                               levels=8, symmetric=True, normed=True)
            
            for prop in ['contrast', 'homogeneity', 'energy', 'correlation']:
                value = graycoprops(glcm, prop).mean()
                features[prop].append(float(value))
                
        except Exception as e:
            print(f"    [WARN] Error en textura: {str(e)[:50]}")
            continue
    
    # Calcular promedios con valores por defecto si no hay suficientes muestras
    texture_features = {}
    for prop in ['contrast', 'homogeneity', 'energy', 'correlation']:
        if features[prop]:
            avg_val = np.mean(features[prop])
            # Normalizar a [0, 1]
            if prop == 'contrast':
                norm_val = min(avg_val / 100, 1.0)
            elif prop == 'correlation':
                norm_val = (avg_val + 1) / 2  # De [-1, 1] a [0, 1]
            else:
                norm_val = min(max(avg_val, 0), 1)
            texture_features[f'texture_{prop}'] = float(norm_val)
        else:
            texture_features[f'texture_{prop}'] = 0.5
    
    # Complejidad de textura
    if 'texture_homogeneity' in texture_features:
        texture_features['texture_complexity'] = 1.0 - texture_features['texture_homogeneity']
    
    return texture_features

def _compute_edge_features(dataset, sample_size=80):
    """Características de bordes consistentes."""
    if len(dataset) == 0:
        return {
            'edge_density': 0.1,
            'edge_strength': 0.1,
            'shape_complexity': 0.5
        }
    
    edge_stats = []
    count = min(sample_size, len(dataset))
    
    np.random.seed(42)
    indices = np.random.choice(len(dataset), count, replace=False)
    
    for idx in indices:
        try:
            img, _ = dataset[idx]
            
            # Convertir a escala de grises
            if isinstance(img, Image.Image):
                img_gray = np.array(img.convert('L')).astype(np.float32) / 255.0
            elif isinstance(img, torch.Tensor):
                img_np = img.numpy()
                if img_np.ndim > 2:
                    img_gray = img_np.mean(axis=0).astype(np.float32)
                else:
                    img_gray = img_np.astype(np.float32)
                if img_gray.max() > 1:
                    img_gray = img_gray / 255.0
            else:
                img_array = np.array(img)
                if img_array.ndim > 2:
                    img_gray = img_array.mean(axis=2).astype(np.float32)
                else:
                    img_gray = img_array.astype(np.float32)
                if img_gray.max() > 1:
                    img_gray = img_gray / 255.0
            
            # Detección de bordes
            edges = sobel(img_gray)
            
            # Densidad de bordes
            edge_threshold = np.percentile(np.abs(edges.flatten()), 95)
            edge_density = np.mean(np.abs(edges) > edge_threshold)
            
            # Fuerza de bordes (normalizada)
            edge_strength = np.mean(np.abs(edges))
            
            # Complejidad de forma (entropía normalizada)
            shape_complexity = shannon_entropy(np.abs(edges) + 1e-10) / 8
            
            edge_stats.append({
                'density': min(edge_density, 0.5),
                'strength': min(edge_strength, 0.3),
                'complexity': min(shape_complexity, 1.0)
            })
        except Exception as e:
            print(f"    [WARN] Error en bordes: {str(e)[:50]}")
            continue
    
    if len(edge_stats) == 0:
        return {
            'edge_density': 0.1,
            'edge_strength': 0.1,
            'shape_complexity': 0.5
        }
    
    return {
        'edge_density': float(np.mean([s['density'] for s in edge_stats])),
        'edge_strength': float(np.mean([s['strength'] for s in edge_stats])),
        'shape_complexity': float(np.mean([s['complexity'] for s in edge_stats]))
    }

def _compute_robust_landmarkers(dataset, sample_size=800):
    """Landmarkers robustos y consistentes."""
    try:
        print(f"  [LANDMARKERS] Iniciando con sample_size={sample_size}...")
        
        # Reducir para datasets pequeños
        sample_size = min(sample_size, len(dataset))
        if sample_size < 100:
            sample_size = min(100, len(dataset))
        
        if len(dataset) < 30:
            print(f"  [WARN] Dataset muy pequeño: {len(dataset)}")
            return {k: 0.5 for k in ['landmarker_1nn', 'landmarker_lda', 
                                    'landmarker_tree', 'landmarker_svm', 'landmarker_nb']}
        
        # Muestreo estratificado si es posible
        try:
            targets = _safe_get_targets(dataset)
            if targets is not None and len(targets) > 0:
                unique_classes = np.unique(targets)
                if len(unique_classes) > 1:
                    # Muestreo estratificado por clase
                    samples_per_class = max(2, sample_size // len(unique_classes))
                    indices = []
                    
                    for cls in unique_classes:
                        class_indices = np.where(targets == cls)[0]
                        if len(class_indices) > samples_per_class:
                            np.random.seed(42)
                            selected = np.random.choice(class_indices, samples_per_class, replace=False)
                        else:
                            selected = class_indices
                        indices.extend(selected)
                    
                    indices = indices[:sample_size]
                else:
                    # Muestreo uniforme
                    indices = list(range(min(sample_size, len(dataset))))
            else:
                indices = list(range(min(sample_size, len(dataset))))
        except:
            indices = list(range(min(sample_size, len(dataset))))
        
        # Extracción de características simplificada pero consistente
        X_samples = []
        y_samples = []
        
        print(f"  [LANDMARKERS] Extrayendo {len(indices)} muestras...")
        
        for idx in indices:
            try:
                img, label = dataset[idx]
                
                # Conversión consistente
                if isinstance(img, Image.Image):
                    # Reducir a tamaño fijo
                    img_small = img.resize((32, 32), Image.Resampling.LANCZOS)
                    img_array = np.array(img_small.convert('L')).astype(np.float32) / 255.0
                elif isinstance(img, torch.Tensor):
                    img_np = img.numpy()
                    if img_np.ndim > 2:
                        # Promediar canales y redimensionar
                        if img_np.shape[0] in [1, 3]:
                            img_gray = img_np.mean(axis=0)
                        else:
                            img_gray = img_np.mean(axis=-1)
                    else:
                        img_gray = img_np
                    
                    # Redimensionar si es necesario
                    if img_gray.shape[0] > 32 or img_gray.shape[1] > 32:
                        from skimage.transform import resize
                        img_array = resize(img_gray, (32, 32), anti_aliasing=True)
                    else:
                        img_array = img_gray
                    
                    if img_array.max() > 1:
                        img_array = img_array / 255.0
                else:
                    img_array = np.array(img)
                    if img_array.ndim > 2:
                        img_array = img_array.mean(axis=2)
                    
                    if img_array.shape[0] > 32 or img_array.shape[1] > 32:
                        from skimage.transform import resize
                        img_array = resize(img_array, (32, 32), anti_aliasing=True)
                    
                    if img_array.max() > 1:
                        img_array = img_array / 255.0
                
                # Aplanar y normalizar
                features = img_array.flatten()
                
                # Normalización
                if features.std() > 0:
                    features = (features - features.mean()) / features.std()
                else:
                    features = features - features.mean()
                
                X_samples.append(features)
                y_samples.append(int(label))
            except Exception as e:
                continue
        
        print(f"  [LANDMARKERS] Extraídos {len(X_samples)} samples válidos...")
        
        # Verificar condiciones mínimas
        if len(X_samples) < 30 or len(set(y_samples)) < 2:
            print(f"  [WARN] Insuficientes muestras: {len(X_samples)} samples, {len(set(y_samples))} clases")
            return {k: 0.5 for k in ['landmarker_1nn', 'landmarker_lda', 
                                    'landmarker_tree', 'landmarker_svm', 'landmarker_nb']}
        
        X = np.array(X_samples)
        y = np.array(y_samples)
        
        # Reducir dimensionalidad si es necesario
        if X.shape[1] > 500:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(500, X.shape[0]), random_state=42)
            X = pca.fit_transform(X)
        
        # Para datasets con muchas clases, limitar
        unique_classes = np.unique(y)
        if len(unique_classes) > 50:
            # Tomar las 50 clases más frecuentes
            from collections import Counter
            class_counts = Counter(y)
            top_classes = [cls for cls, _ in class_counts.most_common(50)]
            mask = np.isin(y, top_classes)
            X = X[mask]
            y = y[mask]
        
        # Holdout estratificado
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42, shuffle=True
        )
        
        print(f"  [LANDMARKERS] Entrenando con {len(X_train)} train, {len(X_test)} test...")
        
        results = {}
        
        # Modelos con configuración consistente
        model_configs = [
            ('landmarker_1nn', KNeighborsClassifier(n_neighbors=1, n_jobs=-1)),
            ('landmarker_lda', LinearDiscriminantAnalysis()),
            ('landmarker_tree', DecisionTreeClassifier(max_depth=5, random_state=42)),
            ('landmarker_svm', LinearSVC(max_iter=1000, random_state=42, dual='auto', tol=1e-3)),
            ('landmarker_nb', GaussianNB()),
        ]
        
        for name, model in model_configs:
            try:
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                # Asegurar que el score esté en [0, 1]
                results[name] = float(np.clip(score, 0, 1))
                print(f"    {name}: {results[name]:.3f}")
            except Exception as e:
                print(f"    [ERROR] {name}: {str(e)[:50]}")
                results[name] = 0.5
        
        print(f"  [LANDMARKERS] Resultados: {results}")
        return results
        
    except Exception as e:
        print(f"  [ERROR CRÍTICO] en landmarkers: {str(e)[:100]}")
        return {k: 0.5 for k in ['landmarker_1nn', 'landmarker_lda', 
                                'landmarker_tree', 'landmarker_svm', 'landmarker_nb']}

def _compute_dataset_complexity(dataset):
    """Métricas de complejidad consistentes."""
    targets = _safe_get_targets(dataset)
    
    if targets is None or len(targets) == 0:
        return {
            'class_entropy': 0.5,
            'imbalance_ratio': 1.0,
            'n_classes': 0,
            'class_overlap_index': 0.5
        }
    
    unique, counts = np.unique(targets, return_counts=True)
    n_classes = len(unique)
    
    if n_classes <= 1:
        return {
            'class_entropy': 0,
            'imbalance_ratio': 1.0,
            'n_classes': n_classes,
            'class_overlap_index': 0
        }
    
    # Entropía de clases normalizada
    probs = counts / counts.sum()
    class_entropy = -np.sum(probs * np.log2(probs + 1e-10))
    max_entropy = np.log2(n_classes) if n_classes > 1 else 1
    normalized_entropy = class_entropy / max_entropy
    
    # Ratio de desbalanceo
    if counts.min() > 0:
        imbalance_ratio = counts.max() / counts.min()
        log_imbalance = np.log1p(imbalance_ratio - 1) / 5  # Normalizado
    else:
        log_imbalance = 1.0
    
    # Índice de superposición (aproximado)
    overlap_index = min(n_classes / 50, 1.0)
    
    return {
        'class_entropy': float(normalized_entropy),
        'imbalance_ratio': float(log_imbalance),
        'n_classes': int(n_classes),
        'class_overlap_index': float(overlap_index)
    }

def _compute_spatial_features(dataset, sample_size=80):
    """Características espaciales consistentes."""
    if len(dataset) == 0:
        return {
            'spatial_compactness': 0.5,
            'aspect_ratio_variation': 0.5
        }
    
    spatial_stats = []
    count = min(sample_size, len(dataset))
    
    np.random.seed(42)
    indices = np.random.choice(len(dataset), count, replace=False)
    
    for idx in indices:
        try:
            img, _ = dataset[idx]
            
            # Convertir a escala de grises
            if isinstance(img, Image.Image):
                img_array = np.array(img.convert('L')).astype(np.float32) / 255.0
            elif isinstance(img, torch.Tensor):
                img_np = img.numpy()
                if img_np.ndim > 2:
                    img_array = img_np.mean(axis=0).astype(np.float32)
                else:
                    img_array = img_np.astype(np.float32)
                if img_array.max() > 1:
                    img_array = img_array / 255.0
            else:
                img_array = np.array(img)
                if img_array.ndim > 2:
                    img_array = img_array.mean(axis=2).astype(np.float32)
                else:
                    img_array = img_array.astype(np.float32)
                if img_array.max() > 1:
                    img_array = img_array / 255.0
            
            # Binarizar
            threshold = np.percentile(img_array.flatten(), 50)
            binary = img_array > threshold
            
            if np.any(binary):
                rows = np.any(binary, axis=1)
                cols = np.any(binary, axis=0)
                
                if np.any(rows) and np.any(cols):
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]
                    
                    height = rmax - rmin + 1
                    width = cmax - cmin + 1
                    
                    # Compacidad
                    object_area = np.sum(binary)
                    bbox_area = height * width
                    compactness = object_area / bbox_area if bbox_area > 0 else 0
                    
                    # Ratio de aspecto (log para normalizar)
                    aspect_ratio = width / max(height, 1)
                    log_aspect = np.log1p(abs(aspect_ratio - 1))
                    
                    spatial_stats.append({
                        'compactness': min(compactness, 1.0),
                        'aspect_ratio': min(log_aspect, 2.0) / 2
                    })
        except Exception as e:
            print(f"    [WARN] Error en características espaciales: {str(e)[:50]}")
            continue
    
    if len(spatial_stats) == 0:
        return {
            'spatial_compactness': 0.5,
            'aspect_ratio_variation': 0.5
        }
    
    compactness_vals = [s['compactness'] for s in spatial_stats]
    aspect_ratios = [s['aspect_ratio'] for s in spatial_stats]
    
    return {
        'spatial_compactness': float(np.mean(compactness_vals)),
        'aspect_ratio_variation': float(np.std(aspect_ratios))
    }

def _normalize_features(features_dict):
    """Normaliza características a rangos apropiados."""
    normalized = features_dict.copy()
    
    # Características que SÍ deben estar en [0, 1]
    zero_one_features = [
        'landmarker_1nn', 'landmarker_lda', 'landmarker_tree', 
        'landmarker_svm', 'landmarker_nb',
        'texture_contrast', 'texture_homogeneity', 'texture_energy',
        'texture_correlation', 'texture_complexity',
        'color_entropy_mean', 'color_diversity', 'color_saturation', 'color_contrast',
        'edge_density', 'edge_strength', 'shape_complexity',
        'spatial_compactness', 'aspect_ratio_variation',
        'class_entropy', 'imbalance_ratio', 'class_overlap_index',
        'dataset_difficulty', 'visual_variability', 'structural_regularity'
    ]
    
    # Características que NO deben normalizarse a [0, 1]
    non_normalized_features = [
        'pixel_skewness', 'pixel_kurtosis', 'pixel_entropy',
        'pixel_mean', 'pixel_std', 'pixel_median', 'pixel_iqr', 'pixel_gmean',
        'n_classes', 'channels', 'num_train', 'total_size_mb'
    ]
    
    # Normalizar solo las que deben estar en [0, 1]
    for key in zero_one_features:
        if key in normalized and isinstance(normalized[key], (int, float)):
            normalized[key] = float(np.clip(normalized[key], 0, 1))
    
    # Para características estadísticas, aplicar transformación logarítmica si es necesario
    stats_features = ['pixel_skewness', 'pixel_kurtosis', 'pixel_entropy']
    for key in stats_features:
        if key in normalized:
            value = normalized[key]
            if key == 'pixel_skewness':
                # Skewness puede ser negativo o positivo grande
                # Transformar a [0, 1] usando tanh
                normalized[key] = float((np.tanh(value/5) + 1) / 2)
            elif key == 'pixel_kurtosis':
                # Kurtosis es >= 1, normal para distribución normal = 3
                # Transformar: kurtosis -> [0, 1]
                normalized[key] = float(np.clip(value / 10, 0, 1))
            elif key == 'pixel_entropy':
                # Entropía en bits, para imágenes 8-bit max ~8
                normalized[key] = float(np.clip(value / 8, 0, 1))
    
    return normalized

def _compute_meta_features(features_dict):
    """Características meta derivadas."""
    meta_features = {}
    
    # Dificultad del dataset (basado en landmarkers)
    landmarker_keys = ['landmarker_1nn', 'landmarker_lda', 'landmarker_svm']
    if all(k in features_dict for k in landmarker_keys):
        scores = [features_dict[k] for k in landmarker_keys]
        # Excluir valores 0.5 (default)
        valid_scores = [s for s in scores if s != 0.5]
        if valid_scores:
            avg_score = np.mean(valid_scores)
            meta_features['dataset_difficulty'] = float(1.0 - avg_score)
        else:
            meta_features['dataset_difficulty'] = 0.5
    
    # Variabilidad visual
    visual_keys = ['color_diversity', 'texture_complexity', 'edge_density']
    if all(k in features_dict for k in visual_keys):
        visual_components = [
            features_dict['color_diversity'],
            features_dict['texture_complexity'],
            features_dict['edge_density']
        ]
        meta_features['visual_variability'] = float(np.mean(visual_components))
    
    # Regularidad estructural
    if 'spatial_compactness' in features_dict:
        meta_features['structural_regularity'] = float(features_dict['spatial_compactness'])
    
    # Tipo de dataset aproximado
    if 'channels' in features_dict:
        channels = features_dict['channels']
        if isinstance(channels, (int, float)):
            if channels == 1:
                meta_features['dataset_type'] = 'grayscale'
            elif channels == 3:
                meta_features['dataset_type'] = 'color'
            else:
                meta_features['dataset_type'] = 'other'
    
    return meta_features

# ============================================================================
# FUNCIÓN PRINCIPAL MEJORADA
# ============================================================================

def extract_metafeatures_for_metalearning(name, dataset_class, train_kwargs, test_kwargs):
    """Extracción optimizada y consistente de metafeatures."""
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
        
        print("  [1/8] Cargando dataset...")
        train_dataset = dataset_class(**train_kwargs_full)
        
        num_train = len(train_dataset)
        if num_train == 0:
            raise ValueError(f"Dataset {name} vacío")
        
        # Información básica
        print("  [2/8] Obteniendo información básica...")
        if len(train_dataset) > 0:
            img, _ = train_dataset[0]
            if isinstance(img, Image.Image):
                resolution = f"{img.size[0]}x{img.size[1]}"
                channels = len(img.getbands()) if hasattr(img, 'getbands') else 1
            elif isinstance(img, torch.Tensor):
                if img.ndim > 2:
                    resolution = f"{img.shape[1]}x{img.shape[2]}"
                    channels = img.shape[0]
                else:
                    resolution = f"{img.shape[0]}x{img.shape[1]}"
                    channels = 1
            else:
                img_array = np.array(img)
                if img_array.ndim > 2:
                    resolution = f"{img_array.shape[0]}x{img_array.shape[1]}"
                    channels = img_array.shape[2]
                else:
                    resolution = f"{img_array.shape[0]}x{img_array.shape[1]}"
                    channels = 1
        else:
            resolution = 'N/A'
            channels = 'N/A'
        
        # Calcular tamaño estimado
        if len(train_dataset) > 0:
            img_sample, _ = train_dataset[0]
            if isinstance(img_sample, Image.Image):
                img_size = img_sample.size[0] * img_sample.size[1] * channels
            elif isinstance(img_sample, torch.Tensor):
                img_size = img_sample.numel()
            else:
                img_size = np.array(img_sample).size
            
            total_size_mb = (img_size * 4 * num_train) / (1024 ** 2)
        else:
            total_size_mb = 0
        
        # Calcular metafeatures
        print("  [3/8] Calculando estadísticas de píxeles...")
        pixel_stats = _compute_robust_pixel_stats(train_dataset, SAMPLE_CONFIG['pixel_stats'])
        
        print("  [4/8] Calculando complejidad del dataset...")
        complexity_metrics = _compute_dataset_complexity(train_dataset)
        
        print("  [5/8] Calculando características de color...")
        color_features = _compute_consistent_color_features(train_dataset, SAMPLE_CONFIG['color_features'])
        
        print("  [6/8] Calculando características de textura...")
        texture_features = _compute_consistent_texture_features(train_dataset, SAMPLE_CONFIG['texture_features'])
        
        print("  [7/8] Calculando características de bordes...")
        edge_features = _compute_edge_features(train_dataset, SAMPLE_CONFIG['edge_features'])
        
        print("  [8/8] Calculando landmarkers...")
        landmarkers = _compute_cached_landmarkers(train_dataset, name, sample_size=SAMPLE_CONFIG['landmarkers'])
        
        print("  [9/8] Calculando características espaciales...")
        spatial_features = _compute_spatial_features(train_dataset, SAMPLE_CONFIG['edge_features'])
        
        # Compilar resultados
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
            
            # Complejidad
            **complexity_metrics,
            
            # Características visuales
            **color_features,
            **texture_features,
            **edge_features,
            **spatial_features,
            
            # Landmarkers
            **landmarkers,
        }
        
        # Añadir características meta
        meta_features = _compute_meta_features(all_features)
        all_features.update(meta_features)
        
        # Normalizar
        all_features = _normalize_features(all_features)
        
        print(f"  ✓ Completado: {len(all_features)} metafeatures extraídos")
        return all_features
        
    except Exception as e:
        print(f"  ✗ Error: {str(e)[:200]}")
        import traceback
        traceback.print_exc()
        return {'dataset': name, 'error': str(e)}

# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal de ejecución."""
    print("=" * 70)
    print("EXTRACCIÓN DE METAFEATURES (VERSIÓN CONSISTENTE)")
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
        # Ordenar columnas
        field_order = [
            'dataset',
            'landmarker_1nn', 'landmarker_lda', 'landmarker_svm', 
            'landmarker_tree', 'landmarker_nb',
            'dataset_difficulty', 'visual_variability', 'structural_regularity',
            'dataset_type',
            'class_entropy', 'imbalance_ratio', 'class_overlap_index', 'n_classes',
            'color_entropy_mean', 'color_diversity', 'color_saturation', 'color_contrast',
            'texture_contrast', 'texture_homogeneity', 'texture_energy', 
            'texture_correlation', 'texture_complexity',
            'edge_density', 'edge_strength', 'shape_complexity',
            'spatial_compactness', 'aspect_ratio_variation',
            'num_train', 'channels', 'resolution', 'total_size_mb',
            'pixel_mean', 'pixel_std', 'pixel_entropy',
            'pixel_skewness', 'pixel_kurtosis', 'pixel_gmean', 'pixel_iqr', 'pixel_median'
        ]
        
        # Filtrar campos existentes
        existing_fields = []
        for field in field_order:
            if any(field in result for result in results):
                existing_fields.append(field)
        
        # Añadir campos adicionales
        all_keys = set()
        for result in results:
            all_keys.update(result.keys())
        
        additional_fields = sorted(list(all_keys - set(existing_fields)))
        fieldnames = existing_fields + additional_fields
        
        # Guardar CSV principal
        output_file = 'metafeatures_consistent.csv'
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                row = {field: result.get(field, 'N/A') for field in fieldnames}
                writer.writerow(row)
        
        # Guardar CSV simplificado
        key_features = [
            'dataset', 'dataset_difficulty', 'visual_variability', 
            'structural_regularity', 'landmarker_1nn', 'landmarker_lda',
            'landmarker_svm', 'n_classes', 'channels', 'class_entropy'
        ]
        
        with open('metafeatures_key_consistent.csv', 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=key_features)
            writer.writeheader()
            for result in results:
                row = {field: result.get(field, 'N/A') for field in key_features}
                writer.writerow(row)
        
        # Resumen estadístico
        print(f"\n{'='*70}")
        print("RESUMEN ESTADÍSTICO")
        print("=" * 70)
        
        # Estadísticas de landmarkers
        landmarker_cols = [c for c in fieldnames if c.startswith('landmarker_')]
        if landmarker_cols:
            print("\nAccuracy de Landmarkers:")
            for col in landmarker_cols:
                values = [r.get(col, 0) for r in results if isinstance(r.get(col), (int, float))]
                if values:
                    print(f"  {col}: min={min(values):.3f}, avg={np.mean(values):.3f}, max={max(values):.3f}")
        
        # Mostrar consistencia
        print("\nVerificación de consistencia:")
        inconsistent = []
        for result in results:
            dataset_name = result.get('dataset', 'Unknown')
            # Verificar que los valores estén en rangos razonables
            for key, value in result.items():
                if isinstance(value, (int, float)) and key != 'error':
                    if value < 0 or value > 1:
                        # Algunas características pueden estar fuera de [0,1]
                        if key not in ['num_train', 'total_size_mb', 'n_classes', 'channels']:
                            inconsistent.append((dataset_name, key, value))
        
        if inconsistent:
            print(f"  Se encontraron {len(inconsistent)} valores inconsistentes")
            for ds, key, val in inconsistent[:5]:  # Mostrar solo los primeros 5
                print(f"    {ds}.{key}: {val}")
        else:
            print("  ✓ Todos los valores están en rangos consistentes")
        
        print(f"\n{'='*70}")
        print(f"¡PROCESO COMPLETADO EXITOSAMENTE!")
        print(f"Metafeatures guardados en: {output_file}")
        print(f"Total de datasets procesados: {len(results)}")
        print("=" * 70)
    
    else:
        print("No se encontraron datasets para procesar")

if __name__ == "__main__":
    main()