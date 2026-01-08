import csv
import os
import numpy as np
import torch
from PIL import Image
import torchvision.datasets as datasets
import pymfe
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
from pymfe.mfe import MFE

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

# Lista de datasets ya descargados
datasets_list = [
    ('MNIST', datasets.MNIST, {'train': True}, {'train': False}),
    ('FashionMNIST', datasets.FashionMNIST, {'train': True}, {'train': False}),
    ('EMNIST', datasets.EMNIST, {'split': 'letters', 'train': True}, {'split': 'letters', 'train': False}),
    ('CIFAR10', datasets.CIFAR10, {'train': True}, {'train': False}),
    ('CIFAR100', datasets.CIFAR100, {'train': True}, {'train': False}),
    ('SVHN', datasets.SVHN, {'split': 'train'}, {'split': 'test'}),
    ('USPS', datasets.USPS, {'train': True}, {'train': False}),
    ('Omniglot', datasets.Omniglot, {}, {}),
    ('EuroSAT', datasets.EuroSAT, {}, {}),
    ('Flowers102', datasets.Flowers102, {'split': 'train'}, {'split': 'test'}),
    ('DTD', datasets.DTD, {'split': 'train'}, {'split': 'test'}),
    ('GTSRB', datasets.GTSRB, {'split': 'train'}, {'split': 'test'}),
    ('OxfordIIITPet', datasets.OxfordIIITPet, {'split': 'trainval'}, {'split': 'test'}),
]

# Filtrar datasets disponibles
available_datasets = []
for name, dataset_class, train_kwargs, test_kwargs in datasets_list:
    folder = folder_names.get(name, name)
    root = f'./data/{folder}'
    if os.path.exists(root):
        available_datasets.append((name, dataset_class, train_kwargs, test_kwargs))

def _compute_pixel_stats(dataset, sample_size=10):
    if len(dataset) == 0:
        return {'mean': 'N/A', 'std': 'N/A'}
    
    all_pixels = []
    count = min(sample_size, len(dataset))
    
    for i in range(count):
        img, _ = dataset[i]
        if isinstance(img, Image.Image):
            img_array = np.array(img).astype(np.float32) / 255.0
        elif isinstance(img, torch.Tensor):
            img_array = img.numpy()
        else:
            img_array = np.array(img)
        
        all_pixels.extend(img_array.flatten())
    
    overall_mean = np.mean(all_pixels)
    overall_std = np.std(all_pixels)
    
    return {
        'mean': overall_mean,
        'std': overall_std
    }

def _compute_class_distribution(dataset):
    if not hasattr(dataset, 'targets') or dataset.targets is None or len(dataset.targets) == 0:
        return 'N/A'
    
    targets = np.array(dataset.targets)
    unique, counts = np.unique(targets, return_counts=True)
    distribution = dict(zip(unique, counts))
    return distribution

def _estimate_dataset_size(dataset, sample_size=10):
    if len(dataset) == 0:
        return 'N/A'
    
    total_size = 0
    count = min(sample_size, len(dataset))
    
    for i in range(count):
        img, _ = dataset[i]
        if isinstance(img, Image.Image):
            total_size += img.size[0] * img.size[1] * len(img.getbands()) * 3
        elif isinstance(img, torch.Tensor):
            total_size += img.numel() * 4
        else:
            total_size += np.array(img).nbytes
    
    avg_size_per_img = total_size / count
    estimated_total = avg_size_per_img * len(dataset) / (1024 ** 2)
    return round(estimated_total, 2)

def _compute_texture_features(dataset, sample_size=5):
    if len(dataset) == 0:
        return {'contrast': 'N/A', 'dissimilarity': 'N/A', 'homogeneity': 'N/A', 'energy': 'N/A', 'correlation': 'N/A'}
    
    features = {'contrast': [], 'dissimilarity': [], 'homogeneity': [], 'energy': [], 'correlation': []}
    count = min(sample_size, len(dataset))
    
    for i in range(count):
        img, _ = dataset[i]
        if isinstance(img, Image.Image):
            img_array = np.array(img.convert('L')).astype(np.uint8)  # Convert to grayscale
        elif isinstance(img, torch.Tensor):
            img_array = img.mean(dim=0).numpy().astype(np.uint8)  # Average channels
        else:
            img_array = np.array(img).astype(np.uint8)
        
        # Compute GLCM
        glcm = graycomatrix(img_array, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        features['contrast'].append(graycoprops(glcm, 'contrast')[0, 0])
        features['dissimilarity'].append(graycoprops(glcm, 'dissimilarity')[0, 0])
        features['homogeneity'].append(graycoprops(glcm, 'homogeneity')[0, 0])
        features['energy'].append(graycoprops(glcm, 'energy')[0, 0])
        features['correlation'].append(graycoprops(glcm, 'correlation')[0, 0])
    
    # Average over samples
    return {k: np.mean(v) for k, v in features.items()}

def _compute_advanced_pixel_stats(dataset, sample_size=10):
    if len(dataset) == 0:
        return {'skewness': 'N/A', 'kurtosis': 'N/A'}
    
    all_pixels = []
    count = min(sample_size, len(dataset))
    
    for i in range(count):
        img, _ = dataset[i]
        if isinstance(img, Image.Image):
            img_array = np.array(img).astype(np.float32) / 255.0
        elif isinstance(img, torch.Tensor):
            img_array = img.numpy()
        else:
            img_array = np.array(img)
        
        all_pixels.extend(img_array.flatten())
    
    return {
        'skewness': skew(all_pixels),
        'kurtosis': kurtosis(all_pixels)
    }

def _compute_pymfe_features(dataset):
    if not hasattr(dataset, 'targets') or dataset.targets is None or len(dataset.targets) == 0:
        return {'entropy': 'N/A', 'imbalance_ratio': 'N/A'}
    
    targets = np.array(dataset.targets)
    # For pymfe, we need X (dummy) and y
    X_dummy = np.random.rand(len(targets), 1)  # Dummy features
    y = targets
    
    mfe = MFE(features=['class_ent', 'tree_imbalance'])
    mfe.fit(X_dummy, y)
    ft = mfe.extract()
    
    # Extract specific features
    features = {}
    if 'class_ent' in ft[0]:
        idx = ft[0].index('class_ent')
        features['entropy'] = ft[1][idx]
    else:
        features['entropy'] = 'N/A'
    
    if 'tree_imbalance' in ft[0]:
        idx = ft[0].index('tree_imbalance')
        features['imbalance_ratio'] = ft[1][idx]
    else:
        features['imbalance_ratio'] = 'N/A'
    
    return features

def extract_metafeatures(name, dataset_class, train_kwargs, test_kwargs):
    try:
        folder = folder_names.get(name, name)
        root = f'./data/{folder}'
        
        # Instanciar sin download (ya descargados)
        train_kwargs_full = train_kwargs.copy()
        train_kwargs_full['root'] = root
        train_kwargs_full['download'] = False
        train_dataset = dataset_class(**train_kwargs_full)
        
        if test_kwargs:
            try:
                test_kwargs_full = test_kwargs.copy()
                test_kwargs_full['root'] = root
                test_kwargs_full['download'] = False
                test_dataset = dataset_class(**test_kwargs_full)
                num_test = len(test_dataset)
            except Exception:
                num_test = 'N/A'
        else:
            num_test = 'N/A'
        
        num_classes = len(train_dataset.classes) if hasattr(train_dataset, 'classes') else 'N/A'
        num_train = len(train_dataset)
        
        if len(train_dataset) > 0:
            img, _ = train_dataset[0]
            if isinstance(img, Image.Image):
                resolution = f"{img.size[0]}x{img.size[1]}"
                channels = len(img.getbands()) if hasattr(img, 'getbands') else 1
            else:
                resolution = str(img.shape[1:])
                channels = img.shape[0] if len(img.shape) > 2 else 1
        else:
            resolution = 'N/A'
            channels = 'N/A'
        
        pixel_stats = _compute_pixel_stats(train_dataset)
        class_distribution = _compute_class_distribution(train_dataset)
        total_size_mb = _estimate_dataset_size(train_dataset)
        texture_features = _compute_texture_features(train_dataset)
        advanced_pixel_stats = _compute_advanced_pixel_stats(train_dataset)
        pymfe_features = _compute_pymfe_features(train_dataset)
        
        return {
            'num_classes': num_classes,
            'num_train': num_train,
            'num_test': num_test,
            'resolution': resolution,
            'channels': channels,
            'pixel_mean': pixel_stats['mean'],
            'pixel_std': pixel_stats['std'],
            'pixel_skewness': advanced_pixel_stats['skewness'],
            'pixel_kurtosis': advanced_pixel_stats['kurtosis'],
            'texture_contrast': texture_features['contrast'],
            'texture_dissimilarity': texture_features['dissimilarity'],
            'texture_homogeneity': texture_features['homogeneity'],
            'texture_energy': texture_features['energy'],
            'texture_correlation': texture_features['correlation'],
            'class_entropy': pymfe_features['entropy'],
            'class_imbalance_ratio': pymfe_features['imbalance_ratio'],
            'class_distribution': class_distribution,
            'total_size_mb': total_size_mb,
            'error': None
        }
    except Exception as e:
        return {
            'num_classes': 'Error',
            'num_train': 'Error',
            'num_test': 'Error',
            'resolution': 'Error',
            'channels': 'Error',
            'pixel_mean': 'Error',
            'pixel_std': 'Error',
            'pixel_skewness': 'Error',
            'pixel_kurtosis': 'Error',
            'texture_contrast': 'Error',
            'texture_dissimilarity': 'Error',
            'texture_homogeneity': 'Error',
            'texture_energy': 'Error',
            'texture_correlation': 'Error',
            'class_entropy': 'Error',
            'class_imbalance_ratio': 'Error',
            'class_distribution': 'Error',
            'total_size_mb': 'Error',
            'error': str(e)
        }

# Procesar
results = []
for name, dataset_class, train_kwargs, test_kwargs in available_datasets:
    print(f"Procesando {name}...")
    meta = extract_metafeatures(name, dataset_class, train_kwargs, test_kwargs)
    results.append({
        'Dataset': name,
        'Num_Classes': meta['num_classes'],
        'Num_Train': meta['num_train'],
        'Num_Test': meta['num_test'],
        'Resolution': meta['resolution'],
        'Channels': meta['channels'],
        'Pixel_Mean': meta['pixel_mean'],
        'Pixel_Std': meta['pixel_std'],
        'Pixel_Skewness': meta['pixel_skewness'],
        'Pixel_Kurtosis': meta['pixel_kurtosis'],
        'Texture_Contrast': meta['texture_contrast'],
        'Texture_Dissimilarity': meta['texture_dissimilarity'],
        'Texture_Homogeneity': meta['texture_homogeneity'],
        'Texture_Energy': meta['texture_energy'],
        'Texture_Correlation': meta['texture_correlation'],
        'Class_Entropy': meta['class_entropy'],
        'Class_Imbalance_Ratio': meta['class_imbalance_ratio'],
        'Class_Distribution': meta['class_distribution'],
        'Total_Size_MB': meta['total_size_mb'],
        'Error': meta['error']
    })

# Escribir CSV
with open('metafeatures_descargados.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Dataset', 'Num_Classes', 'Num_Train', 'Num_Test', 'Resolution', 'Channels', 'Pixel_Mean', 'Pixel_Std', 'Pixel_Skewness', 'Pixel_Kurtosis', 'Texture_Contrast', 'Texture_Dissimilarity', 'Texture_Homogeneity', 'Texture_Energy', 'Texture_Correlation', 'Class_Entropy', 'Class_Imbalance_Ratio', 'Class_Distribution', 'Total_Size_MB', 'Error']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print("Proceso completado. Metadatos guardados en metafeatures_descargados.csv")