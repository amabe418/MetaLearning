import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

# ============================================================================
# CONFIGURACIÓN DE DATASETS (Solución 1: Diccionario)
# ============================================================================
DATASET_CONFIG = {
    # ========================================================================
    # ESCALA DE GRISES (9)
    # ========================================================================
    'MNIST': {
        'loader': lambda root: datasets.MNIST(root, train=True, download=True),
        'test_loader': lambda root: datasets. MNIST(root, train=False, download=True),
        'num_classes': 10,
        'grayscale': True
    },
    'FashionMNIST': {
        'loader': lambda root: datasets. FashionMNIST(root, train=True, download=True),
        'test_loader': lambda root: datasets.  FashionMNIST(root, train=False, download=True),
        'num_classes': 10,
        'grayscale': True
    },
    'KMNIST': {
        'loader': lambda root:  datasets.KMNIST(root, train=True, download=True),
        'test_loader':  lambda root: datasets.KMNIST(root, train=False, download=True),
        'num_classes': 10,
        'grayscale': True
    },
    'EMNIST': {
        'loader': lambda root: datasets.EMNIST(root, split='letters', train=True, download=True),
        'test_loader':  lambda root: datasets.EMNIST(root, split='letters', train=False, download=True),
        'num_classes': 26,
        'grayscale': True
    },
    'USPS': {
        'loader': lambda root:   datasets.USPS(root, train=True, download=True),
        'test_loader': lambda root: datasets.USPS(root, train=False, download=True),
        'num_classes': 10,
        'grayscale':  True
    },
    'QMNIST': {
        'loader': lambda root: datasets. QMNIST(root, train=True, download=True),
        'test_loader': lambda root:  datasets.QMNIST(root, train=False, download=True),
        'num_classes':  10,
        'grayscale': True
    },
    'Omniglot': {
        'loader': lambda root:  datasets.Omniglot(root, background=True, download=True),
        'test_loader': lambda root: datasets.  Omniglot(root, background=False, download=True),
        'num_classes': 1623,
        'grayscale':  True
    },
    'RenderedSST2': {
        'loader': lambda root: datasets.RenderedSST2(root, split='train', download=True),
        'test_loader': lambda root: datasets.RenderedSST2(root, split='test', download=True),
        'num_classes': 2,
        'grayscale': True
    },
    'FER2013': {
        'loader': lambda root: datasets.FER2013(root, split='train', download=True),
        'test_loader': lambda root: datasets.FER2013(root, split='test', download=True),
        'num_classes': 7,
        'grayscale': True
    },
    
    # ========================================================================
    # RGB (21)
    # ========================================================================
    'CIFAR10': {
        'loader': lambda root: datasets. CIFAR10(root, train=True, download=True),
        'test_loader': lambda root:  datasets.  CIFAR10(root, train=False, download=True),
        'num_classes': 10,
        'grayscale':  False
    },
    'CIFAR100': {
        'loader': lambda root:   datasets.CIFAR100(root, train=True, download=True),
        'test_loader': lambda root: datasets. CIFAR100(root, train=False, download=True),
        'num_classes': 100,
        'grayscale': False
    },
    'SVHN': {
        'loader': lambda root: datasets.SVHN(root, split='train', download=True),
        'test_loader': lambda root: datasets.SVHN(root, split='test', download=True),
        'num_classes': 10,
        'grayscale': False
    },
    'STL10':  {
        'loader': lambda root: datasets.  STL10(root, split='train', download=True),
        'test_loader': lambda root:   datasets. STL10(root, split='test', download=True),
        'num_classes': 10,
        'grayscale': False
    },
    'Flowers102': {
        'loader':  lambda root: datasets.Flowers102(root, split='train', download=True),
        'test_loader': lambda root: datasets. Flowers102(root, split='test', download=True),
        'num_classes': 102,
        'grayscale': False
    },
    'OxfordIIITPet': {
        'loader': lambda root:  datasets.OxfordIIITPet(root, split='trainval', download=True),
        'test_loader': lambda root:  datasets.OxfordIIITPet(root, split='test', download=True),
        'num_classes': 37,
        'grayscale': False
    },
    'DTD':  {
        'loader': lambda root: datasets.DTD(root, split='train', download=True),
        'test_loader': lambda root: datasets.DTD(root, split='test', download=True),
        'num_classes':  47,
        'grayscale':  False
    },
    'GTSRB': {
        'loader': lambda root: datasets.  GTSRB(root, split='train', download=True),
        'test_loader': lambda root: datasets. GTSRB(root, split='test', download=True),
        'num_classes': 43,
        'grayscale': False
    },
    'EuroSAT': {
        'loader': lambda root: datasets.EuroSAT(root, download=True),
        'test_loader': lambda root: datasets.EuroSAT(root, download=True),
        'num_classes': 10,
        'grayscale':  False,
        'needs_split': True,
        'test_size': 0.2
    },
    'PCAM': {
        'loader': lambda root: datasets.PCAM(root, split='train', download=True),
        'test_loader': lambda root: datasets.  PCAM(root, split='test', download=True),
        'num_classes': 2,
        'grayscale':  False
    },
    'StanfordCars': {
        'loader': lambda root: datasets. StanfordCars(root, split='train', download=True),
        'test_loader': lambda root: datasets.StanfordCars(root, split='test', download=True),
        'num_classes': 196,
        'grayscale': False
    },
    'FGVCAircraft': {
        'loader': lambda root: datasets.FGVCAircraft(root, split='train', download=True),
        'test_loader': lambda root: datasets.FGVCAircraft(root, split='test', download=True),
        'num_classes': 100,
        'grayscale': False
    },
    'Country211': {
        'loader': lambda root:  datasets.Country211(root, split='train', download=True),
        'test_loader': lambda root: datasets.Country211(root, split='test', download=True),
        'num_classes': 211,
        'grayscale':  False
    },
    'Caltech101': {
        'loader': lambda root: datasets. Caltech101(root, download=True),
        'test_loader': lambda root: datasets.  Caltech101(root, download=True),
        'num_classes': 101,
        'grayscale': False,
        'needs_split': True,
        'test_size': 0.2
    },
    'LFWPeople': {
        'loader': lambda root: datasets. LFWPeople(root, split='train', download=True),
        'test_loader': lambda root: datasets.LFWPeople(root, split='test', download=True),
        'num_classes': 5749,
        'grayscale':  False
    },
    
    # Añadir 6 más (con split manual para llegar a 30)
    'EMNIST_Balanced': {
        'loader': lambda root: datasets.EMNIST(root, split='balanced', train=True, download=True),
        'test_loader': lambda root: datasets.EMNIST(root, split='balanced', train=False, download=True),
        'num_classes': 47,
        'grayscale':  True
    },
    'EMNIST_Digits': {
        'loader': lambda root: datasets.EMNIST(root, split='digits', train=True, download=True),
        'test_loader': lambda root: datasets.EMNIST(root, split='digits', train=False, download=True),
        'num_classes': 10,
        'grayscale': True
    },
    'EMNIST_MNIST': {
        'loader': lambda root: datasets.  EMNIST(root, split='mnist', train=True, download=True),
        'test_loader': lambda root: datasets. EMNIST(root, split='mnist', train=False, download=True),
        'num_classes': 10,
        'grayscale': True
    },
    'EMNIST_ByClass': {
        'loader':  lambda root: datasets.EMNIST(root, split='byclass', train=True, download=True),
        'test_loader':  lambda root: datasets.EMNIST(root, split='byclass', train=False, download=True),
        'num_classes':  62,
        'grayscale':  True
    },
    'EMNIST_ByMerge': {
        'loader':  lambda root: datasets.EMNIST(root, split='bymerge', train=True, download=True),
        'test_loader': lambda root: datasets. EMNIST(root, split='bymerge', train=False, download=True),
        'num_classes': 47,
        'grayscale':  True
    },
    'STL10_Unlabeled': {
        'loader': lambda root: datasets. STL10(root, split='train+unlabeled', download=True),
        'test_loader': lambda root: datasets.STL10(root, split='test', download=True),
        'num_classes': 10,
        'grayscale': False,
        'needs_split': True,
        'test_size': 0.2
    },
}

# ============================================================================
# CONFIGURACIONES DE MODELOS (Iguales para TODOS los datasets)
# ============================================================================
# ============================================================================
# CONFIGURACIONES DE MODELOS (8 hiperparámetros numéricos, 2 por algoritmo)
# ============================================================================
configs = [
    # ResNet18 (2 configuraciones)
    {
        'architecture': 'ResNet18',
        'learning_rate': 0.01,           # 1.  Learning rate
        'batch_size':  64,                # 2. Batch size
        'weight_decay': 0.0001,          # 3. Weight decay (L2 regularization)
        'momentum': 0.0,                 # 4. Momentum (0 para Adam)
        'dropout_rate': 0.0,             # 5. Dropout rate
        'alpha': 1.0,                    # 6. Width multiplier (1.0 para ResNet)
        'label_smoothing': 0.0,          # 7. Label smoothing
        'grad_clip': 0.0,                # 8. Gradient clipping (0 = sin clip)
        'optimizer': 'Adam'
    },
    {
        'architecture': 'ResNet18',
        'learning_rate': 0.001,
        'batch_size': 64,
        'weight_decay':  0.0001,
        'momentum': 0.0,
        'dropout_rate': 0.1,             # Añadir dropout
        'alpha':  1.0,
        'label_smoothing': 0.1,          # Label smoothing activo
        'grad_clip':  1.0,                # Gradient clipping activo
        'optimizer': 'Adam'
    },
    
    # EfficientNetB0 (2 configuraciones)
    {
        'architecture': 'EfficientNetB0',
        'learning_rate': 0.01,
        'batch_size':  64,
        'weight_decay': 0.0001,
        'momentum': 0.0,
        'dropout_rate': 0.2,             # Dropout en clasificador
        'alpha': 1.0,
        'label_smoothing': 0.0,
        'grad_clip':  0.0,
        'optimizer': 'Adam'
    },
    {
        'architecture': 'EfficientNetB0',
        'learning_rate': 0.001,
        'batch_size': 64,
        'weight_decay': 0.0001,
        'momentum': 0.0,
        'dropout_rate': 0.2,
        'alpha': 1.0,
        'label_smoothing': 0.1,
        'grad_clip': 1.0,
        'optimizer': 'Adam'
    },
    
    # MobileNetV2 (2 configuraciones)
    {
        'architecture': 'MobileNetV2',
        'learning_rate': 0.01,
        'batch_size': 64,
        'weight_decay':  0.0001,
        'momentum': 0.0,
        'dropout_rate': 0.0,
        'alpha': 1.0,                    # Width multiplier completo
        'label_smoothing': 0.0,
        'grad_clip': 0.0,
        'optimizer': 'Adam'
    },
    {
        'architecture': 'MobileNetV2',
        'learning_rate': 0.001,
        'batch_size': 64,
        'weight_decay':  0.0001,
        'momentum': 0.0,
        'dropout_rate': 0.1,
        'alpha': 1.0,
        'label_smoothing': 0.1,
        'grad_clip': 1.0,
        'optimizer': 'Adam'
    },
]

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def get_transforms(grayscale=False):
    """Genera transformaciones según tipo de imagen"""
    if grayscale:
        return transforms.Compose([
            transforms. Grayscale(3),  # Convertir a 3 canales
            transforms. Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms. Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

def get_dataset_loader(dataset_name, data_root='./data'):
    """Carga dataset de forma dinámica usando DATASET_CONFIG"""
    
    # Verificar si el dataset existe en configuración
    if dataset_name not in DATASET_CONFIG:
        available = list(DATASET_CONFIG.keys())
        raise ValueError(
            f"❌ Dataset '{dataset_name}' no está configurado.\n"
            f"📋 Disponibles ({len(available)}): {', '.join(available)}"
        )
    
    config = DATASET_CONFIG[dataset_name]
    
    # Obtener transformaciones
    transform = get_transforms(grayscale=config['grayscale'])
    
    # Construir ruta específica del dataset (para evitar descargas duplicadas)
    import os
    dataset_path = os.path.join(data_root, dataset_name)
    
    # Cargar datasets
    try:
        train_dataset = config['loader'](dataset_path)
        test_dataset = config['test_loader'](dataset_path)
    except Exception as e:
        raise RuntimeError(f"❌ Error cargando '{dataset_name}': {e}")
    
    # Aplicar transformaciones
    train_dataset. transform = transform
    test_dataset.transform = transform
    
    # Manejar datasets sin split (como EuroSAT, Caltech101, etc.)
    if config. get('needs_split', False):
        print(f"  ⚠️  '{dataset_name}' no tiene split train/test, creando split manual...")
        test_size = config.get('test_size', 0.2)
        
        # Obtener índices y labels
        indices = list(range(len(train_dataset)))
        try:
            # Intentar obtener labels para estratificación
            labels = [train_dataset[i][1] for i in indices]
            stratify = labels
        except: 
            stratify = None
            print(f"  ⚠️  No se pudo estratificar, usando split aleatorio")
        
        train_idx, test_idx = train_test_split(
            indices, 
            test_size=test_size, 
            random_state=42,
            stratify=stratify
        )
        
        # Crear subsets
        full_dataset = train_dataset
        train_dataset = Subset(full_dataset, train_idx)
        test_dataset = Subset(full_dataset, test_idx)
        
        print(f"  ✓ Split creado:  {len(train_idx)} train, {len(test_idx)} test")
    
    num_classes = config['num_classes']
    
    return train_dataset, test_dataset, num_classes

def get_model(architecture, num_classes, config):
    """Crea el modelo según la arquitectura con transfer learning"""
    if architecture == 'ResNet18': 
        model = models.resnet18(pretrained=True)
        # Congelar capas para transfer learning
        for param in model. parameters():
            param.requires_grad = False
        # Solo entrenar última capa
        model.fc = nn.Linear(model.fc. in_features, num_classes)
        
    elif architecture == 'EfficientNetB0':
        model = models.efficientnet_b0(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        
        # Aplicar dropout si está configurado
        if config['dropout_rate']: 
            model.classifier = nn. Sequential(
                nn.Dropout(p=config['dropout_rate'], inplace=True),
                nn.Linear(model.classifier[1].in_features, num_classes)
            )
        else:
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif architecture == 'MobileNetV2':
        # Alpha controla el ancho de la red
        alpha = config['alpha'] if config['alpha'] else 1.0
        model = models.mobilenet_v2(pretrained=True, width_mult=alpha)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    return model

def train_and_evaluate(model, train_loader, test_loader, config, device, dataset_name):
    """Entrena y evalúa el modelo con 5 epochs"""
    criterion = nn.CrossEntropyLoss()
    
    # Configurar optimizador según config
    if config['optimizer'] == 'Adam':
        if config['weight_decay']:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), 
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), 
                lr=config['learning_rate']
            )
    elif config['optimizer'] == 'SGD':
        optimizer = torch. optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=config['learning_rate'],
            momentum=0.9,  # SGD funciona mejor con momentum
            weight_decay=config['weight_decay'] if config['weight_decay'] else 0.0
        )
    
    model = model.to(device)
    
    # Entrenar 3 epochs
    epochs = 3
    print(f"    Entrenando {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct_train = 0
        total_train = 0
        
        # Progress bar para cada epoch
        pbar = tqdm(train_loader, desc=f"      Epoch {epoch+1}/{epochs}", 
                   leave=False, ncols=100)
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            # Actualizar progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'acc':  f'{100.*correct_train/total_train:.1f}%'
            })
        
        train_accuracy = correct_train / total_train
        avg_loss = epoch_loss / len(train_loader)
        
        print(f"      Epoch {epoch+1}:  Loss={avg_loss:.4f}, Train Acc={train_accuracy:.4f}")
    
    # Evaluar en test
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_accuracy = correct / total
    avg_test_loss = test_loss / len(test_loader)
    
    return test_accuracy, train_accuracy, avg_test_loss

# ============================================================================
# SCRIPT PRINCIPAL
# ============================================================================

if __name__ == '__main__':
    # Configuración
    DATA_ROOT = './data'
    OUTPUT_FILE = 'target_representations.csv'
    
    # ⭐ SOLO 1 DATASET PARA PRUEBA RÁPIDA
    datasets_list = [
        'MNIST',  # Solo MNIST (50 MB, muy rápido)
    ]
    
    results = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*70}")
    print(f"🚀 INICIANDO EXPERIMENTOS DE META-LEARNING")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Datasets configurados: {len(DATASET_CONFIG)}")
    print(f"Datasets a procesar: {len(datasets_list)}")
    print(f"Configuraciones por dataset: {len(configs)}")
    print(f"Total experimentos: {len(datasets_list) * len(configs)}")
    print(f"Epochs por experimento: 3")
    print(f"{'='*70}\n")
    
    start_time_total = time.time()
    
    for idx, dataset_name in enumerate(datasets_list, 1):
        print(f"\n{'='*70}")
        print(f"📦 [{idx}/{len(datasets_list)}] DATASET: {dataset_name}")
        print(f"{'='*70}")
        
        try:
            # Cargar dataset
            train_dataset, test_dataset, num_classes = get_dataset_loader(dataset_name, DATA_ROOT)
            print(f"✓ Dataset cargado:  {len(train_dataset)} train, {len(test_dataset)} test")
            print(f"✓ Número de clases:  {num_classes}")
            
            # Usar subset para velocidad (máximo 5000 train, 1000 test)
            max_train = min(5000, len(train_dataset))
            max_test = min(1000, len(test_dataset))
            
            train_indices = np.random.choice(len(train_dataset), max_train, replace=False)
            test_indices = np.random.choice(len(test_dataset), max_test, replace=False)
            
            train_subset = Subset(train_dataset, train_indices)
            test_subset = Subset(test_dataset, test_indices)
            
            print(f"✓ Usando subset: {len(train_subset)} train, {len(test_subset)} test")
            
            # Probar todas las configuraciones
            for config_idx, config in enumerate(configs, 1):
                print(f"\n  [{config_idx}/{len(configs)}] {config['architecture']} | "
                      f"LR: {config['learning_rate']} | Opt: {config['optimizer']} | "
                      f"BS: {config['batch_size']}")
                
                start_time = time.time()
                
                # Crear modelo
                model = get_model(config['architecture'], num_classes, config)
                
                # Crear DataLoaders
                train_loader = DataLoader(train_subset, batch_size=config['batch_size'], 
                                         shuffle=True, num_workers=2)
                test_loader = DataLoader(test_subset, batch_size=config['batch_size'],
                                        num_workers=2)
                
                # Entrenar y evaluar
                test_accuracy, train_accuracy, test_loss = train_and_evaluate(
                    model, train_loader, test_loader, config, device, dataset_name
                )
                
                elapsed_time = time.time() - start_time
                
                # Guardar resultado (TODOS los 8 hiperparámetros)
                result = {
                    'task_id': dataset_name,
                    'architecture': config['architecture'],
                    'learning_rate': config['learning_rate'],
                    'optimizer': config['optimizer'],
                    'batch_size': config['batch_size'],
                    'weight_decay': config['weight_decay'] if config['weight_decay'] else '',
                    'momentum': config['momentum'] if config['momentum'] else '',
                    'dropout_rate': config['dropout_rate'] if config['dropout_rate'] else '',
                    'alpha': config['alpha'] if config['alpha'] else '',
                    'label_smoothing': config['label_smoothing'] if config['label_smoothing'] else '',
                    'grad_clip': config['grad_clip'] if config['grad_clip'] else '',
                    'test_accuracy': test_accuracy,
                    'train_accuracy': train_accuracy,
                    'test_loss': test_loss,
                    'training_time_sec': elapsed_time,
                    'epochs': 3
                }
                results.append(result)
                
                print(f"    ✓ Test Accuracy: {test_accuracy:.4f} | "
                      f"Train Accuracy: {train_accuracy:.4f} | "
                      f"Time: {elapsed_time:.1f}s")
                
                # Limpiar memoria
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"❌ Error con {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Guardar CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    
    total_time = time.time() - start_time_total
    print(f"\n{'='*70}")
    print(f"✅ EXPERIMENTOS COMPLETADOS")
    print(f"{'='*70}")
    print(f"Total experimentos:  {len(results)}")
    print(f"Tiempo total: {total_time/60:.1f} minutos")
    print(f"Tiempo promedio por experimento: {total_time/len(results):.1f}s" if len(results) > 0 else "N/A")
    print(f"Archivo guardado: {OUTPUT_FILE}")
    print(f"{'='*70}\n")
    
    # Mostrar preview de resultados
    if len(results) > 0:
        print("\n📊 PREVIEW DE RESULTADOS:")
        print(df.head(10).to_string())
        print(f"\n📈 ESTADÍSTICAS:")
        print(df.groupby('architecture')['test_accuracy'].describe())
    else:
        print("⚠️ No se generaron resultados")