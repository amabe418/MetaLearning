# Descargador de Datasets de Visión por Computadora

Script para descargar 30 datasets de imágenes de Torchvision para proyectos de Meta-Learning.

**Total: 30 datasets (14 escala de grises + 16 RGB)**

## 📋 Datasets Incluidos

### 🔷 Torchvision - Escala de Grises (14 datasets)

| # | Dataset | Tamaño | Clases | Imágenes | Descripción |
|---|---------|--------|--------|----------|-------------|
| 1 | MNIST | 50 MB | 10 | 70,000 | Dígitos escritos |
| 2 | FashionMNIST | 60 MB | 10 | 70,000 | Ropa y accesorios |
| 3 | KMNIST | 50 MB | 10 | 70,000 | Caracteres japoneses |
| 4 | EMNIST | 90 MB | 26 | 145,000 | Letras escritas |
| 5 | EMNIST_Balanced | 90 MB | 47 | 131,600 | EMNIST balanceado |
| 6 | EMNIST_Digits | 90 MB | 10 | 280,000 | Solo dígitos |
| 7 | EMNIST_MNIST | 90 MB | 10 | 70,000 | Compatible con MNIST |
| 8 | EMNIST_ByClass | 90 MB | 62 | 814,255 | Por clase |
| 9 | EMNIST_ByMerge | 90 MB | 47 | 814,255 | Clases fusionadas |
| 10 | USPS | 20 MB | 10 | 9,298 | Dígitos escritos (postal) |
| 11 | QMNIST | 60 MB | 10 | 120,000 | MNIST extendido |
| 12 | Omniglot | 25 MB | 1,623 | 32,000 | Alfabetos escritos |
| 13 | RenderedSST2 | 50 MB | 2 | ~70,000 | Sentiment analysis |
| 14 | FER2013 | 350 MB | 7 | 35,887 | Reconocimiento emociones |

### 🔶 Torchvision - RGB (16 datasets)

| # | Dataset | Tamaño | Clases | Imágenes | Descripción |
|---|---------|--------|--------|----------|-------------|
| 15 | CIFAR10 | 170 MB | 10 | 60,000 | Objetos generales |
| 16 | CIFAR100 | 170 MB | 100 | 60,000 | Objetos (100 clases) |
| 17 | SVHN | 500 MB | 10 | 600,000 | Números de casas |
| 18 | STL10 | 2.6 GB | 10 | 13,000 | Objetos alta resolución |
| 19 | STL10_Unlabeled | 2.6 GB | - | 100,000 | STL10 sin etiquetas |
| 20 | Flowers102 | 350 MB | 102 | 8,189 | Flores |
| 21 | OxfordIIITPet | 800 MB | 37 | 7,400 | Mascotas |
| 22 | DTD | 600 MB | 47 | 5,640 | Texturas |
| 23 | GTSRB | 300 MB | 43 | 50,000 | Señales de tráfico |
| 24 | EuroSAT | 90 MB | 10 | 27,000 | Imágenes satelitales |
| 25 | PCAM | 7.5 GB | 2 | 327,680 | Detección cáncer |
| 26 | StanfordCars | 1.9 GB | 196 | 16,185 | Modelos de autos |
| 27 | FGVCAircraft | 2.6 GB | 100 | 10,000 | Modelos de aviones |
| 28 | Country211 | 5.5 GB | 211 | 211,000 | Países |
| 29 | Caltech101 | 130 MB | 101 | 9,146 | Objetos variados |
| 30 | LFWPeople | 200 MB | 5,749 | 13,233 | Rostros famosos |

## 🚀 Instalación

### 1. Instalar dependencias

```bash
pip install torch torchvision
```

### 2. Ejecutar descarga automática

```bash
python download_all_datasets.py
```

Este script descargará automáticamente:
- ✅ 30 datasets de Torchvision (14 grises + 16 RGB)
- ✅ Sin necesidad de Kaggle API
- ✅ Sin configuraciones adicionales

## 📂 Estructura de carpetas

```
images/
├── download_all_datasets.py  # Script de descarga
├── download.md               # Este archivo
└── data/                     # Carpeta con los datasets
    ├── MNIST/
    ├── FashionMNIST/
    ├── KMNIST/
    ├── CIFAR10/
    ├── CIFAR100/
    └── ... (30 datasets totales)
```

## 💡 Notas Importantes

1. **Espacio en disco**: Necesitarás aproximadamente 15-25 GB de espacio libre
2. **Tiempo de descarga**: La descarga completa puede tomar 1-3 horas dependiendo de tu conexión
3. **Todos automáticos**: Los 30 datasets se descargan automáticamente desde Torchvision
4. **Sin configuración**: No se requiere API de Kaggle ni otras configuraciones

## 🔗 Enlaces Útiles

- [PyTorch Datasets](https://pytorch.org/vision/stable/datasets.html)
- [Hugging Face Datasets](https://huggingface.co/datasets)
- [TensorFlow Datasets](https://www.tensorflow.org/datasets)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api)

## 📊 Uso de los Datasets

### Ejemplo: Cargar datasets con PyTorch

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Dataset en escala de grises
train_dataset = datasets.MNIST(
    root='./data/MNIST',
    train=True,
    download=False,  # Ya descargado
    transform=transform
)

# Dataset RGB
cifar_dataset = datasets.CIFAR10(
    root='./data/CIFAR10',
    train=True,
    download=False,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

## ⚠️ Troubleshooting

### Error de SSL/certificado
```bash
pip install --upgrade certifi
```

### Error de memoria durante la descarga
- Descarga los datasets uno por uno modificando el script
- Comenta datasets grandes como PCAM, Country211 o STL10

### Dataset ya existe
- El script detecta automáticamente datasets ya descargados
- Solo descarga los que faltan

## 📝 Licencias

Cada dataset tiene su propia licencia. Verifica los términos de uso antes de utilizar los datos en proyectos comerciales.

## 🤝 Contribuciones

Si encuentras algún error o quieres agregar más datasets, ¡las contribuciones son bienvenidas!
