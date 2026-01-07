# Descargador de Datasets de Visión por Computadora

Script para descargar 30 datasets ligeros de imágenes para proyectos de Machine Learning.

## 📋 Datasets Incluidos

### 🔷 Torchvision (13 datasets)

| # | Dataset | Tamaño | Clases | Imágenes | Descripción |
|---|---------|--------|--------|----------|-------------|
| 1 | MNIST | 50 MB | 10 | 70,000 | Dígitos escritos |
| 2 | Fashion-MNIST | 60 MB | 10 | 70,000 | Ropa y accesorios |
| 3 | USPS | 20 MB | 10 | 9,298 | Dígitos escritos (postal) |
| 4 | EMNIST-Letters | 90 MB | 26 | 145,000 | Letras escritas |
| 5 | CIFAR-10 | 170 MB | 10 | 60,000 | Objetos generales |
| 6 | CIFAR-100 | 170 MB | 100 | 60,000 | Objetos (100 clases) |
| 7 | SVHN | 500 MB | 10 | 600,000 | Números de casas |
| 8 | Flowers102 | 350 MB | 102 | 8,189 | Flores |
| 9 | OxfordIIITPet | 800 MB | 37 | 7,400 | Mascotas |
| 10 | DTD | 600 MB | 47 | 5,640 | Texturas |
| 11 | GTSRB | 300 MB | 43 | 50,000 | Señales de tráfico |
| 12 | EuroSAT | 90 MB | 10 | 27,000 | Imágenes satelitales |
| 13 | Omniglot | 25 MB | 1,623 | 32,000 | Alfabetos escritos |

### 🟢 Hugging Face (2 datasets)

| # | Dataset | Tamaño | Clases | Imágenes | Descripción |
|---|---------|--------|--------|----------|-------------|
| 14 | Beans | 180 MB | 3 | 1,295 | Hojas de plantas |
| 15 | Rendered SST2 | 50 MB | 2 | ~70,000 | Sentiment analysis |

### 🟠 Kaggle (15 datasets)

| # | Dataset | Tamaño | Clases | Imágenes | Descripción |
|---|---------|--------|--------|----------|-------------|
| 16 | Intel Image | 350 MB | 6 | 25,000 | Escenas naturales |
| 17 | Malaria Cell | 350 MB | 2 | 27,558 | Células parasitadas |
| 18 | Sign Language MNIST | 50 MB | 24 | 35,000 | Lenguaje de señas |
| 19 | Blood Cell Images | 300 MB | 4 | 12,500 | Células sanguíneas |
| 20 | Cats vs Dogs | 400 MB | 2 | 25,000 | Mascotas |
| 21 | Brain Tumor MRI | 300 MB | 4 | 7,023 | Tumores cerebrales |
| 22 | Concrete Crack | 250 MB | 2 | 40,000 | Grietas en concreto |
| 23 | Weather Dataset | 300 MB | 4 | 6,862 | Condiciones climáticas |
| 24 | Tiny ImageNet | 250 MB | 200 | 100,000 | ImageNet reducido |
| 25 | Garbage Classification | 450 MB | 12 | 15,150 | Clasificación de basura |
| 26 | Colorectal Histology | 45 MB | 8 | 5,000 | Tejido colorrectal |
| 27 | Tomato Leaf Disease | 900 MB | 10 | 10,000 | Enfermedades tomate |
| 28 | Flowers Recognition | 200 MB | 5 | 4,242 | Flores |
| 29 | Animals10 | 180 MB | 10 | 28,000 | Animales |
| 30 | Rock Paper Scissors | 200 MB | 3 | 2,892 | Piedra, papel o tijera |

## 🚀 Instalación

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Configurar Kaggle API (opcional, para datasets de Kaggle)

```bash
# Instalar Kaggle CLI
pip install kaggle

# Descargar tus credenciales desde https://www.kaggle.com/account
# Ir a "Create New API Token" y descargar kaggle.json
python download_all_datasets.py
```

Este script descargará automáticamente:
- ✅ Todos los datasets de Torchvision (13 datasets)
- ✅ Datasets de Hugging Face (2 datasets)
- ✅ Datasets de Kaggle (15 datasets - requiere configuración)
```

Este script descargará automáticamente:
- ✅ Todos los datasets de Torchvision (14 datasets)
- ✅ Datasets de Hugging Face (5 datasets)
- ✅ Datasets de TensorFlow (2 datasets)
- ✅ Datasets de Kaggle (si está configurado)

### Descargar datasets específicos de Kaggle

```bash
# Intel Image Classification
kaggle datasets download -d puneet6060/intel-image-classification
unzip intel-image-classification.zip -d data/Intel_Image/

# Malaria Cell Images
kaggle datasets download -d iarunava/cell-images-for-detecting-malaria
unzip cell-images-for-detecting-malaria.zip -d data/Malaria_Cell/

# Sign Language MNIST
kaggle datasets download -d datamunge/sign-language-mnist
unzip sign-language-mnist.zip -d data/Sign_Language_MNIST/

# Blood Cell Images
kaggle datasets download -d paultimothymooney/blood-cells
unzip blood-cells.zip -d data/Blood_Cell_Images/

# Cats vs Dogs
kaggle competitions download -c dogs-vs-cats
unzip dogs-vs-cats.zip -d data/Cats_vs_Dogs/
```

## 📂 Estructura de carpetas

```
datasets/
├── download_datasets.py    # Script principal
├── requirements.txt        # Dependencias
├── README.md              # Este archivo
└── data/                  # Carpeta donde se guardan los datasets
    ├── MNIST/
    ├── Fashion-MNIST/
    ├── CIFAR-10/
    ├── Imagenette/
    └── ...
```

## 💡 Notas Importantes

1. **Espacio en disco**: Necesitarás aproximadamente 10-15 GB de espacio libre
2. **Tiempo de descarga**: Algunos datasets pueden tardar varios minutos u horas dependiendo de tu conexión
3. **Kaggle**: Requiere configuración previa de credenciales
4. **Datasets manuales**: Algunos datasets requieren descarga manual desde sus sitios web originales

## 🔗 Enlaces Útiles

- [PyTorch Datasets](https://pytorch.org/vision/stable/datasets.html)
- [Hugging Face Datasets](https://huggingface.co/datasets)
- [TensorFlow Datasets](https://www.tensorflow.org/datasets)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api)

## 📊 Uso de los Datasets

### Ejemplo: Cargar MNIST con PyTorch

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(
    root='./data/MNIST',
    train=True,
    download=False,  # Ya descargado
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

### Ejemplo: Cargar dataset de Hugging Face

```python
from datasets import load_dataset

dataset = load_dataset("beans", cache_dir="./data/Beans")
print(dataset)
```

## ⚠️ Troubleshooting

### Error de SSL/certificado
```bash
pip install --upgrade certifi
```

### Error de memoria durante la descarga
- Descarga los datasets uno por uno modificando el script
- Comenta las funciones que no necesites

### Kaggle API no funciona
- Verifica que kaggle.json esté en ~/.kaggle/
- Verifica permisos: `chmod 600 ~/.kaggle/kaggle.json`
- Verifica credenciales en https://www.kaggle.com/account

## 📝 Licencias

Cada dataset tiene su propia licencia. Verifica los términos de uso antes de utilizar los datos en proyectos comerciales.

## 🤝 Contribuciones

Si encuentras algún error o quieres agregar más datasets, ¡las contribuciones son bienvenidas!
