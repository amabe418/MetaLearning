#!/usr/bin/env python3
"""
Script para descargar TODOS los 30 datasets de visión por computadora
Requisitos: Ejecutar setup_environment.sh primero
Uso: python download_all_datasets.py
"""

import os
import sys
from pathlib import Path

# Directorio base para guardar todos los datasets
BASE_DIR = Path(__file__).parent / "data"
BASE_DIR.mkdir(exist_ok=True)

print("\n" + "🎯" * 35)
print("  DESCARGADOR AUTOMÁTICO DE 30 DATASETS (TORCHVISION)")
print("🎯" * 35)
print(f"\n📁 Los datasets se guardarán en: {BASE_DIR.absolute()}\n")


def print_section(title):
    """Imprime una sección destacada"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def download_torchvision_datasets():
    """Descarga todos los datasets de Torchvision"""
    print_section("DESCARGANDO DATASETS DE TORCHVISION (30 datasets)")
    
    try:
        from torchvision import datasets, transforms
        
        transform = transforms.ToTensor()
        
        # Lista completa de datasets (14 grises + 16 RGB = 30 total)
        datasets_list = [
            # DATASETS EN ESCALA DE GRISES (14)
            ("MNIST", datasets.MNIST, {}, True),
            ("FashionMNIST", datasets.FashionMNIST, {}, True),
            ("KMNIST", datasets.KMNIST, {}, True),
            ("EMNIST", datasets.EMNIST, {"split": "letters"}, False),
            ("EMNIST_Balanced", datasets.EMNIST, {"split": "balanced"}, False),
            ("EMNIST_Digits", datasets.EMNIST, {"split": "digits"}, False),
            ("EMNIST_MNIST", datasets.EMNIST, {"split": "mnist"}, False),
            ("EMNIST_ByClass", datasets.EMNIST, {"split": "byclass"}, False),
            ("EMNIST_ByMerge", datasets.EMNIST, {"split": "bymerge"}, False),
            ("USPS", datasets.USPS, {}, True),
            ("QMNIST", datasets.QMNIST, {}, True),
            ("Omniglot", datasets.Omniglot, {"background": True}, False),
            ("RenderedSST2", datasets.RenderedSST2, {"split": "train"}, False),
            ("FER2013", datasets.FER2013, {"split": "train"}, False),
            
            # DATASETS RGB (16)
            ("CIFAR10", datasets.CIFAR10, {}, True),
            ("CIFAR100", datasets.CIFAR100, {}, True),
            ("SVHN", datasets.SVHN, {"split": "train"}, False),
            ("STL10", datasets.STL10, {"split": "train"}, False),
            ("STL10_Unlabeled", datasets.STL10, {"split": "unlabeled"}, False),
            ("Flowers102", datasets.Flowers102, {"split": "train"}, False),
            ("OxfordIIITPet", datasets.OxfordIIITPet, {"split": "trainval"}, False),
            ("DTD", datasets.DTD, {"split": "train"}, False),
            ("GTSRB", datasets.GTSRB, {"split": "train"}, False),
            ("EuroSAT", datasets.EuroSAT, {}, False),
            ("PCAM", datasets.PCAM, {"split": "train"}, False),
            ("StanfordCars", datasets.StanfordCars, {"split": "train"}, False),
            ("FGVCAircraft", datasets.FGVCAircraft, {"split": "train"}, False),
            ("Country211", datasets.Country211, {"split": "train"}, False),
            ("Caltech101", datasets.Caltech101, {}, False),
            ("LFWPeople", datasets.LFWPeople, {"split": "train"}, False),
        ]
        
        total = len(datasets_list)
        success_count = 0
        
        for idx, (name, dataset_class, kwargs, has_train_test) in enumerate(datasets_list, 1):
            try:
                print(f"\n[{idx}/{total}] 📦 Descargando {name}...")
                dataset_dir = BASE_DIR / name
                dataset_dir.mkdir(exist_ok=True)
                
                if has_train_test:
                    # Datasets con train/test
                    train_dataset = dataset_class(
                        root=str(dataset_dir),
                        train=True,
                        download=True,
                        transform=transform
                    )
                    test_dataset = dataset_class(
                        root=str(dataset_dir),
                        train=False,
                        download=True,
                        transform=transform
                    )
                    print(f"✅ {name}: {len(train_dataset)} train + {len(test_dataset)} test")
                else:
                    # Datasets con splits específicos
                    dataset = dataset_class(
                        root=str(dataset_dir),
                        download=True,
                        transform=transform,
                        **kwargs
                    )
                    print(f"✅ {name}: {len(dataset)} imágenes")
                
                success_count += 1
                
            except Exception as e:
                print(f"❌ Error con {name}: {str(e)[:150]}")
        
        print(f"\n✅ Torchvision: {success_count}/{total} datasets descargados")
        return success_count
        
    except ImportError as e:
        print(f"❌ Error: torchvision no está instalado")
        print(f"   Ejecuta: pip install torch torchvision")
        return 0


def download_huggingface_datasets():
    """Omitir Hugging Face - solo usar Torchvision"""
    print_section("DATASETS DE HUGGING FACE")
    print("⏭️  Omitiendo Hugging Face - usando solo Torchvision")
    return 0


def download_huggingface_datasets_OLD():
    """Descarga datasets de Hugging Face"""
    print_section("DESCARGANDO DATASETS DE HUGGING FACE (2 datasets)")
    
    try:
        from datasets import load_dataset
        
        hf_datasets = [
            ("Beans", "beans", None),
            ("Rendered_SST2", "sst2", None),
        ]
        
        total = len(hf_datasets)
        success_count = 0
        
        for idx, (name, dataset_id, config) in enumerate(hf_datasets, 1):
            try:
                print(f"\n[{idx}/{total}] 📦 Descargando {name} desde Hugging Face...")
                dataset_dir = BASE_DIR / name
                dataset_dir.mkdir(exist_ok=True)
                
                if config:
                    dataset = load_dataset(dataset_id, config, cache_dir=str(dataset_dir))
                else:
                    dataset = load_dataset(dataset_id, cache_dir=str(dataset_dir))
                
                total_samples = sum(len(dataset[split]) for split in dataset.keys())
                print(f"✅ {name}: {total_samples} imágenes")
                success_count += 1
                
            except Exception as e:
                print(f"❌ Error con {name}: {str(e)[:150]}")
        
        print(f"\n✅ Hugging Face: {success_count}/{total} datasets descargados")
        return success_count
        
    except ImportError:
        print(f"❌ Error: datasets (Hugging Face) no está instalado")
        print(f"   Ejecuta: pip install datasets")
        return 0


def download_tensorflow_datasets():
    """Descarga datasets de TensorFlow Datasets"""
    print_section("DESCARGANDO DATASETS DE TENSORFLOW")
    
    print("⏭️  Datasets de TensorFlow movidos a Kaggle")
    print("   Ejecutando descarga desde Kaggle en su lugar...")
    return 0


def download_kaggle_datasets():
    """Omitir Kaggle - solo usar Torchvision"""
    print_section("DATASETS DE KAGGLE")
    print("⏭️  Omitiendo Kaggle - usando solo Torchvision")
    return 0


def download_kaggle_datasets_OLD():
    """Descarga datasets de Kaggle"""
    print_section("DESCARGANDO DATASETS DE KAGGLE (15 datasets)")
    # Verificar configuración de Kaggle
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("❌ Kaggle NO está configurado")
        print("\n📋 Para configurar Kaggle:")
        print("   1. Ve a: https://www.kaggle.com/account")
        print("   2. Click en 'Create New Token' → descarga kaggle.json")
        print("   3. Ejecuta:")
        print("      mkdir -p ~/.kaggle")
        print("      mv ~/Downloads/kaggle.json ~/.kaggle/")
        print("      chmod 600 ~/.kaggle/kaggle.json")
        print("\n⏭️  Omitiendo datasets de Kaggle por ahora")
        return 0
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        # Autenticar
        api = KaggleApi()
        api.authenticate()
        print("✅ Autenticación exitosa con Kaggle\n")
        
        # Lista de datasets de Kaggle
        kaggle_datasets = [
            ("Intel_Image", "puneet6060/intel-image-classification"),
            ("Malaria_Cell", "iarunava/cell-images-for-detecting-malaria"),
            ("Sign_Language_MNIST", "datamunge/sign-language-mnist"),
            ("Blood_Cell_Images", "paultimothymooney/blood-cells"),
            ("Cats_vs_Dogs", "tongpython/cat-and-dog"),
            ("Brain_Tumor_MRI", "masoudnickparvar/brain-tumor-mri-dataset"),
            ("Concrete_Crack", "arunrk7/surface-crack-detection"),
            ("Weather_Dataset", "jehanbhathena/weather-dataset"),
            ("Tiny_ImageNet", "akash2sharma/tiny-imagenet"),
            ("Garbage_Classification", "mostafaabla/garbage-classification"),
            ("Colorectal_Histology", "kmader/colorectal-histology-mnist"),
            ("Tomato_Leaf_Disease", "kaustubhb999/tomatoleaf"),
            ("Flowers_Recognition", "alxmamaev/flowers-recognition"),
            ("Animals10", "alessiocorrado99/animals10"),
            ("Rock_Paper_Scissors", "drgfreeman/rockpaperscissors"),
        ]
        
        total = len(kaggle_datasets)
        success_count = 0
        
        for idx, (name, dataset_id) in enumerate(kaggle_datasets, 1):
            try:
                dataset_dir = BASE_DIR / name
                dataset_dir.mkdir(exist_ok=True)
                
                # Verificar si ya existe contenido descomprimido
                existing_files = list(dataset_dir.glob("*"))
                # Filtrar archivos .zip para evitar contar descargas incompletas
                non_zip_files = [f for f in existing_files if not f.name.endswith('.zip')]
                
                if non_zip_files:
                    print(f"[{idx}/{total}] ⏭️  {name} ya descargado, omitiendo...\n")
                    success_count += 1
                    continue
                
                print(f"[{idx}/{total}] 📦 Descargando {name} desde Kaggle...")
                api.dataset_download_files(
                    dataset_id,
                    path=str(dataset_dir),
                    unzip=True,
                    quiet=False
                )
                print(f"✅ {name} descargado\n")
                success_count += 1
                
            except Exception as e:
                print(f"❌ Error con {name}: {str(e)[:150]}\n")
        
        print(f"✅ Kaggle: {success_count}/{total} datasets descargados")
        return success_count
        
    except ImportError:
        print("❌ Error: kaggle no está instalado")
        print("   Ejecuta: pip install kaggle")
        return 0
    except Exception as e:
        print(f"❌ Error de autenticación: {str(e)[:150]}")
        return 0


def download_manual_datasets_info():
    """Información sobre datasets que requieren descarga manual"""
    print_section("DATASETS QUE REQUIEREN DESCARGA MANUAL")
    
    manual_datasets = [
        {
            "name": "COIL-100",
            "url": "http://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php",
            "description": "Objetos 3D rotados - Descarga manual desde Columbia University"
        },
        {
            "name": "notMNIST",
            "url": "http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html",
            "description": "Letras con diferentes fuentes - Descarga desde Google Drive"
        },
        {
            "name": "UC Merced Land Use",
            "url": "http://weegee.vision.ucmerced.edu/datasets/landuse.html",
            "description": "Imágenes aéreas - Descarga manual desde UC Merced"
        },
        {
            "name": "CINIC-10",
            "url": "https://datashare.ed.ac.uk/handle/10283/3192",
            "description": "CIFAR + ImageNet - Descarga desde Edinburgh DataShare"
        },
    ]
    
    print("\n📋 Estos datasets requieren descarga manual:\n")
    
    for idx, dataset in enumerate(manual_datasets, 1):
        print(f"{idx}. {dataset['name']}")
        print(f"   URL: {dataset['url']}")
        print(f"   📝 {dataset['description']}\n")
    
    print("💡 Descarga estos archivos manualmente y colócalos en data/[nombre_dataset]/")


def show_summary(stats):
    """Muestra resumen final de descargas"""
    print_section("✨ DESCARGA COMPLETADA ✨")
    
    total_downloaded = sum(stats.values())
    
    print("\n📊 Resumen de descargas:")
    print(f"   ✅ Torchvision:     {stats.get('torchvision', 0)}/30 datasets")
    print(f"\n   📦 TOTAL:           {total_downloaded}/30 datasets")
    
    # Calcular espacio usado
    try:
        total_size = sum(f.stat().st_size for f in BASE_DIR.rglob('*') if f.is_file())
        size_gb = total_size / (1024**3)
        size_mb = total_size / (1024**2)
        
        if size_gb > 1:
            print(f"   💾 Espacio usado:   {size_gb:.2f} GB")
        else:
            print(f"   💾 Espacio usado:   {size_mb:.2f} MB")
    except:
        pass
    
    print(f"\n📂 Ubicación: {BASE_DIR.absolute()}")
    
    print("\n💡 Próximos pasos:")
    print("   1. Revisa los datasets descargados en data/")
    print("   2. Usa los datasets en tus proyectos de ML")
    print("   3. Para datasets manuales, descárgalos de las URLs mostradas")
    
    print("\n" + "=" * 70)


def main():
    """Función principal"""
    
    # Verificar que estamos en un entorno virtual
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  ADVERTENCIA: No parece que estés en un entorno virtual")
        print("   Se recomienda ejecutar primero: source venv_datasets/bin/activate")
        response = input("\n¿Continuar de todos modos? (s/N): ")
        if response.lower() != 's':
            print("❌ Cancelado. Activa el entorno virtual primero.")
            return
    
    print("\nEste script descargará automáticamente 30 datasets de Torchvision")
    print("Esto puede tomar entre 1 y 3 horas dependiendo de tu conexión")
    print("Se necesitan aproximadamente 15-20 GB de espacio en disco\n")
    
    response = input("¿Deseas continuar? (S/n): ")
    if response.lower() == 'n':
        print("❌ Cancelado por el usuario")
        return
    
    # Estadísticas de descarga
    stats = {}
    
    try:
        # Descargar desde diferentes fuentes
        stats['torchvision'] = download_torchvision_datasets()
        stats['huggingface'] = download_huggingface_datasets()
        stats['tensorflow'] = download_tensorflow_datasets()
        stats['kaggle'] = download_kaggle_datasets()
        
        # Información sobre datasets manuales
        download_manual_datasets_info()
        
        # Mostrar resumen
        show_summary(stats)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Descarga interrumpida por el usuario")
        print("   Puedes ejecutar el script de nuevo para continuar")
        print("   Los datasets ya descargados no se volverán a descargar")
    except Exception as e:
        print(f"\n❌ Error inesperado: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
