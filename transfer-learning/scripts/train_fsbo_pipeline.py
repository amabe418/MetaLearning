"""
Script para entrenar FSBO con datos de pipeline completo.

Entrena el modelo FSBO usando los datos preparados que incluyen
tanto configuración del pipeline como hiperparámetros del clasificador.

Uso:
    python scripts/train_fsbo_pipeline.py --algorithm adaboost
    python scripts/train_fsbo_pipeline.py --algorithm all --epochs 3000
    python scripts/train_fsbo_pipeline.py --algorithm random_forest --epochs 5000

Autor: Proyecto académico MetaLearning
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Modelos Deep Kernel GP
# =============================================================================

class DeepKernelNetwork(nn.Module):
    """
    Red neuronal que transforma configuraciones (pipeline + hiperparámetros) 
    a espacio latente.
    
    Arquitectura adaptativa según la dimensión de entrada.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, n_layers: int = 3):
        super().__init__()
        
        # Arquitectura más profunda para capturar interacciones pipeline-HP
        layers = []
        
        # Primera capa
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))
        
        # Capas ocultas
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dim
        
    def forward(self, x):
        return self.network(x)


class DeepKernelGP(ExactGP):
    """Gaussian Process con Deep Kernel para pipeline completo."""
    
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        super().__init__(train_x, train_y, likelihood)
        self.feature_extractor = feature_extractor
        self.mean_module = ConstantMean()
        latent_dim = feature_extractor.output_dim
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))
        
    def forward(self, x):
        projected_x = self.feature_extractor(x)
        mean = self.mean_module(projected_x)
        covar = self.covar_module(projected_x)
        return MultivariateNormal(mean, covar)


# =============================================================================
# Dataset
# =============================================================================

class PipelineMetaDataset:
    """
    Meta-dataset para FSBO con datos de pipeline completo.
    
    Carga datos preparados por prepare_pipeline_data.py
    """
    
    def __init__(self, min_evaluations: int = 5):
        self.min_evaluations = min_evaluations
        self.tasks = {}
        self.train_tasks = []
        self.val_tasks = []
        self.test_tasks = []
        self.y_min_global = float('inf')
        self.y_max_global = float('-inf')
        self.feature_names = []
        
    def load_from_csv(self, file_path: str):
        """Carga datos desde CSV preparado."""
        df = pd.read_csv(file_path)
        
        # Identificar columnas
        task_col = 'task_id'
        score_col = 'accuracy'
        
        # Columnas de features (todo excepto task_id y accuracy)
        feature_cols = [c for c in df.columns if c not in [task_col, score_col]]
        self.feature_names = feature_cols
        
        logger.info(f"Cargando {len(df)} muestras con {len(feature_cols)} features")
        
        # Agrupar por tarea
        for task_id, group in df.groupby(task_col):
            if len(group) < self.min_evaluations:
                continue
            
            X = group[feature_cols].values.astype(np.float32)
            y = group[score_col].values.astype(np.float32)
            
            # Manejar NaN
            X = np.nan_to_num(X, nan=0.5)
            
            self.tasks[str(task_id)] = {'X': X, 'y': y}
            self.y_min_global = min(self.y_min_global, y.min())
            self.y_max_global = max(self.y_max_global, y.max())
        
        logger.info(f"Cargadas {len(self.tasks)} tareas")
        logger.info(f"Rango de accuracy: [{self.y_min_global:.4f}, {self.y_max_global:.4f}]")
        
        return self
    
    def split_tasks(self, train_ratio=0.7, val_ratio=0.15, random_state=42):
        """Divide tareas en train/val/test."""
        task_ids = list(self.tasks.keys())
        
        self.train_tasks, temp = train_test_split(
            task_ids, test_size=(1-train_ratio), random_state=random_state
        )
        
        if len(temp) > 1:
            self.val_tasks, self.test_tasks = train_test_split(
                temp, test_size=0.5, random_state=random_state
            )
        else:
            self.val_tasks = temp
            self.test_tasks = []
        
        logger.info(f"Split: train={len(self.train_tasks)}, val={len(self.val_tasks)}, test={len(self.test_tasks)}")
        
        return self
    
    def sample_batch(self, batch_size=50):
        """Muestrea un batch de una tarea aleatoria."""
        task_id = np.random.choice(self.train_tasks)
        task = self.tasks[task_id]
        
        n = len(task['y'])
        batch_size = min(batch_size, n)
        indices = np.random.choice(n, batch_size, replace=False)
        
        return task['X'][indices], task['y'][indices], task_id
    
    def get_input_dim(self):
        """Retorna dimensión de entrada."""
        first_task = self.tasks[list(self.tasks.keys())[0]]
        return first_task['X'].shape[1]


# =============================================================================
# Task Augmentation
# =============================================================================

def task_augmentation(y_batch, y_min_global, y_max_global):
    """
    Aplica task augmentation escalando labels a rango aleatorio.
    
    Esto ayuda al modelo a generalizar a nuevas tareas con diferentes
    rangos de accuracy.
    """
    l = np.random.uniform(y_min_global, y_max_global)
    u = np.random.uniform(y_min_global, y_max_global)
    if l > u:
        l, u = u, l
    
    y_min, y_max = y_batch.min(), y_batch.max()
    if y_max - y_min < 1e-8:
        return y_batch
    
    y_scaled = l + (y_batch - y_min) / (y_max - y_min) * (u - l)
    return y_scaled.astype(np.float32)


# =============================================================================
# Entrenamiento
# =============================================================================

def train_fsbo_pipeline(
    dataset: PipelineMetaDataset,
    n_iterations: int = 3000,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    n_layers: int = 3,
    use_augmentation: bool = True,
    device: str = 'cpu'
):
    """
    Entrena modelo FSBO para pipeline completo.
    
    Args:
        dataset: PipelineMetaDataset cargado
        n_iterations: Número de iteraciones de entrenamiento
        batch_size: Tamaño del batch
        lr: Learning rate
        hidden_dim: Dimensión de capas ocultas
        n_layers: Número de capas en la red
        use_augmentation: Si usar task augmentation
        device: 'cpu' o 'cuda'
        
    Returns:
        model, likelihood, losses
    """
    input_dim = dataset.get_input_dim()
    
    logger.info(f"Input dimension: {input_dim}")
    logger.info(f"Hidden dimension: {hidden_dim}")
    logger.info(f"Number of layers: {n_layers}")
    
    # Crear modelo
    train_x = torch.zeros(1, input_dim).to(device)
    train_y = torch.zeros(1).to(device)
    
    feature_extractor = DeepKernelNetwork(input_dim, hidden_dim, n_layers).to(device)
    likelihood = GaussianLikelihood().to(device)
    model = DeepKernelGP(train_x, train_y, likelihood, feature_extractor).to(device)
    
    # Contar parámetros
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {n_params:,}")
    
    # Optimizador con weight decay
    optimizer = torch.optim.AdamW([
        {'params': model.feature_extractor.parameters(), 'lr': lr},
        {'params': model.covar_module.parameters(), 'lr': lr},
        {'params': model.mean_module.parameters(), 'lr': lr},
        {'params': likelihood.parameters(), 'lr': lr},
    ], weight_decay=1e-4)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_iterations)
    
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    # Training loop
    losses = []
    best_loss = float('inf')
    patience_counter = 0
    patience = 500  # Early stopping patience
    
    pbar = tqdm(range(n_iterations), desc="Training FSBO Pipeline")
    
    for iteration in pbar:
        model.train()
        likelihood.train()
        
        # Muestrear batch
        X_batch, y_batch, _ = dataset.sample_batch(batch_size)
        
        # Task augmentation
        if use_augmentation:
            y_batch = task_augmentation(
                y_batch, 
                dataset.y_min_global, 
                dataset.y_max_global
            )
        
        # Convertir a tensores
        X_batch = torch.tensor(X_batch, dtype=torch.float32).to(device)
        y_batch = torch.tensor(y_batch, dtype=torch.float32).to(device)
        
        # Actualizar datos del GP
        model.set_train_data(X_batch, y_batch, strict=False)
        
        # Forward y backward
        optimizer.zero_grad()
        output = model(X_batch)
        loss = -mll(output, y_batch)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        current_loss = loss.item()
        losses.append(current_loss)
        
        # Early stopping check
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Actualizar progress bar
        if (iteration + 1) % 100 == 0:
            avg_loss = np.mean(losses[-100:])
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'best': f'{best_loss:.4f}',
                'lr': f'{current_lr:.6f}'
            })
        
        # Early stopping
        if patience_counter >= patience and iteration > 1000:
            logger.info(f"Early stopping at iteration {iteration}")
            break
    
    return model, likelihood, losses


def validate_model(
    model: DeepKernelGP,
    likelihood: GaussianLikelihood,
    dataset: PipelineMetaDataset,
    device: str = 'cpu'
) -> float:
    """Valida el modelo en tareas de validación."""
    model.eval()
    likelihood.eval()
    
    if not dataset.val_tasks:
        return 0.0
    
    total_error = 0.0
    n_tasks = 0
    
    with torch.no_grad():
        for task_id in dataset.val_tasks:
            task = dataset.tasks[task_id]
            X = torch.tensor(task['X'], dtype=torch.float32).to(device)
            y = task['y']
            
            # Usar subset como "training" para el GP
            n = len(y)
            train_idx = np.random.choice(n, min(10, n), replace=False)
            test_idx = np.array([i for i in range(n) if i not in train_idx])
            
            if len(test_idx) == 0:
                continue
            
            model.set_train_data(X[train_idx], torch.tensor(y[train_idx]).to(device), strict=False)
            
            pred = likelihood(model(X[test_idx]))
            pred_mean = pred.mean.cpu().numpy()
            
            error = np.abs(pred_mean - y[test_idx]).mean()
            total_error += error
            n_tasks += 1
    
    return total_error / max(n_tasks, 1)


# =============================================================================
# Main
# =============================================================================

AVAILABLE_ALGORITHMS = [
    'adaboost', 'bernoulli_nb', 'decision_tree', 'extra_trees',
    'gaussian_nb', 'hist_gradient_boosting', 'kneighbors', 'lda',
    'linear_svc', 'mlp', 'multinomial_nb', 'passive_aggressive',
    'qda', 'random_forest', 'sgd', 'svc'
]


def main():
    parser = argparse.ArgumentParser(description='Entrenar FSBO con pipeline completo')
    parser.add_argument('--algorithm', type=str, default='adaboost',
                       choices=AVAILABLE_ALGORITHMS + ['all'])
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--no_augmentation', action='store_true')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Directorio con datos preparados')
    
    args = parser.parse_args()
    
    # Rutas
    base_dir = Path(__file__).parent.parent
    
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = base_dir / 'data' / 'pipeline_representation'
    
    checkpoint_dir = base_dir / 'experiments' / 'checkpoints_pipeline'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print("🚀 FSBO Pipeline Training")
    print("=" * 60)
    print(f"\nConfiguración:")
    print(f"  Algoritmo: {args.algorithm}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Layers: {args.n_layers}")
    print(f"  Task augmentation: {not args.no_augmentation}")
    print(f"  Device: {device}")
    print(f"  Data dir: {data_dir}")
    
    # Algoritmos a entrenar
    if args.algorithm == 'all':
        algorithms = AVAILABLE_ALGORITHMS
    else:
        algorithms = [args.algorithm]
    
    results = {}
    
    for algorithm in algorithms:
        print(f"\n{'='*60}")
        print(f"📊 Entrenando: {algorithm.upper()}")
        print(f"{'='*60}")
        
        # Verificar datos
        file_path = data_dir / f"{algorithm}_pipeline_representation.csv"
        
        if not file_path.exists():
            print(f"❌ Datos no encontrados: {file_path}")
            print(f"   Ejecuta primero: python scripts/prepare_pipeline_data.py --algorithm {algorithm}")
            continue
        
        # Cargar datos
        dataset = PipelineMetaDataset(min_evaluations=5)
        dataset.load_from_csv(str(file_path))
        dataset.split_tasks()
        
        print(f"\nDataset:")
        print(f"  Tareas totales: {len(dataset.tasks)}")
        print(f"  Train: {len(dataset.train_tasks)}")
        print(f"  Val: {len(dataset.val_tasks)}")
        print(f"  Test: {len(dataset.test_tasks)}")
        print(f"  Input dim: {dataset.get_input_dim()}")
        print(f"  y range: [{dataset.y_min_global:.4f}, {dataset.y_max_global:.4f}]")
        
        # Entrenar
        model, likelihood, losses = train_fsbo_pipeline(
            dataset=dataset,
            n_iterations=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            use_augmentation=not args.no_augmentation,
            device=device
        )
        
        # Validar
        val_error = validate_model(model, likelihood, dataset, device)
        
        # Guardar checkpoint
        checkpoint_path = checkpoint_dir / f"fsbo_pipeline_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        torch.save({
            'model_state': model.state_dict(),
            'likelihood_state': likelihood.state_dict(),
            'losses': losses,
            'config': {
                'algorithm': algorithm,
                'input_dim': dataset.get_input_dim(),
                'hidden_dim': args.hidden_dim,
                'n_layers': args.n_layers,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'feature_names': dataset.feature_names,
            },
            'dataset_info': {
                'n_tasks': len(dataset.tasks),
                'n_train_tasks': len(dataset.train_tasks),
                'y_min': dataset.y_min_global,
                'y_max': dataset.y_max_global,
            }
        }, checkpoint_path)
        
        final_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
        
        results[algorithm] = {
            'final_loss': final_loss,
            'val_error': val_error,
            'checkpoint': str(checkpoint_path)
        }
        
        print(f"\n✅ {algorithm} completado!")
        print(f"   Loss final: {final_loss:.4f}")
        print(f"   Val error: {val_error:.4f}")
        print(f"   Checkpoint: {checkpoint_path.name}")
    
    # Resumen
    print("\n" + "=" * 60)
    print("📋 RESUMEN FINAL")
    print("=" * 60)
    
    for algo, res in results.items():
        print(f"\n✅ {algo}:")
        print(f"   Loss = {res['final_loss']:.4f}")
        print(f"   Val Error = {res['val_error']:.4f}")
    
    print(f"\n📁 Checkpoints en: {checkpoint_dir}")
    print("\n¡Entrenamiento completado!")


if __name__ == "__main__":
    main()
