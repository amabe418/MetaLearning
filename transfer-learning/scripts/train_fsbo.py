"""
Script principal para entrenar FSBO (Few-Shot Bayesian Optimization).

Uso:
    python scripts/train_fsbo.py --algorithm adaboost
    python scripts/train_fsbo.py --algorithm random_forest --epochs 5000
    python scripts/train_fsbo.py --algorithm all

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
# Modelos (copiados de deep_kernel.py para evitar problemas de import)
# =============================================================================

class DeepKernelNetwork(nn.Module):
    """Red neuronal que transforma hiperparámetros a espacio latente."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, n_layers: int = 2):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dim
        
    def forward(self, x):
        return self.network(x)


class DeepKernelGP(ExactGP):
    """Gaussian Process con Deep Kernel."""
    
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

class SimpleMetaDataset:
    """Meta-dataset simplificado para FSBO."""
    
    def __init__(self, min_evaluations: int = 5):
        self.min_evaluations = min_evaluations
        self.tasks = {}
        self.train_tasks = []
        self.val_tasks = []
        self.test_tasks = []
        self.y_min_global = float('inf')
        self.y_max_global = float('-inf')
        
    def load_from_csv(self, file_path: str):
        """Carga datos desde CSV."""
        df = pd.read_csv(file_path)
        
        # Identificar columnas
        task_col = 'task_id'
        score_col = 'accuracy'
        
        # Columnas de hiperparámetros (todo excepto task_id y accuracy)
        hp_cols = [c for c in df.columns if c not in [task_col, score_col]]
        
        # Agrupar por tarea
        for task_id, group in df.groupby(task_col):
            if len(group) < self.min_evaluations:
                continue
            
            X = group[hp_cols].values.astype(np.float32)
            y = group[score_col].values.astype(np.float32)
            
            self.tasks[str(task_id)] = {'X': X, 'y': y}
            self.y_min_global = min(self.y_min_global, y.min())
            self.y_max_global = max(self.y_max_global, y.max())
        
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
    """Aplica task augmentation escalando labels a rango aleatorio."""
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

def train_fsbo(
    dataset: SimpleMetaDataset,
    n_iterations: int = 2000,
    batch_size: int = 50,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    use_augmentation: bool = True,
    device: str = 'cpu'
):
    """Entrena modelo FSBO."""
    
    input_dim = dataset.get_input_dim()
    
    # Crear modelo
    train_x = torch.zeros(1, input_dim).to(device)
    train_y = torch.zeros(1).to(device)
    
    feature_extractor = DeepKernelNetwork(input_dim, hidden_dim).to(device)
    likelihood = GaussianLikelihood().to(device)
    model = DeepKernelGP(train_x, train_y, likelihood, feature_extractor).to(device)
    
    # Optimizador
    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters()},
        {'params': model.covar_module.parameters()},
        {'params': model.mean_module.parameters()},
        {'params': likelihood.parameters()},
    ], lr=lr)
    
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    # Training loop
    losses = []
    
    pbar = tqdm(range(n_iterations), desc="Training FSBO")
    
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
        optimizer.step()
        
        losses.append(loss.item())
        
        # Actualizar progress bar
        if (iteration + 1) % 50 == 0:
            avg_loss = np.mean(losses[-50:])
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    return model, likelihood, losses


# =============================================================================
# Main
# =============================================================================

ALGORITHM_FILES = {
    'adaboost': 'adaboost_target_representation_with_scores.csv',
    'random_forest': 'random_forest_target_representation_with_scores.csv',
    'libsvm_svc': 'libsvm_svc_target_representation_with_scores.csv',
    'autosklearn': 'autosklearn_target_representation_with_scores.csv',
}


def main():
    parser = argparse.ArgumentParser(description='Entrenar modelo FSBO')
    parser.add_argument('--algorithm', type=str, default='adaboost',
                       choices=list(ALGORITHM_FILES.keys()) + ['all'])
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--no_augmentation', action='store_true')
    
    args = parser.parse_args()
    
    # Rutas
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / 'representation_with_scores'
    checkpoint_dir = base_dir / 'experiments' / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print("🚀 FSBO Training")
    print("=" * 60)
    print(f"\nConfiguración:")
    print(f"  Algoritmo: {args.algorithm}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Task augmentation: {not args.no_augmentation}")
    print(f"  Device: {device}")
    
    # Algoritmos a entrenar
    algorithms = list(ALGORITHM_FILES.keys()) if args.algorithm == 'all' else [args.algorithm]
    
    results = {}
    
    for algorithm in algorithms:
        print(f"\n{'='*60}")
        print(f"📊 Entrenando: {algorithm.upper()}")
        print(f"{'='*60}")
        
        # Cargar datos
        file_path = data_dir / ALGORITHM_FILES[algorithm]
        
        if not file_path.exists():
            print(f"❌ Archivo no encontrado: {file_path}")
            continue
        
        dataset = SimpleMetaDataset(min_evaluations=5)
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
        model, likelihood, losses = train_fsbo(
            dataset=dataset,
            n_iterations=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            use_augmentation=not args.no_augmentation,
            device=device
        )
        
        # Guardar checkpoint
        checkpoint_path = checkpoint_dir / f"fsbo_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        torch.save({
            'model_state': model.state_dict(),
            'likelihood_state': likelihood.state_dict(),
            'losses': losses,
            'config': vars(args)
        }, checkpoint_path)
        
        final_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
        
        results[algorithm] = {
            'final_loss': final_loss,
            'checkpoint': str(checkpoint_path)
        }
        
        print(f"\n✅ {algorithm} completado!")
        print(f"   Loss final: {final_loss:.4f}")
        print(f"   Checkpoint: {checkpoint_path.name}")
    
    # Resumen
    print("\n" + "=" * 60)
    print("📋 RESUMEN FINAL")
    print("=" * 60)
    
    for algo, res in results.items():
        print(f"\n✅ {algo}: Loss = {res['final_loss']:.4f}")
    
    print(f"\n📁 Checkpoints en: {checkpoint_dir}")
    print("\n¡Entrenamiento completado!")


if __name__ == "__main__":
    main()
