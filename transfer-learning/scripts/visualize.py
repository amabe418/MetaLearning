"""
Visualización de resultados experimentales.

Genera:
1. Curvas de convergencia (media ± std)
2. Tablas de comparación LaTeX
3. Box plots de métricas
4. Ranking de métodos

Uso:
    python scripts/visualize.py --results experiments/results/experiment_adaboost_*.json
    python scripts/visualize.py --results experiments/results/ --output figures/

Autor: Proyecto académico MetaLearning
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# Intentar importar matplotlib (opcional)
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Visualization disabled.")


# =============================================================================
# Colores y estilos
# =============================================================================

COLORS = {
    'FSBO': '#E63946',      # Rojo
    'RANDOM': '#457B9D',    # Azul
    'GP-LHS': '#2A9D8F',    # Verde
    'GP-RS': '#E9C46A',     # Amarillo
}

MARKERS = {
    'FSBO': 'o',
    'RANDOM': 's',
    'GP-LHS': '^',
    'GP-RS': 'D',
}


# =============================================================================
# Carga de resultados
# =============================================================================

def load_results(results_path: Path) -> List[Dict]:
    """Carga resultados de experimentos."""
    results = []
    
    if results_path.is_file():
        with open(results_path, 'r') as f:
            results.append(json.load(f))
    elif results_path.is_dir():
        for filepath in results_path.glob('experiment_*.json'):
            with open(filepath, 'r') as f:
                results.append(json.load(f))
    
    return results


# =============================================================================
# Curvas de convergencia
# =============================================================================

def plot_convergence_curves(
    results: Dict,
    algorithm: str,
    output_path: Optional[Path] = None,
    show: bool = True
):
    """
    Grafica curvas de convergencia (media ± std).
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method, data in results['results'].items():
        method_upper = method.upper()
        color = COLORS.get(method_upper, '#333333')
        
        mean = np.array(data['convergence_mean'])
        std = np.array(data['convergence_std'])
        x = np.arange(1, len(mean) + 1)
        
        # Línea principal
        ax.plot(x, mean, color=color, label=method_upper, linewidth=2)
        
        # Banda de error
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
    
    ax.set_xlabel('Number of Evaluations', fontsize=12)
    ax.set_ylabel('Best Value Found', fontsize=12)
    ax.set_title(f'Convergence Curves - {algorithm.upper()}', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_normalized_regret_curves(
    results: Dict,
    algorithm: str,
    output_path: Optional[Path] = None,
    show: bool = True
):
    """
    Grafica curvas de normalized regret.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method, data in results['results'].items():
        method_upper = method.upper()
        color = COLORS.get(method_upper, '#333333')
        
        # Calcular NR a lo largo del tiempo
        mean = np.array(data['convergence_mean'])
        std = np.array(data['convergence_std'])
        
        # Normalizar (asumiendo que el óptimo es el máximo y peor es el mínimo)
        y_max = mean.max()
        y_min = mean.min()
        
        if y_max - y_min > 1e-8:
            nr_mean = (y_max - mean) / (y_max - y_min)
            nr_std = std / (y_max - y_min)
        else:
            nr_mean = np.zeros_like(mean)
            nr_std = np.zeros_like(std)
        
        x = np.arange(1, len(mean) + 1)
        
        ax.plot(x, nr_mean, color=color, label=method_upper, linewidth=2)
        ax.fill_between(x, nr_mean - nr_std, nr_mean + nr_std, color=color, alpha=0.2)
    
    ax.set_xlabel('Number of Evaluations', fontsize=12)
    ax.set_ylabel('Normalized Regret (↓)', fontsize=12)
    ax.set_title(f'Normalized Regret - {algorithm.upper()}', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    
    plt.close()


# =============================================================================
# Box plots
# =============================================================================

def plot_boxplot_comparison(
    all_results: List[Dict],
    metric: str = 'normalized_regret',
    output_path: Optional[Path] = None,
    show: bool = True
):
    """
    Crea box plot comparando métodos.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available")
        return
    
    # Recolectar datos por método
    method_data = {}
    
    for result in all_results:
        for method, data in result['results'].items():
            method_upper = method.upper()
            if method_upper not in method_data:
                method_data[method_upper] = []
            
            if metric == 'normalized_regret':
                method_data[method_upper].append(data['normalized_regret']['mean'])
            elif metric == 'auc':
                method_data[method_upper].append(data['auc']['mean'])
    
    # Crear plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(method_data.keys())
    data = [method_data[m] for m in methods]
    colors = [COLORS.get(m, '#333333') for m in methods]
    
    bp = ax.boxplot(data, labels=methods, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    metric_label = 'Normalized Regret (↓)' if metric == 'normalized_regret' else 'AUC (↑)'
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(f'{metric_label} Comparison Across Algorithms', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    
    plt.close()


# =============================================================================
# Tablas LaTeX
# =============================================================================

def generate_latex_table(results: Dict, algorithm: str) -> str:
    """
    Genera tabla LaTeX de resultados.
    """
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        f"\\caption{{Results for {algorithm.upper()}}}",
        f"\\label{{tab:{algorithm}}}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & NR ($\downarrow$) & SR ($\downarrow$) & AUC ($\uparrow$) & Time to 95\% \\",
        r"\midrule",
    ]
    
    # Ordenar por NR
    sorted_methods = sorted(
        results['results'].items(),
        key=lambda x: x[1]['normalized_regret']['mean']
    )
    
    for method, data in sorted_methods:
        nr = data['normalized_regret']
        sr = data['simple_regret']
        auc = data['auc']
        time = data['time_to_95']
        
        nr_str = f"${nr['mean']:.4f} \\pm {nr['std']:.4f}$"
        sr_str = f"${sr['mean']:.4f} \\pm {sr['std']:.4f}$"
        auc_str = f"${auc['mean']:.4f} \\pm {auc['std']:.4f}$"
        time_str = f"${time['mean']:.1f}$" if time['mean'] else "N/A"
        
        # Marcar el mejor en negrita
        if method == sorted_methods[0][0]:
            nr_str = f"\\textbf{{{nr_str}}}"
        
        lines.append(f"{method.upper()} & {nr_str} & {sr_str} & {auc_str} & {time_str} \\\\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


def generate_comparison_latex_table(comparisons: List[Dict]) -> str:
    """
    Genera tabla LaTeX de comparaciones estadísticas.
    """
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Statistical Comparisons (Wilcoxon signed-rank test)}",
        r"\label{tab:comparisons}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Method 1 & Method 2 & NR p-value & AUC p-value & Winner \\",
        r"\midrule",
    ]
    
    for comp in comparisons:
        m1 = comp['method1'].upper()
        m2 = comp['method2'].upper()
        nr_p = comp['normalized_regret']['p_value']
        auc_p = comp['auc']['p_value']
        winner = comp['normalized_regret']['winner'].upper()
        
        # Marcar significativo
        nr_str = f"${nr_p:.4f}$"
        if comp['normalized_regret']['significant']:
            nr_str = f"\\textbf{{{nr_str}}}*"
        
        auc_str = f"${auc_p:.4f}$"
        if comp['auc']['significant']:
            auc_str = f"\\textbf{{{auc_str}}}*"
        
        lines.append(f"{m1} & {m2} & {nr_str} & {auc_str} & {winner} \\\\")
    
    lines.extend([
        r"\bottomrule",
        r"\multicolumn{5}{l}{\footnotesize * indicates $p < 0.05$} \\",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


# =============================================================================
# Ranking de métodos
# =============================================================================

def compute_ranking(all_results: List[Dict]) -> Dict:
    """
    Computa ranking promedio de métodos.
    """
    method_ranks = {}
    
    for result in all_results:
        # Ordenar por NR (menor es mejor)
        sorted_methods = sorted(
            result['results'].items(),
            key=lambda x: x[1]['normalized_regret']['mean']
        )
        
        for rank, (method, _) in enumerate(sorted_methods, 1):
            method_upper = method.upper()
            if method_upper not in method_ranks:
                method_ranks[method_upper] = []
            method_ranks[method_upper].append(rank)
    
    # Calcular ranking promedio
    avg_ranks = {
        method: np.mean(ranks)
        for method, ranks in method_ranks.items()
    }
    
    return avg_ranks


def print_ranking_table(all_results: List[Dict]):
    """
    Imprime tabla de ranking.
    """
    avg_ranks = compute_ranking(all_results)
    
    print("\n" + "=" * 50)
    print("AVERAGE RANKING (lower is better)")
    print("=" * 50)
    
    sorted_ranks = sorted(avg_ranks.items(), key=lambda x: x[1])
    
    for i, (method, rank) in enumerate(sorted_ranks, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{medal} {i}. {method:<15} {rank:.2f}")
    
    print("=" * 50)


# =============================================================================
# Generar reporte completo
# =============================================================================

def generate_report(
    all_results: List[Dict],
    output_dir: Path,
    show_plots: bool = False
):
    """
    Genera reporte completo con todas las visualizaciones.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("📊 GENERATING VISUALIZATION REPORT")
    print("=" * 70)
    
    # 1. Curvas de convergencia por algoritmo
    if HAS_MATPLOTLIB:
        print("\n📈 Generating convergence curves...")
        for result in all_results:
            algorithm = result['algorithm']
            plot_convergence_curves(
                result, 
                algorithm,
                output_dir / f"convergence_{algorithm}.png",
                show=show_plots
            )
            plot_normalized_regret_curves(
                result,
                algorithm,
                output_dir / f"regret_{algorithm}.png",
                show=show_plots
            )
    
    # 2. Box plots
    if HAS_MATPLOTLIB and len(all_results) > 1:
        print("\n📦 Generating box plots...")
        plot_boxplot_comparison(
            all_results,
            metric='normalized_regret',
            output_path=output_dir / "boxplot_nr.png",
            show=show_plots
        )
        plot_boxplot_comparison(
            all_results,
            metric='auc',
            output_path=output_dir / "boxplot_auc.png",
            show=show_plots
        )
    
    # 3. Tablas LaTeX
    print("\n📝 Generating LaTeX tables...")
    latex_content = []
    
    for result in all_results:
        latex_content.append(generate_latex_table(result, result['algorithm']))
        latex_content.append("\n")
        
        if 'comparisons' in result:
            latex_content.append(generate_comparison_latex_table(result['comparisons']))
            latex_content.append("\n")
    
    latex_path = output_dir / "tables.tex"
    with open(latex_path, 'w') as f:
        f.write("\n\n".join(latex_content))
    print(f"   Saved: {latex_path}")
    
    # 4. Ranking
    print_ranking_table(all_results)
    
    # 5. Summary markdown
    print("\n📋 Generating summary...")
    summary_path = output_dir / "summary.md"
    
    with open(summary_path, 'w') as f:
        f.write("# Experiment Results Summary\n\n")
        
        for result in all_results:
            f.write(f"## {result['algorithm'].upper()}\n\n")
            f.write("| Method | NR | AUC |\n")
            f.write("|--------|-----|-----|\n")
            
            for method, data in result['results'].items():
                nr = data['normalized_regret']
                auc = data['auc']
                f.write(f"| {method.upper()} | {nr['mean']:.4f} ± {nr['std']:.4f} | {auc['mean']:.4f} ± {auc['std']:.4f} |\n")
            
            f.write("\n")
        
        # Ranking
        f.write("## Average Ranking\n\n")
        avg_ranks = compute_ranking(all_results)
        sorted_ranks = sorted(avg_ranks.items(), key=lambda x: x[1])
        
        for i, (method, rank) in enumerate(sorted_ranks, 1):
            f.write(f"{i}. **{method}**: {rank:.2f}\n")
    
    print(f"   Saved: {summary_path}")
    
    print("\n" + "=" * 70)
    print("✅ Report generation completed!")
    print(f"   Output directory: {output_dir}")
    print("=" * 70)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Visualize experiment results')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to results file or directory')
    parser.add_argument('--output', type=str, default='figures',
                       help='Output directory for figures')
    parser.add_argument('--show', action='store_true',
                       help='Show plots interactively')
    
    args = parser.parse_args()
    
    # Cargar resultados
    results_path = Path(args.results)
    all_results = load_results(results_path)
    
    if not all_results:
        print(f"No results found in {results_path}")
        return
    
    print(f"Loaded {len(all_results)} experiment results")
    
    # Generar reporte
    output_dir = Path(args.output)
    generate_report(all_results, output_dir, show_plots=args.show)


if __name__ == "__main__":
    main()

