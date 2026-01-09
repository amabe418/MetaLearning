"""
Demo: Cómo funciona el sistema sin Knowledge Base.

Muestra las estrategias de fallback cuando no hay datos históricos.
"""

import numpy as np
from sklearn.datasets import make_classification
import sys
sys.path.insert(0, '.')

from pipeline import KnowledgeBase, DatasetMetaFeatures
from meta_learner import TransferLearningMetaLearner


def demo_sin_kb():
    """Demo con knowledge base vacía."""
    print("\n" + "=" * 70)
    print("🧪 DEMO: Sistema SIN Knowledge Base (Primera Ejecución)")
    print("=" * 70)
    
    # 1. KB vacía
    kb = KnowledgeBase()  # No cargar archivo
    print(f"\n📦 Knowledge Base: VACÍA ({len(kb.entries)} entradas)")
    
    # 2. Meta-learner con KB vacía
    meta_learner = TransferLearningMetaLearner(kb)
    
    # 3. Diferentes tipos de datasets
    datasets = [
        ("Pequeño (100 samples, 5 features)", 
         make_classification(n_samples=100, n_features=5, n_informative=3, n_classes=2, random_state=1)),
        
        ("Grande (10000 samples, 10 features)",
         make_classification(n_samples=10000, n_features=10, n_informative=8, n_classes=2, random_state=2)),
        
        ("Alta dimensionalidad (500 samples, 100 features)",
         make_classification(n_samples=500, n_features=100, n_informative=20, n_classes=2, random_state=3)),
    ]
    
    for name, (X, y) in datasets:
        print(f"\n{'─' * 70}")
        print(f"📊 Dataset: {name}")
        print(f"   Shape: {X.shape}")
        
        # Extraer meta-features
        meta_features = DatasetMetaFeatures.from_data(X, y, dataset_id=name)
        
        # Pedir sugerencias (sin KB)
        print(f"\n🤖 Meta-Learner (sin datos históricos):")
        suggestions = meta_learner.suggest_algorithms(meta_features, top_k=3)
        
        print(f"\n✨ Sugerencias AJUSTADAS por características del dataset:")
        for i, s in enumerate(suggestions, 1):
            print(f"   {i}. {s.name:<20} confianza={s.confidence:.3f}")
        
        # Explicar por qué
        log_n_samples = meta_features.meta_vector[0]
        log_n_features = meta_features.meta_vector[1]
        n_samples = int(np.exp(log_n_samples))
        n_features = int(np.exp(log_n_features))
        
        print(f"\n   💡 Heurística:")
        if n_samples < 500:
            print(f"      • Dataset pequeño ({n_samples}) → Random Forest + AdaBoost")
        elif n_samples > 10000:
            print(f"      • Dataset grande ({n_samples}) → Random Forest (escala bien)")
        
        if n_features > 50:
            print(f"      • Alta dimensionalidad ({n_features}) → Gradient Boosting")
        elif n_features < 10:
            print(f"      • Baja dimensionalidad ({n_features}) → Cualquier algoritmo")


def demo_estrategia_completa():
    """Muestra la estrategia completa de fallback."""
    print("\n" + "=" * 70)
    print("📚 ESTRATEGIA COMPLETA DE FALLBACK")
    print("=" * 70)
    
    print("""
🔄 NIVELES DE FALLBACK:

1️⃣ IDEAL: Knowledge Base con tareas similares
   ├─ Buscar en KB
   ├─ Filtrar por similitud > threshold (0.5)
   ├─ Rankear por performance en tareas similares
   ├─ Warm start con configs reales
   └─ ✅ Sugerencias basadas en DATOS REALES

2️⃣ FALLBACK 1: KB existe pero no hay suficientemente similares
   ├─ Bajar threshold de similitud (0.5 → 0.3)
   ├─ Si aún no hay suficientes → siguiente nivel
   └─ Usar lo mejor disponible + heurísticas

3️⃣ FALLBACK 2: KB vacía o sin tareas similares
   ├─ Sugerencias por defecto (RF, GB, AdaBoost)
   ├─ AJUSTAR por meta-features del dataset:
   │  ├─ Dataset pequeño → RF + AdaBoost ⬆️
   │  ├─ Dataset grande → RF ⬆️, GB ⬇️
   │  ├─ Alta dimensionalidad → GB ⬆️
   │  └─ Baja dimensionalidad → todos igual
   └─ ✅ Sugerencias HEURÍSTICAS pero INTELIGENTES

4️⃣ FALLBACK 3: Warm start sin KB
   ├─ No hay configs de tareas similares
   ├─ Usar modelo FSBO pre-entrenado
   │  (entrenado en datos de OpenML)
   └─ ✅ Mejor que RANDOM PURO

5️⃣ ÚLTIMO RECURSO: Sin modelo FSBO
   ├─ Random search puro
   └─ ⚠️ Menos eficiente pero funciona

═══════════════════════════════════════════════════════════════════════

💡 LO IMPORTANTE:
   • El sistema SIEMPRE puede sugerir algo
   • Mejora automáticamente con cada uso
   • Primera vez: heurísticas inteligentes
   • Segunda vez en adelante: transfer learning real

🔄 AUTO-MEJORA:
   1. Primera tarea → Usa defaults + heurísticas
   2. Guarda resultado en KB
   3. Segunda tarea → Ya tiene 1 entrada en KB
   4. Tercera tarea → Empieza a hacer transfer real
   5. N-ésima tarea → KB rica, transfer muy efectivo ✨
    """)


def demo_mejora_progresiva():
    """Demo de cómo el sistema mejora con el tiempo."""
    print("\n" + "=" * 70)
    print("📈 DEMO: MEJORA PROGRESIVA")
    print("=" * 70)
    
    kb = KnowledgeBase()
    meta_learner = TransferLearningMetaLearner(kb, similarity_threshold=0.3)
    
    # Simular 5 tareas sucesivas
    print("\n🔄 Simulando 5 tareas consecutivas:\n")
    
    for i in range(1, 6):
        X, y = make_classification(
            n_samples=500 + i*100,
            n_features=20,
            n_informative=15,
            n_classes=2,
            random_state=i
        )
        
        meta_features = DatasetMetaFeatures.from_data(
            X, y, 
            dataset_id=f"task_{i}"
        )
        
        print(f"{'━' * 70}")
        print(f"📊 Tarea {i}: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"   KB actual: {len(kb.entries)} entradas")
        
        # Sugerir
        suggestions = meta_learner.suggest_algorithms(meta_features, top_k=2)
        
        print(f"   Sugerencias:")
        for s in suggestions:
            source = "defaults" if s.similar_tasks == 0 else f"{s.similar_tasks} similares"
            print(f"   • {s.name}: confianza={s.confidence:.3f} (de {source})")
        
        # Simular optimización y guardar resultado
        # (en realidad optimizarías con FSBO aquí)
        best_algo = suggestions[0].name
        simulated_score = 0.80 + np.random.uniform(-0.05, 0.15)
        simulated_config = {'dummy': 'config'}
        
        kb.add_entry(meta_features, best_algo, simulated_config, simulated_score)
        
        print(f"   ✓ Guardado: {best_algo} con score={simulated_score:.3f}")
        
        if i == 1:
            print(f"   💭 Primera tarea: usando defaults + heurísticas")
        elif i == 2:
            print(f"   💭 Segunda tarea: ya hay 1 entrada en KB")
        elif i >= 3:
            print(f"   💭 Transfer learning activo! {len(kb.entries)} tareas anteriores")
    
    print(f"\n{'━' * 70}")
    print(f"✨ RESULTADO FINAL:")
    print(f"   • KB: {len(kb.entries)} entradas")
    print(f"   • El sistema ahora puede hacer transfer learning efectivo")
    print(f"   • Cada nueva tarea mejora el sistema 🚀")


if __name__ == "__main__":
    demo_sin_kb()
    demo_estrategia_completa()
    demo_mejora_progresiva()
    
    print("\n" + "=" * 70)
    print("✅ CONCLUSIÓN:")
    print("   • Sistema funciona SIEMPRE (con o sin KB)")
    print("   • Sin KB: usa heurísticas inteligentes")
    print("   • Con KB: usa transfer learning real")
    print("   • Auto-mejora: cada tarea hace el sistema más inteligente")
    print("=" * 70 + "\n")

