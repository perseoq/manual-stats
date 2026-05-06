# Capítulo 29: Clustering

[← Anterior](28-analisis-multivariante.md) | [Índice](index.md) | [Siguiente →](30-arboles-decision.md)

---

## 29.1 Introducción

El **clustering** (o segmentación) es una técnica de aprendizaje no supervisado que agrupa observaciones similares sin etiquetas previas. Es fundamental para segmentación de clientes, productos y proveedores.

### Tipos de Clustering

| Algoritmo | Tipo | Características |
|-----------|------|-----------------|
| **K-Means** | Particional | Rápido, requiere k |
| **Jerárquico** | Jerárquico | Dendrograma, no requiere k |
| **DBSCAN** | Basado en densidad | Detecta outliers, formas arbitrarias |
| **GMM** | Probabilístico | Asigna probabilidades de pertenencia |

---

## 29.2 K-Means

### Algoritmo
1. Elegir k centroides iniciales
2. Asignar cada punto al centroide más cercano
3. Recalcular centroides como media del cluster
4. Repetir 2-3 hasta convergencia

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns

np.random.seed(42)

# Datos de clientes
n = 500
clientes = pd.DataFrame({
    'gasto_anual': np.concatenate([
        np.random.normal(3000, 500, 200),   # Segmento 1: bajo gasto
        np.random.normal(8000, 1000, 150),  # Segmento 2: medio
        np.random.normal(15000, 2000, 150)  # Segmento 3: alto
    ]),
    'frecuencia': np.concatenate([
        np.random.normal(5, 2, 200),
        np.random.normal(15, 4, 150),
        np.random.normal(30, 8, 150)
    ]),
    'antiguedad': np.random.randint(1, 10, n)
})

# Estandarizar
scaler = StandardScaler()
X = scaler.fit_transform(clientes)

# K-Means con k=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clientes['cluster_kmeans'] = kmeans.fit_predict(X)

print("=== K-MEANS CLUSTERING ===")
print(f"Número de clusters: 3")
print(f"Inercia: {kmeans.inertia_:.0f}")
print(f"Silhouette Score: {silhouette_score(X, clientes['cluster_kmeans']):.4f}")
```

### Determinación del Número de Clusters (k)

```python
def evaluar_k(X, max_k=10):
    """Evalúa el número óptimo de clusters"""
    inertias = []
    silhouettes = []
    
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Método del codo
    axes[0].plot(range(2, max_k + 1), inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Número de clusters (k)', fontweight='bold')
    axes[0].set_ylabel('Inercia', fontweight='bold')
    axes[0].set_title('Método del Codo', fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # Silhouette
    axes[1].plot(range(2, max_k + 1), silhouettes, 'ro-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Número de clusters (k)', fontweight='bold')
    axes[1].set_ylabel('Silhouette Score', fontweight='bold')
    axes[1].set_title('Silhouette Score', fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('evaluacion_k.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    best_k = np.argmax(silhouettes) + 2
    print(f"Mejor k (Silhouette): {best_k} (score = {silhouettes[best_k-2]:.4f})")
    
    return best_k

best_k = evaluar_k(X, max_k=8)
```

---

## 29.3 Clustering Jerárquico

```python
# Dendrograma
fig, ax = plt.subplots(figsize=(12, 8))

# Muestra pequeña para visualización
X_sample = X[:50]

Z = linkage(X_sample, method='ward')
dendrogram(Z, ax=ax, truncate_mode='level', p=5)
ax.set_title('Dendrograma - Clustering Jerárquico', fontweight='bold')
ax.set_xlabel('Muestras')
ax.set_ylabel('Distancia')
plt.savefig('dendrograma.png', dpi=150, bbox_inches='tight')
plt.show()

# Clustering jerárquico con 3 clusters
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
clientes['cluster_hc'] = hc.fit_predict(X)

print("=== CLUSTERING JERÁRQUICO ===")
print(f"Silhouette Score: {silhouette_score(X, clientes['cluster_hc']):.4f}")
```

---

## 29.4 DBSCAN

Detecta clusters de **forma arbitraria** y **outliers**:

```python
# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
clientes['cluster_dbscan'] = dbscan.fit_predict(X)

n_clusters = len(set(clientes['cluster_dbscan'])) - (1 if -1 in clientes['cluster_dbscan'] else 0)
n_outliers = (clientes['cluster_dbscan'] == -1).sum()

print("=== DBSCAN ===")
print(f"Clusters encontrados: {n_clusters}")
print(f"Outliers detectados: {n_outliers} ({n_outliers/len(clientes):.1%})")
print(f"Silhouette Score: {silhouette_score(X[clientes['cluster_dbscan'] != -1], 
                                           clientes.loc[clientes['cluster_dbscan'] != -1, 'cluster_dbscan']):.4f}")
```

---

## 29.5 Comparación de Métodos

```python
# Visualización comparativa
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, col, titulo in zip(axes, 
                            ['cluster_kmeans', 'cluster_hc', 'cluster_dbscan'],
                            ['K-Means (k=3)', 'Jerárquico', 'DBSCAN']):
    scatter = ax.scatter(X[:, 0], X[:, 1], c=clientes[col], 
                         cmap='viridis', s=20, alpha=0.7)
    ax.set_title(titulo, fontweight='bold')
    ax.set_xlabel('Gasto (std)')
    ax.set_ylabel('Frecuencia (std)')

plt.tight_layout()
plt.savefig('comparacion_clustering.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 29.6 Perfilamiento de Clusters

```python
def perfilar_clusters(df, col_cluster, vars_perfil):
    """Perfila los clusters con estadísticas descriptivas"""
    perfil = df.groupby(col_cluster)[vars_perfil].agg(['mean', 'std', 'count'])
    perfil.columns = ['_'.join(col).strip() for col in perfil.columns.values]
    return perfil

# Perfil de los clusters K-Means
perfil = perfilar_clusters(clientes, 'cluster_kmeans', 
                             ['gasto_anual', 'frecuencia', 'antiguedad'])
print("=== PERFIL DE CLUSTERS ===")
print(perfil.round(1))

# Nombrar clusters
nombres = {
    0: 'Bajo valor', 
    1: 'Medio valor', 
    2: 'Alto valor'
}
clientes['segmento'] = clientes['cluster_kmeans'].map(nombres)
print(f"\nDistribución de segmentos:")
print(clientes['segmento'].value_counts())
```

---

## 29.7 Visualización de Clusters en 2D

```python
# PCA para visualizar clusters
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

fig, ax = plt.subplots(figsize=(10, 8))
for cluster in sorted(clientes['cluster_kmeans'].unique()):
    mask = clientes['cluster_kmeans'] == cluster
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1], 
               label=f'{nombres[cluster]}', s=50, alpha=0.7, edgecolors='white')

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontweight='bold')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontweight='bold')
ax.set_title('Clusters K-Means - Proyección PCA', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.savefig('clusters_pca.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 29.8 Ejemplo en Ventas: Segmentación de Clientes

```python
# Segmentación completa de clientes para estrategia comercial
np.random.seed(42)
n = 1000

clientes_ventas = pd.DataFrame({
    'ingreso_anual': np.random.gamma(5, 10000, n),
    'frecuencia_compra': np.random.poisson(8, n),
    'ticket_promedio': np.random.gamma(3, 50, n),
    'antiguedad_meses': np.random.randint(1, 60, n),
    'devoluciones': np.random.poisson(0.5, n),
    'canal_preferido': np.random.choice([0, 1, 2], n)  # 0: online, 1: tienda, 2: ambos
})

# Estandarizar y clusterizar
scaler_v = StandardScaler()
X_v = scaler_v.fit_transform(clientes_ventas)

kmeans_v = KMeans(n_clusters=4, random_state=42, n_init=10)
clientes_ventas['segmento'] = kmeans_v.fit_predict(X_v)

print("=== SEGMENTACIÓN DE CLIENTES ===")
segmentos = clientes_ventas.groupby('segmento').agg({
    'ingreso_anual': 'mean',
    'frecuencia_compra': 'mean',
    'ticket_promedio': 'mean',
    'antiguedad_meses': 'mean',
    'devoluciones': 'mean'
}).round(1)
print(segmentos)

# Estrategia por segmento
for seg in sorted(clientes_ventas['segmento'].unique()):
    datos_seg = clientes_ventas[clientes_ventas['segmento'] == seg]
    print(f"\nSegmento {seg} ({len(datos_seg)} clientes):")
    print(f"  Ingreso: ${datos_seg['ingreso_anual'].mean():.0f}")
    print(f"  Frecuencia: {datos_seg['frecuencia_compra'].mean():.1f}/año")
    print(f"  Ticket: ${datos_seg['ticket_promedio'].mean():.1f}")
```

---

## 29.9 Ejemplo en Compras: Segmentación de Proveedores

```python
# Segmentación de proveedores para estrategia de compras
np.random.seed(42)
n = 80

proveedores_seg = pd.DataFrame({
    'precio': np.random.normal(100, 20, n),
    'calidad_puntaje': np.random.uniform(3, 5, n),
    'lead_time_dias': np.random.exponential(5, n) + 2,
    'confiabilidad': np.random.uniform(0.6, 1.0, n),
    'volumen_minimo': np.random.uniform(100, 5000, n),
    'distancia_km': np.random.uniform(10, 800, n)
})

scaler_p = StandardScaler()
X_p = scaler_p.fit_transform(proveedores_seg)

kmeans_p = KMeans(n_clusters=3, random_state=42, n_init=10)
proveedores_seg['cluster'] = kmeans_p.fit_predict(X_p)

print("=== SEGMENTACIÓN DE PROVEEDORES ===")
perfil_prov = proveedores_seg.groupby('cluster').mean().round(2)
print(perfil_prov)

# Estrategias de compra por cluster
for cluster in sorted(proveedores_seg['cluster'].unique()):
    datos = proveedores_seg[proveedores_seg['cluster'] == cluster]
    print(f"\nCluster {cluster} ({len(datos)} proveedores):")
    print(f"  Precio: ${datos['precio'].mean():.0f}")
    print(f"  Calidad: {datos['calidad_puntaje'].mean():.2f}/5")
    print(f"  Lead time: {datos['lead_time_dias'].mean():.1f} días")
```

---

## 29.10 Ejemplo en Inventarios: Clasificación de Productos

```python
# Segmentación de productos para gestión de inventario
np.random.seed(42)
n = 200

productos_inv = pd.DataFrame({
    'demanda_mensual': np.random.gamma(3, 30, n),
    'valor_unitario': np.random.uniform(5, 500, n),
    'lead_time': np.random.choice([2, 3, 5, 7, 10], n, p=[0.2, 0.3, 0.3, 0.1, 0.1]),
    'volatilidad': np.random.exponential(0.3, n),
    'rotacion_dias': np.random.exponential(20, n) + 5,
    'costo_almacenamiento': np.random.uniform(0.5, 5, n)
})

scaler_inv = StandardScaler()
X_inv = scaler_inv.fit_transform(productos_inv)

kmeans_inv = KMeans(n_clusters=4, random_state=42, n_init=10)
productos_inv['cluster'] = kmeans_inv.fit_predict(X_inv)

print("=== CLASIFICACIÓN DE PRODUCTOS (INVENTARIO) ===")
perfil_inv = productos_inv.groupby('cluster').agg({
    'demanda_mensual': 'mean',
    'valor_unitario': 'mean',
    'lead_time': 'mean',
    'volatilidad': 'mean',
    'rotacion_dias': 'mean'
}).round(1)
print(perfil_inv)

# Estrategia de inventario por cluster
for cluster in sorted(productos_inv['cluster'].unique()):
    datos = productos_inv[productos_inv['cluster'] == cluster]
    print(f"\nCluster {cluster}:")
    print(f"  Demanda: {datos['demanda_mensual'].mean():.0f}/mes")
    print(f"  Valor: ${datos['valor_unitario'].mean():.0f}")
    print(f"  Rotación: cada {datos['rotacion_dias'].mean():.0f} días")
```

---

## 29.11 Resumen

| Método | Ventajas | Desventajas |
|--------|----------|-------------|
| **K-Means** | Rápido, simple, escalable | Requiere k, clusters esféricos |
| **Jerárquico** | Dendrograma informativo, no requiere k | No escala bien (O(n³)) |
| **DBSCAN** | Detecta outliers, formas arbitrarias | Sensible a parámetros eps/min_samples |
| **GMM** | Probabilístico, clusters elípticos | Complejo, puede sobreajustar |

### Métricas de Evaluación
- **Inercia**: Suma de distancias intra-cluster (menor = mejor)
- **Silhouette Score**: [-1, 1] cohesión vs separación
- **Calinski-Harabasz**: Ratio varianza entre/dentro de clusters

---

## Ejercicios Propuestos

1. **Ventas**: Segmenta 500 clientes por gasto anual y frecuencia. ¿Qué estrategia comercial para cada segmento?
2. **Compras**: Clasifica proveedores en 3 grupos según precio, calidad y lead time. Describe cada grupo
3. **Inventarios**: Aplica K-Means a productos con demanda, valor y rotación. ¿Qué política de inventario para cada cluster?

---

[← Anterior](28-analisis-multivariante.md) | [Índice](index.md) | [Siguiente →](30-arboles-decision.md)
