# Capítulo 28: Análisis Multivariante y PCA

[← Anterior](27-bootstrap.md) | [Índice](index.md) | [Siguiente →](29-clustering.md)

---

## 28.1 Introducción

El **análisis multivariante** estudia múltiples variables simultáneamente. Cuando tenemos muchas variables, surgen problemas como la **maldición de la dimensionalidad** y la **multicolinealidad**.

### Técnicas Principales

| Técnica | Propósito | Tipo |
|---------|-----------|------|
| **PCA** (ACP) | Reducir dimensionalidad | No supervisado |
| **Análisis Factorial** | Identificar factores latentes | No supervisado |
| **MCA** (Análisis de Correspondencias) | Variables categóricas | No supervisado |
| **PLS** (Partial Least Squares) | Regresión con muchos predictores | Supervisado |

---

## 28.2 Análisis de Componentes Principales (PCA)

PCA transforma las variables originales en **componentes principales** no correlacionados que capturan la máxima varianza.

### Matemática

PCA busca las direcciones de máxima varianza en los datos resolviendo el problema de autovalores de la matriz de covarianza $\Sigma$:

$$\Sigma \mathbf{v}_i = \lambda_i \mathbf{v}_i$$

Donde $\lambda_i$ son los **autovalores** (varianza explicada por cada componente) y $\mathbf{v}_i$ son los **autovectores** (direcciones de los componentes). Cada componente principal es una combinación lineal de las variables originales:

$$PC_i = a_{i1}X_1 + a_{i2}X_2 + ... + a_{ip}X_p$$

La **proporción de varianza explicada** por el componente $i$ es:

$$\frac{\lambda_i}{\sum_{j=1}^{p} \lambda_j}$$

Esto permite decidir cuántos componentes retener: normalmente se busca que la varianza acumulada supere el 70-80% o se usa el criterio de Kaiser (autovalores > 1).

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

np.random.seed(42)

# Datos multivariantes: rendimiento de tiendas
n_tiendas = 100

tiendas = pd.DataFrame({
    'ventas': np.random.normal(500000, 100000, n_tiendas),
    'clientes': np.random.poisson(2000, n_tiendas),
    'ticket_promedio': np.random.normal(250, 50, n_tiendas),
    'area_m2': np.random.normal(500, 200, n_tiendas),
    'empleados': np.random.poisson(15, n_tiendas),
    'antiguedad': np.random.randint(1, 20, n_tiendas),
    'inventario_prom': np.random.gamma(5, 100000, n_tiendas),
    'satisfaccion': np.random.uniform(3.5, 5, n_tiendas)
})

# Estandarizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(tiendas)

# PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print("=== ANÁLISIS DE COMPONENTES PRINCIPALES ===")
print(f"Número de variables: {X_scaled.shape[1]}")
print(f"Número de componentes: {pca.n_components_}")

# Varianza explicada
varianza_explicada = pca.explained_variance_ratio_
varianza_acumulada = np.cumsum(varianza_explicada)

print(f"\nVarianza explicada por componente:")
for i, (ve, va) in enumerate(zip(varianza_explicada, varianza_acumulada), 1):
    print(f"  PC{i}: {ve:.2%} (acumulado: {va:.2%})")

# Número de componentes para 80% de varianza
n_80 = np.argmax(varianza_acumulada >= 0.80) + 1
print(f"\n→ {n_80} componentes explican el 80% de la varianza")
```

### Scree Plot

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scree plot
axes[0].plot(range(1, len(varianza_explicada) + 1), varianza_explicada, 'bo-', linewidth=2, markersize=8)
axes[0].plot(range(1, len(varianza_explicada) + 1), varianza_acumulada, 'ro--', linewidth=2, markersize=8)
axes[0].axhline(y=0.80, color='gray', linestyle='--', alpha=0.5, label='80%')
axes[0].set_xlabel('Componente Principal', fontweight='bold')
axes[0].set_ylabel('Varianza Explicada', fontweight='bold')
axes[0].set_title('Scree Plot', fontweight='bold')
axes[0].legend(['Individual', 'Acumulada'])

# Cargas (loadings)
loadings = pd.DataFrame(
    pca.components_.T,
    index=tiendas.columns,
    columns=[f'PC{i+1}' for i in range(pca.n_components_)]
)

sns.heatmap(loadings.iloc[:, :4], annot=True, fmt='.2f', cmap='RdBu_r', 
            center=0, ax=axes[1])
axes[1].set_title('Cargas (Loadings) de las Variables', fontweight='bold')

plt.tight_layout()
plt.savefig('pca_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Interpretación de loadings
print("\n=== INTERPRETACIÓN DE COMPONENTES ===")
for i in range(min(3, pca.n_components_)):
    pc = loadings.iloc[:, i]
    print(f"\nPC{i+1} ({varianza_explicada[i]:.1%} de varianza):")
    print(f"  Variables con mayor carga positiva: {pc.nlargest(3).to_dict()}")
    print(f"  Variables con mayor carga negativa: {pc.nsmallest(3).to_dict()}")
```

---

## 28.3 Visualización con PCA

```python
# Proyectar datos en 2D
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

df_2d = pd.DataFrame({
    'PC1': X_2d[:, 0],
    'PC2': X_2d[:, 1],
    'ventas': tiendas['ventas'],
    'tamaño': tiendas['area_m2']
})

# Crear categorías de ventas para color
df_2d['categoria_ventas'] = pd.qcut(df_2d['ventas'], 4, labels=['Bajo', 'Medio-Bajo', 'Medio-Alto', 'Alto'])

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(df_2d['PC1'], df_2d['PC2'], 
                     c=df_2d['ventas'], s=df_2d['tamaño']/5,
                     cmap='viridis', alpha=0.7, edgecolors='white')
ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})', fontweight='bold')
ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})', fontweight='bold')
ax.set_title('Proyección PCA - Tiendas (color=ventas, tamaño=área)', fontweight='bold')
plt.colorbar(scatter, label='Ventas ($)')
ax.grid(alpha=0.3)
plt.savefig('pca_proyeccion.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 28.4 Análisis Factorial

Identifica **factores latentes** que explican las correlaciones entre variables observadas:

```python
from sklearn.decomposition import FactorAnalysis

# Análisis factorial con 3 factores
fa = FactorAnalysis(n_components=3, random_state=42)
factores = fa.fit_transform(X_scaled)

# Cargas factoriales
cargas_fa = pd.DataFrame(
    fa.components_.T,
    index=tiendas.columns,
    columns=['Factor 1', 'Factor 2', 'Factor 3']
)

print("=== ANÁLISIS FACTORIAL ===")
print("Cargas factoriales (rotadas):")
print(cargas_fa.round(3))

# Varianza explicada por cada factor
varianza_fa = np.var(factores, axis=0)
prop_varianza_fa = varianza_fa / varianza_fa.sum()
print(f"\nProporción de varianza por factor:")
for i, pv in enumerate(prop_varianza_fa, 1):
    print(f"  Factor {i}: {pv:.2%}")

# Interpretación
print("\n=== INTERPRETACIÓN DE FACTORES ===")
for i in range(3):
    factor = cargas_fa.iloc[:, i]
    print(f"\nFactor {i+1}:")
    print(f"  + Alto: {factor.nlargest(3).index.tolist()}")
    print(f"  - Bajo: {factor.nsmallest(3).index.tolist()}")
```

---

## 28.5 PCA para Reducción de Dimensionalidad en Regresión

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Regresión sin PCA
X_orig = X_scaled
y = tiendas['ventas'] + np.random.normal(0, 20000, n_tiendas)

lr_orig = LinearRegression()
score_orig = cross_val_score(lr_orig, X_orig, y, cv=5, scoring='r2')

# Regresión con PCA (5 componentes)
pca_reg = PCA(n_components=5)
X_pca_reg = pca_reg.fit_transform(X_scaled)
lr_pca = LinearRegression()
score_pca = cross_val_score(lr_pca, X_pca_reg, y, cv=5, scoring='r2')

print("=== PCA PARA REGRESIÓN ===")
print(f"R² (original, {X_orig.shape[1]} vars): {score_orig.mean():.4f} ± {score_orig.std():.4f}")
print(f"R² (PCA, 5 comps): {score_pca.mean():.4f} ± {score_pca.std():.4f}")
print(f"Reducción: {X_orig.shape[1]} → 5 dimensiones")
```

---

## 28.6 Ejemplo en Ventas: Segmentación de Tiendas

```python
# Segmentar tiendas usando PCA + Clustering
from sklearn.cluster import KMeans

# PCA a 2 componentes
pca_seg = PCA(n_components=2)
X_seg = pca_seg.fit_transform(X_scaled)

# Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_seg)

df_seg = pd.DataFrame({
    'PC1': X_seg[:, 0], 'PC2': X_seg[:, 1],
    'cluster': clusters,
    'ventas': tiendas['ventas'],
    'area': tiendas['area_m2'],
    'empleados': tiendas['empleados']
})

print("=== SEGMENTACIÓN DE TIENDAS (PCA + KMEANS) ===")
print(df_seg.groupby('cluster').agg({
    'ventas': ['mean', 'count'],
    'area': 'mean',
    'empleados': 'mean'
}))

fig, ax = plt.subplots(figsize=(10, 8))
for cluster in sorted(df_seg['cluster'].unique()):
    subset = df_seg[df_seg['cluster'] == cluster]
    ax.scatter(subset['PC1'], subset['PC2'], label=f'Cluster {cluster}', 
               s=50, alpha=0.7, edgecolors='white')
ax.set_xlabel('PC1', fontweight='bold')
ax.set_ylabel('PC2', fontweight='bold')
ax.set_title('Segmentación de Tiendas (PCA + K-Means)', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.savefig('pca_clustering.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 28.7 Ejemplo en Compras: Análisis de Proveedores

```python
# Evaluación multivariante de proveedores
np.random.seed(42)
n_prov = 50

proveedores = pd.DataFrame({
    'precio': np.random.normal(100, 15, n_prov),
    'calidad': np.random.uniform(3, 5, n_prov),
    'lead_time': np.random.exponential(5, n_prov) + 2,
    'confiabilidad': np.random.uniform(0.7, 1.0, n_prov),
    'flexibilidad': np.random.uniform(1, 5, n_prov),
    'distancia_km': np.random.uniform(10, 500, n_prov),
    'volumen_min': np.random.uniform(100, 5000, n_prov)
})

# PCA
scaler_prov = StandardScaler()
X_prov = scaler_prov.fit_transform(proveedores)
pca_prov = PCA(n_components=2)
X_prov_pca = pca_prov.fit_transform(X_prov)

# Score compuesto (inverso del precio + calidad + confiabilidad)
proveedores['score'] = (
    -0.3 * (proveedores['precio'] - proveedores['precio'].mean()) / proveedores['precio'].std() +
    0.3 * (proveedores['calidad'] - proveedores['calidad'].mean()) / proveedores['calidad'].std() +
    0.2 * proveedores['confiabilidad'] +
    0.2 * proveedores['flexibilidad'] / 5
)

print("=== ANÁLISIS MULTIVARIANTE DE PROVEEDORES ===")
print(f"Top 5 proveedores:")
top_prov = proveedores.nlargest(5, 'score')
print(top_prov[['precio', 'calidad', 'lead_time', 'confiabilidad', 'score']].round(3))
```

---

## 28.8 Ejemplo en Inventarios: Clasificación Multivariante

```python
# Clasificación ABC multivariante (no solo por valor)
np.random.seed(42)
n_sku = 100

inventario = pd.DataFrame({
    'valor_inventario': np.random.gamma(2, 50000, n_sku),
    'rotacion': np.random.exponential(15, n_sku) + 1,
    'volatilidad_demanda': np.random.exponential(0.3, n_sku),
    'costo_stockout': np.random.uniform(100, 5000, n_sku),
    'lead_time': np.random.choice([2, 3, 5, 7, 10], n_sku, p=[0.2, 0.3, 0.3, 0.1, 0.1])
})

# PCA para visualización
scaler_inv = StandardScaler()
X_inv = scaler_inv.fit_transform(inventario)
pca_inv = PCA(n_components=2)
X_inv_pca = pca_inv.fit_transform(X_inv)

# Score de criticidad (multivariante)
inventario['criticidad'] = (
    0.3 * (inventario['valor_inventario'] - inventario['valor_inventario'].mean()) / inventario['valor_inventario'].std() +
    0.2 * (1/inventario['rotacion']) +  # Baja rotación = más crítico
    0.3 * inventario['volatilidad_demanda'] +
    0.2 * inventario['costo_stockout'] / inventario['costo_stockout'].mean()
)

inventario['clasificacion'] = pd.qcut(inventario['criticidad'], 3, labels=['C', 'B', 'A'])

print("=== CLASIFICACIÓN MULTIVARIANTE ABC ===")
print(inventario.groupby('clasificacion', observed=True).agg({
    'valor_inventario': 'mean',
    'rotacion': 'mean',
    'volatilidad_demanda': 'mean',
    'criticidad': 'mean'
}).round(2))
```

---

## 28.9 Resumen

| Técnica | Uso Principal | Output |
|---------|--------------|--------|
| **PCA** | Reducir dimensionalidad | Componentes no correlacionados |
| **Análisis Factorial** | Identificar factores latentes | Factores interpretables |
| **Análisis de Correspondencias** | Variables categóricas | Mapa perceptual |
| **PLS** | Regresión con muchos predictores | Componentes predictivos |

---

## Ejercicios Propuestos

1. **Ventas**: Aplica PCA a 10 variables de rendimiento de tiendas. ¿Cuántos componentes retienes?
2. **Compras**: Usa análisis factorial para identificar factores subyacentes en la evaluación de proveedores
3. **Inventarios**: Crea un score multivariante de criticidad de inventario combinando valor, rotación y volatilidad

---

[← Anterior](27-bootstrap.md) | [Índice](index.md) | [Siguiente →](29-clustering.md)
