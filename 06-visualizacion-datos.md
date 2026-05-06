# Capítulo 6: Visualización de Datos

[← Anterior](05-medidas-dispersion.md) | [Índice](index.md) | [Siguiente →](07-distribuciones-discretas.md)

---

## 6.1 Principios de Visualización

La visualización de datos es la representación gráfica de información para **revelar patrones, tendencias y anomalías** que no son evidentes en datos tabulares.

### Principios Clave (Edward Tufte)

1. **Mostrar los datos** - Sin distorsión
2. **Maximizar la tinta de datos** - Eliminar decoración innecesaria
3. **Facilitar comparaciones** - Misma escala, mismo eje
4. **Revelar múltiples niveles** - Desde lo general a lo detallado

### Tipos de Gráficos según el Objetivo

| Objetivo | Gráfico Recomendado |
|----------|---------------------|
| Comparar categorías | Barras, columnas |
| Mostrar distribución | Histograma, boxplot, violín |
| Relación entre variables | Dispersión, burbujas |
| Tendencia temporal | Líneas, área |
| Composición | Pastel, stacked bars, treemap |
| Correlación | Heatmap, pairplot |

---

## 6.2 Configuración de Matplotlib y Seaborn

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter, PercentFormatter
import warnings
warnings.filterwarnings('ignore')

# Configuración profesional
plt.rcParams.update({
    'figure.figsize': (12, 6),
    'figure.dpi': 150,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

sns.set_theme(style="whitegrid", palette="Set2")

# Datos de ejemplo
np.random.seed(42)
n = 500

df = pd.DataFrame({
    'fecha': pd.date_range('2024-01-01', periods=n, freq='D'),
    'ventas': np.random.normal(1000, 200, n).cumsum() + 50000,
    'producto': np.random.choice(['A', 'B', 'C', 'D'], n),
    'region': np.random.choice(['Norte', 'Sur', 'Este', 'Oeste'], n),
    'unidades': np.random.poisson(30, n),
    'precio': np.random.gamma(5, 20, n) + 50,
    'descuento': np.random.choice([0, 5, 10, 15, 20], n, p=[0.4, 0.3, 0.15, 0.1, 0.05]),
    'categoria': np.random.choice(['Premium', 'Estándar', 'Económico'], n, p=[0.2, 0.5, 0.3])
})

df['ingreso'] = df['unidades'] * df['precio'] * (1 - df['descuento']/100)
df['mes'] = df['fecha'].dt.month
```

---

## 6.3 Gráficos de Barras y Columnas

### Barras Simples

```python
fig, ax = plt.subplots(figsize=(10, 6))

ventas_producto = df.groupby('producto')['ingreso'].sum().sort_values(ascending=False)
bars = ax.bar(ventas_producto.index, ventas_producto.values, 
              color=sns.color_palette("Set2"), edgecolor='white', linewidth=1.5)

# Etiquetas en las barras
for bar, val in zip(bars, ventas_producto.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'${val:,.0f}', ha='center', va='bottom', fontweight='bold')

ax.set_title('Ingresos por Producto', fontweight='bold', pad=15)
ax.set_xlabel('Producto')
ax.set_ylabel('Ingreso Total ($)')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))

plt.tight_layout()
plt.savefig('barras_productos.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Barras Agrupadas

```python
fig, ax = plt.subplots(figsize=(12, 6))

ventas_region_producto = df.pivot_table(
    values='ingreso', index='region', columns='producto', aggfunc='sum'
)

ventas_region_producto.plot(kind='bar', ax=ax, edgecolor='white', linewidth=1)
ax.set_title('Ingresos por Región y Producto', fontweight='bold', pad=15)
ax.set_xlabel('Región')
ax.set_ylabel('Ingreso ($)')
ax.legend(title='Producto', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))

plt.tight_layout()
plt.savefig('barras_agrupadas.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 6.4 Histogramas

Los **histogramas** muestran la distribución de frecuencias de una variable numérica dividiendo el rango de valores en intervalos (bins). Son ideales para visualizar la forma de la distribución, detectar asimetrías, identificar valores atípicos y comparar la media con la mediana. En el contexto de ventas, un histograma de precios revela si la mayoría de productos se concentran en un rango económico o si hay segmentos claramente diferenciados. La superposición de la media y mediana ayuda a diagnosticar asimetría: si difieren significativamente, la distribución no es simétrica y la mediana es más representativa que la media.

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Histograma básico
axes[0].hist(df['precio'], bins=30, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].axvline(df['precio'].mean(), color='red', linestyle='--', linewidth=2, label='Media')
axes[0].axvline(df['precio'].median(), color='green', linestyle='--', linewidth=2, label='Mediana')
axes[0].set_title('Distribución de Precios', fontweight='bold')
axes[0].set_xlabel('Precio ($)')
axes[0].set_ylabel('Frecuencia')
axes[0].legend()

# Histograma con KDE
sns.histplot(df['ventas'], bins=40, kde=True, ax=axes[1], color='forestgreen')
axes[1].set_title('Distribución de Ventas Acumuladas', fontweight='bold')
axes[1].set_xlabel('Ventas ($)')

# Histograma por categoría
for i, cat in enumerate(['Premium', 'Estándar', 'Económico']):
    subset = df[df['categoria'] == cat]['precio']
    axes[2].hist(subset, bins=15, alpha=0.5, label=cat, edgecolor='white')

axes[2].set_title('Distribución de Precios por Categoría', fontweight='bold')
axes[2].set_xlabel('Precio ($)')
axes[2].set_ylabel('Frecuencia')
axes[2].legend()

plt.tight_layout()
plt.savefig('histogramas.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 6.5 Boxplots (Diagramas de Caja)

Los **boxplots** (diagramas de caja y bigotes) resumen la distribución mediante cinco números: mínimo, Q1, mediana, Q3 y máximo. La caja contiene el 50% central de los datos (el rango intercuartílico IQR), y los bigotes se extienden hasta 1.5×IQR más allá de los cuartiles. Los puntos más allá de los bigotes son potenciales outliers. El boxplot es especialmente útil para comparar distribuciones entre categorías: por ejemplo, al comparar precios entre categorías de producto, podemos ver de un vistazo si una categoría tiene mayor variabilidad, si su mediana es más alta o si tiene valores extremos. A diferencia del histograma, el boxplot no muestra la forma multimodal de la distribución, pero es más compacto para comparar múltiples grupos simultáneamente.

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Boxplot simple
sns.boxplot(data=df, x='categoria', y='precio', ax=axes[0], palette='Set2')
axes[0].set_title('Distribución de Precios por Categoría', fontweight='bold')
axes[0].set_xlabel('Categoría')
axes[0].set_ylabel('Precio ($)')

# Boxplot con más variables
sns.boxplot(data=df, x='region', y='ingreso', hue='producto', ax=axes[1], palette='Set2')
axes[1].set_title('Ingresos por Región y Producto', fontweight='bold')
axes[1].set_xlabel('Región')
axes[1].set_ylabel('Ingreso ($)')
axes[1].legend(title='Producto', bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.savefig('boxplots.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 6.6 Gráficos de Dispersión (Scatter)

Los **gráficos de dispersión** (scatter plots) muestran la relación entre dos variables numéricas. Cada punto representa una observación con sus coordenadas (X, Y). Son la herramienta principal para detectar correlaciones, patrones no lineales, clusters y outliers. En negocios, un scatter de precio vs unidades vendidas puede revelar la elasticidad precio de la demanda: si los puntos siguen una tendencia descendente clara, indica que a mayor precio se venden menos unidades. Cuando se añade una línea de regresión (con `regplot`), se visualiza la tendencia lineal central, aunque la relación real podría ser curvilínea. Es fundamental examinar siempre el scatter antes de calcular cualquier coeficiente de correlación, ya que correlaciones cercanas a cero pueden ocultar relaciones no lineales fuertes.

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Dispersión simple
axes[0].scatter(df['precio'], df['unidades'], alpha=0.5, c='steelblue', edgecolors='white')
axes[0].set_title('Relación Precio vs Unidades Vendidas', fontweight='bold')
axes[0].set_xlabel('Precio ($)')
axes[0].set_ylabel('Unidades Vendidas')

# Dispersión con regresión
sns.regplot(data=df, x='precio', y='unidades', ax=axes[1], 
            scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
axes[1].set_title('Regresión: Precio → Unidades', fontweight='bold')
axes[1].set_xlabel('Precio ($)')
axes[1].set_ylabel('Unidades Vendidas')

plt.tight_layout()
plt.savefig('scatter.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 6.7 Series de Tiempo (Líneas)

```python
fig, ax = plt.subplots(figsize=(14, 6))

ventas_diarias = df.set_index('fecha')['ventas']
ventas_diarias.plot(ax=ax, color='steelblue', linewidth=1, alpha=0.7)

# Media móvil de 7 días
ventas_diarias.rolling(7).mean().plot(ax=ax, color='red', linewidth=2, label='Media Móvil 7d')

# Media móvil de 30 días
ventas_diarias.rolling(30).mean().plot(ax=ax, color='darkgreen', linewidth=2, label='Media Móvil 30d')

ax.set_title('Evolución de Ventas Acumuladas', fontweight='bold')
ax.set_xlabel('Fecha')
ax.set_ylabel('Ventas ($)')
ax.legend()
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))

plt.tight_layout()
plt.savefig('series_tiempo.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 6.8 Mapas de Calor (Heatmap)

```python
fig, ax = plt.subplots(figsize=(10, 8))

# Tabla pivote: mes × producto
tabla_mes_producto = df.pivot_table(
    values='ingreso', index='mes', columns='producto', aggfunc='sum'
)

sns.heatmap(tabla_mes_producto, annot=True, fmt='.0f', cmap='YlOrRd',
            linewidths=1, ax=ax)
ax.set_title('Ingresos: Mes × Producto', fontweight='bold', pad=15)
ax.set_xlabel('Producto')
ax.set_ylabel('Mes')

plt.tight_layout()
plt.savefig('heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 6.9 Pairplot (Matriz de Dispersión)

```python
# Matriz de relaciones entre variables numéricas
variables = df[['unidades', 'precio', 'ingreso', 'descuento']]
variables['descuento'] = variables['descuento'].astype(float)

g = sns.pairplot(variables, diag_kind='kde', plot_kws={'alpha': 0.5, 's': 15})
g.fig.suptitle('Matriz de Relaciones entre Variables', y=1.02, fontweight='bold')

plt.savefig('pairplot.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 6.10 Gráficos Avanzados para Negocios

### Gráfico de Pareto (80-20)

```python
fig, ax1 = plt.subplots(figsize=(12, 6))

# Ordenar productos por ingreso descendente
ingresos_productos = df.groupby('producto')['ingreso'].sum().sort_values(ascending=False)
pct_acumulado = ingresos_productos.cumsum() / ingresos_productos.sum() * 100

# Barras
bars = ax1.bar(ingresos_productos.index, ingresos_productos.values,
               color=sns.color_palette("Set2"), edgecolor='white', linewidth=1.5)
ax1.set_ylabel('Ingreso ($)', fontweight='bold')
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Línea de porcentaje acumulado
ax2 = ax1.twinx()
ax2.plot(ingresos_productos.index, pct_acumulado, 'ro-', linewidth=2, markersize=8)
ax2.axhline(y=80, color='gray', linestyle='--', alpha=0.5)
ax2.set_ylabel('% Acumulado', fontweight='bold')
ax2.set_ylim(0, 105)

ax1.set_title('Análisis de Pareto: Ingresos por Producto', fontweight='bold')

plt.tight_layout()
plt.savefig('pareto.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Gráfico de Violín

```python
fig, ax = plt.subplots(figsize=(10, 6))

sns.violinplot(data=df, x='region', y='ingreso', hue='categoria', 
               split=False, palette='Set2', ax=ax)
ax.set_title('Distribución de Ingresos: Región × Categoría', fontweight='bold')
ax.set_xlabel('Región')
ax.set_ylabel('Ingreso ($)')
ax.legend(title='Categoría', bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.savefig('violin.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 6.11 Ejemplo: Dashboard Completo de Ventas

```python
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Serie de tiempo
ax1 = fig.add_subplot(gs[0, :])
ventas_diarias = df.set_index('fecha')['ingreso'].resample('W').sum()
ventas_diarias.plot(ax=ax1, color='steelblue', linewidth=1.5)
ventas_diarias.rolling(4).mean().plot(ax=ax1, color='red', linewidth=2, label='Media Móvil 4s')
ax1.set_title('Ingresos Semanales', fontweight='bold')
ax1.legend()

# 2. Barras por producto
ax2 = fig.add_subplot(gs[1, 0])
df.groupby('producto')['ingreso'].sum().plot(kind='bar', ax=ax2, color='steelblue', edgecolor='white')
ax2.set_title('Ingresos por Producto', fontweight='bold')
ax2.set_xlabel('')

# 3. Boxplot por región
ax3 = fig.add_subplot(gs[1, 1])
sns.boxplot(data=df, x='region', y='ingreso', ax=ax3, palette='Set2')
ax3.set_title('Distribución por Región', fontweight='bold')

# 4. Histograma de precios
ax4 = fig.add_subplot(gs[1, 2])
df['precio'].hist(ax=ax4, bins=30, color='forestgreen', edgecolor='white')
ax4.set_title('Distribución de Precios', fontweight='bold')

# 5. Dispersión
ax5 = fig.add_subplot(gs[2, 0])
ax5.scatter(df['precio'], df['unidades'], alpha=0.3, c='steelblue', edgecolors='white')
ax5.set_title('Precio vs Unidades', fontweight='bold')

# 6. Barras por categoría
ax6 = fig.add_subplot(gs[2, 1])
df.groupby('categoria')['ingreso'].sum().plot(kind='bar', ax=ax6, color=sns.color_palette("Set2"))
ax6.set_title('Ingresos por Categoría', fontweight='bold')
ax6.set_xlabel('')

# 7. Pastel de regiones
ax7 = fig.add_subplot(gs[2, 2])
df.groupby('region')['ingreso'].sum().plot(kind='pie', ax=ax7, autopct='%1.1f%%',
    colors=sns.color_palette("Set2"), startangle=90)
ax7.set_title('Distribución por Región', fontweight='bold')
ax7.set_ylabel('')

fig.suptitle('DASHBOARD DE VENTAS - ANÁLISIS COMPLETO', 
             fontsize=16, fontweight='bold', y=1.02)

plt.savefig('dashboard_ventas.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 6.12 Ejemplo: Visualización para Compras

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Datos de compras
compras_df = pd.DataFrame({
    'proveedor': np.random.choice(['Prov_A', 'Prov_B', 'Prov_C'], 300),
    'costo': np.random.normal(100, 20, 300),
    'lead_time': np.random.exponential(5, 300) + 2,
    'calidad': np.random.choice(['A', 'B', 'C'], 300, p=[0.7, 0.2, 0.1]),
    'volumen': np.random.randint(100, 10000, 300)
})

# 1. Scatter: costo vs lead_time por proveedor
colors = {'Prov_A': 'steelblue', 'Prov_B': 'forestgreen', 'Prov_C': 'coral'}
for prov, color in colors.items():
    subset = compras_df[compras_df['proveedor'] == prov]
    axes[0, 0].scatter(subset['costo'], subset['lead_time'], 
                       c=color, label=prov, alpha=0.5, edgecolors='white')
axes[0, 0].set_xlabel('Costo ($)')
axes[0, 0].set_ylabel('Lead Time (días)')
axes[0, 0].set_title('Costo vs Tiempo de Entrega por Proveedor', fontweight='bold')
axes[0, 0].legend()

# 2. Barras de calidad
calidad_counts = compras_df.groupby(['proveedor', 'calidad']).size().unstack(fill_value=0)
calidad_pct = calidad_counts.div(calidad_counts.sum(axis=1), axis=0)
calidad_pct.plot(kind='bar', stacked=True, ax=axes[0, 1], colormap='Set2', edgecolor='white')
axes[0, 1].set_title('Calidad por Proveedor', fontweight='bold')
axes[0, 1].set_xlabel('Proveedor')
axes[0, 1].set_ylabel('Proporción')
axes[0, 1].legend(title='Calidad')

# 3. Boxplot lead time por proveedor
sns.boxplot(data=compras_df, x='proveedor', y='lead_time', ax=axes[1, 0], palette='Set2')
axes[1, 0].set_title('Lead Time por Proveedor', fontweight='bold')
axes[1, 0].set_xlabel('Proveedor')
axes[1, 0].set_ylabel('Lead Time (días)')

# 4. Histograma de costos
for prov, color in colors.items():
    subset = compras_df[compras_df['proveedor'] == prov]['costo']
    axes[1, 1].hist(subset, bins=20, alpha=0.5, label=prov, color=color, edgecolor='white')
axes[1, 1].set_title('Distribución de Costos por Proveedor', fontweight='bold')
axes[1, 1].set_xlabel('Costo ($)')
axes[1, 1].set_ylabel('Frecuencia')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('dashboard_compras.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 6.13 Ejemplo: Visualización para Inventarios

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Datos de inventario
inv_df = pd.DataFrame({
    'sku': [f'SKU_{i}' for i in range(50)],
    'stock': np.random.randint(0, 500, 50),
    'demanda': np.random.exponential(30, 50).round(),
    'costo': np.random.uniform(10, 200, 50),
    'rotacion': np.random.exponential(10, 50).round(1)
})
inv_df['valor'] = inv_df['stock'] * inv_df['costo']

# 1. Histograma de stock
axes[0, 0].hist(inv_df['stock'], bins=20, color='steelblue', edgecolor='white')
axes[0, 0].axvline(inv_df['stock'].mean(), color='red', linestyle='--', linewidth=2, label='Media')
axes[0, 0].set_title('Distribución de Niveles de Stock', fontweight='bold')
axes[0, 0].set_xlabel('Stock (unidades)')
axes[0, 0].set_ylabel('Frecuencia')
axes[0, 0].legend()

# 2. Scatter stock vs demanda
axes[0, 1].scatter(inv_df['stock'], inv_df['demanda'], alpha=0.6, 
                   c=inv_df['valor'], s=inv_df['costo']*2, edgecolors='white')
axes[0, 1].set_xlabel('Stock (unidades)')
axes[0, 1].set_ylabel('Demanda Diaria (unidades)')
axes[0, 1].set_title('Stock vs Demanda (tamaño = costo)', fontweight='bold')
cbar = plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1])
cbar.set_label('Valor Inventario ($)')

# 3. Boxplot de rotación
sns.boxplot(data=inv_df, y='rotacion', ax=axes[1, 0], color='steelblue')
axes[1, 0].set_title('Distribución de Rotación de Inventario', fontweight='bold')
axes[1, 0].set_ylabel('Días entre rotaciones')

# 4. Pareto de valor de inventario
inv_ordenado = inv_df.sort_values('valor', ascending=False).reset_index(drop=True)
inv_ordenado['pct_acum'] = inv_ordenado['valor'].cumsum() / inv_ordenado['valor'].sum() * 100

ax_pareto = axes[1, 1]
ax_pareto.bar(range(len(inv_ordenado)), inv_ordenado['valor'], color='steelblue', alpha=0.7)
ax_pareto_twin = ax_pareto.twinx()
ax_pareto_twin.plot(inv_ordenado['pct_acum'], 'r-', linewidth=2)
ax_pareto_twin.axhline(80, color='gray', linestyle='--', alpha=0.5)
ax_pareto.set_title('Pareto de Valor de Inventario', fontweight='bold')
ax_pareto.set_xlabel('SKU (ordenado)')
ax_pareto.set_ylabel('Valor ($)')
ax_pareto_twin.set_ylabel('% Acumulado')

plt.tight_layout()
plt.savefig('dashboard_inventarios.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 6.14 Resumen

- La visualización es esencial para entender patrones en datos
- Cada tipo de gráfico tiene un propósito específico
- Matplotlib ofrece control total; Seaborn simplifica gráficos estadísticos
- Los dashboards combinan múltiples vistas para análisis completo
- Un buen gráfico revela información que los números tabulares ocultan

---

## Ejercicios Propuestos

1. **Ventas**: Crea un gráfico de líneas mostrando ventas diarias con media móvil de 7 y 30 días
2. **Compras**: Genera un scatter plot de costo vs calidad por proveedor, usando colores diferentes
3. **Inventarios**: Realiza un heatmap de stock por categoría y ubicación en el almacén

---

[← Anterior](05-medidas-dispersion.md) | [Índice](index.md) | [Siguiente →](07-distribuciones-discretas.md)
