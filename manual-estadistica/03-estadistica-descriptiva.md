# Capítulo 3: Estadística Descriptiva

[← Anterior](02-intro-probabilidad.md) | [Índice](index.md) | [Siguiente →](04-medidas-tendencia-central.md)

---

## 3.1 ¿Qué es la Estadística Descriptiva?

La estadística descriptiva es el conjunto de técnicas que permiten **resumir, organizar y presentar** datos de manera clara y significativa. Es el primer paso en cualquier análisis de datos y constituye la base para la inferencia estadística.

### Objetivos principales:
1. **Sintetizar** grandes volúmenes de datos
2. **Identificar** patrones y tendencias
3. **Detectar** anomalías y valores atípicos
4. **Comunicar** hallazgos de manera efectiva

---

## 3.2 Tablas de Frecuencia

### Para Variables Cualitativas

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

# Datos de categorías de producto
categorias = np.random.choice(
    ['Electrónica', 'Ropa', 'Hogar', 'Deportes', 'Libros'],
    size=500,
    p=[0.30, 0.25, 0.20, 0.15, 0.10]
)

# Tabla de frecuencias absolutas
freq_abs = pd.Series(categorias).value_counts()
print("=== FRECUENCIAS ABSOLUTAS ===")
print(freq_abs)

# Tabla de frecuencias relativas
freq_rel = pd.Series(categorias).value_counts(normalize=True)
print("\n=== FRECUENCIAS RELATIVAS ===")
print(freq_rel)

# Tabla completa
tabla_frecuencias = pd.DataFrame({
    'Categoría': freq_abs.index,
    'Frecuencia': freq_abs.values,
    'Porcentaje': (freq_rel.values * 100).round(2),
    'Frecuencia Acumulada': freq_abs.cumsum().values,
    '% Acumulado': (freq_rel.cumsum().values * 100).round(2)
}).reset_index(drop=True)
print("\n=== TABLA DE FRECUENCIAS COMPLETA ===")
print(tabla_frecuencias)
```

**Salida:**
```
=== FRECUENCIAS ABSOLUTAS ===
Electrónica    152
Ropa           132
Hogar           91
Deportes        74
Libros          51

=== TABLA DE FRECUENCIAS COMPLETA ===
     Categoría  Frecuencia  Porcentaje  Frecuencia Acumulada  % Acumulado
0  Electrónica         152       30.40                   152        30.40
1         Ropa         132       26.40                   284        56.80
2        Hogar          91       18.20                   375        75.00
3     Deportes          74       14.80                   449        89.80
4       Libros          51       10.20                   500       100.00
```

### Para Variables Cuantitativas (Datos Agrupados)

```python
# Datos de precios de productos
precios = np.random.exponential(scale=100, size=200).round(2)
precios = precios.clip(5, 500)  # Limitamos a rango realista

# Creación de intervalos (bins)
bins = [0, 50, 100, 150, 200, 300, 500]
etiquetas = ['0-50', '51-100', '101-150', '151-200', '201-300', '301-500']

precios_agrupados = pd.cut(precios, bins=bins, labels=etiquetas)
tabla_precios = pd.DataFrame({
    'Rango Precio': etiquetas,
    'Frecuencia': precios_agrupados.value_counts().sort_index().values,
    'Porcentaje': (precios_agrupados.value_counts(normalize=True).sort_index().values * 100).round(2)
}).reset_index(drop=True)

print("=== DISTRIBUCIÓN DE PRECIOS ===")
print(tabla_precios)
```

---

## 3.3 Medidas Descriptivas Clave

### Con Pandas

```python
# DataFrame de ventas completo
np.random.seed(42)
datos = pd.DataFrame({
    'ventas': np.random.normal(10000, 2000, 200),
    'costos': np.random.normal(6000, 1000, 200),
    'clientes': np.random.poisson(50, 200),
    'satisfaccion': np.random.uniform(3, 5, 200)
}).round(2)

print("=== DESCRIBE GENERAL ===")
print(datos.describe())

print("\n=== MEDIDAS ESPECÍFICAS ===")
for col in datos.columns:
    print(f"\n--- {col} ---")
    print(f"  Media:      {datos[col].mean():>10.2f}")
    print(f"  Mediana:    {datos[col].median():>10.2f}")
    print(f"  Desv. Est.: {datos[col].std():>10.2f}")
    print(f"  Mínimo:     {datos[col].min():>10.2f}")
    print(f"  Máximo:     {datos[col].max():>10.2f}")
    print(f"  Rango:      {datos[col].max() - datos[col].min():>10.2f}")
    print(f"  Asimetría:  {datos[col].skew():>10.3f}")
    print(f"  Curtosis:   {datos[col].kurtosis():>10.3f}")
```

---

## 3.4 Asimetría y Curtosis

### Asimetría (Skewness)

Mide la simetría de la distribución:
- **Asimetría > 0**: Cola hacia la derecha (mayoría de datos a la izquierda)
- **Asimetría = 0**: Distribución simétrica
- **Asimetría < 0**: Cola hacia la izquierda (mayoría de datos a la derecha)

$$S = \frac{1}{n} \sum_{i=1}^{n} \left(\frac{x_i - \bar{x}}{s}\right)^3$$

### Curtosis (Kurtosis)

Mide el "apuntamiento" de la distribución:
- **Curtosis > 0**: Leptocúrtica (más puntiaguda que normal)
- **Curtosis = 0**: Mesocúrtica (normal)
- **Curtosis < 0**: Platicúrtica (más aplanada que normal)

$$K = \frac{1}{n} \sum_{i=1}^{n} \left(\frac{x_i - \bar{x}}{s}\right)^4 - 3$$

```python
# Visualicemos diferentes formas de distribución
from scipy import stats

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Distribución con asimetría positiva
data_pos = np.random.exponential(2, 1000)
axes[0].hist(data_pos, bins=30, alpha=0.7, color='steelblue', edgecolor='white')
axes[0].set_title(f'Asimetría Positiva (S = {stats.skew(data_pos):.2f})')

# Distribución simétrica
data_sym = np.random.normal(0, 1, 1000)
axes[1].hist(data_sym, bins=30, alpha=0.7, color='forestgreen', edgecolor='white')
axes[1].set_title(f'Simétrica (S = {stats.skew(data_sym):.2f})')

# Distribución con asimetría negativa
data_neg = -np.random.exponential(2, 1000) + 5
axes[2].hist(data_neg, bins=30, alpha=0.7, color='crimson', edgecolor='white')
axes[2].set_title(f'Asimetría Negativa (S = {stats.skew(data_neg):.2f})')

plt.tight_layout()
plt.savefig('asimetria_ejemplos.png', dpi=150)
plt.show()
```

---

## 3.5 Cuantiles y Percentiles

Los **cuantiles** dividen los datos en partes iguales:

| Medida | División | Uso en Negocios |
|--------|----------|-----------------|
| **Cuartiles (Q1, Q2, Q3)** | 4 partes | Clasificación ABC de inventarios |
| **Deciles (D1-D9)** | 10 partes | Segmentación de clientes por gasto |
| **Percentiles (P1-P99)** | 100 partes | Evaluación de desempeño |

```python
# Cálculo de percentiles en datos de ventas
ventas_mensuales = np.random.gamma(shape=5, scale=2000, size=100).round(2)

percentiles = [10, 25, 50, 75, 90, 95, 99]
valores_percentiles = np.percentile(ventas_mensuales, percentiles)

print("=== PERCENTILES DE VENTAS MENSUALES ===")
for p, v in zip(percentiles, valores_percentiles):
    print(f"P{p:2d}: ${v:>10,.2f}")

# Interpretación de negocio
print(f"\nEl 50% de los meses venden menos de ${np.median(ventas_mensuales):,.2f}")
print(f"El 10% de los meses mejores venden más de ${np.percentile(ventas_mensuales, 90):,.2f}")
print(f"Solo el 1% de los meses superan los ${np.percentile(ventas_mensuales, 99):,.2f}")
```

---

## 3.6 Valores Atípicos (Outliers)

### Método del Rango Intercuartílico (IQR)

$$IQR = Q3 - Q1$$
$$Límite\ Inferior = Q1 - 1.5 \times IQR$$
$$Límite\ Superior = Q3 + 1.5 \times IQR$$

```python
def detectar_outliers(serie, metodo='iqr'):
    """Detecta outliers usando el método IQR"""
    Q1 = serie.quantile(0.25)
    Q3 = serie.quantile(0.75)
    IQR = Q3 - Q1
    
    lim_inf = Q1 - 1.5 * IQR
    lim_sup = Q3 + 1.5 * IQR
    
    outliers = serie[(serie < lim_inf) | (serie > lim_sup)]
    
    return {
        'Q1': Q1, 'Q3': Q3, 'IQR': IQR,
        'lim_inf': lim_inf, 'lim_sup': lim_sup,
        'n_outliers': len(outliers),
        'pct_outliers': len(outliers) / len(serie) * 100,
        'outliers': outliers
    }

# Simulamos ventas con algunos valores extremos
ventas = np.random.normal(100, 20, 200)
ventas = np.append(ventas, [250, 280, 30, 25, 300])  # Outliers intencionales

resultado = detectar_outliers(pd.Series(ventas))

print("=== DETECCIÓN DE OUTLIERS (IQR) ===")
print(f"Q1: {resultado['Q1']:.2f}")
print(f"Q3: {resultado['Q3']:.2f}")
print(f"IQR: {resultado['IQR']:.2f}")
print(f"Límite inferior: {resultado['lim_inf']:.2f}")
print(f"Límite superior: {resultado['lim_sup']:.2f}")
print(f"Outliers detectados: {resultado['n_outliers']} ({resultado['pct_outliers']:.1f}%)")
print(f"Valores atípicos: {sorted(resultado['outliers'].values)}")
```

### Método Z-Score

$$z_i = \frac{x_i - \mu}{\sigma}$$

Valores con $|z| > 3$ son potenciales outliers.

```python
def detectar_outliers_zscore(serie, umbral=3):
    """Detecta outliers usando Z-Score"""
    z_scores = np.abs(stats.zscore(serie))
    outliers = serie[z_scores > umbral]
    
    return {
        'umbral': umbral,
        'n_outliers': len(outliers),
        'pct_outliers': len(outliers) / len(serie) * 100,
        'outliers': outliers.values
    }

resultado_z = detectar_outliers_zscore(ventas)
print(f"\n=== OUTLIERS POR Z-SCORE (|z| > {resultado_z['umbral']}) ===")
print(f"Outliers: {resultado_z['n_outliers']} ({resultado_z['pct_outliers']:.1f}%)")
```

---

## 3.7 Tablas de Contingencia

Cruzan dos variables categóricas para mostrar frecuencias conjuntas.

```python
# Datos de canal de venta y categoría
datos_tabla = pd.DataFrame({
    'canal': np.random.choice(['Online', 'Tienda', 'Catálogo'], 1000),
    'categoria': np.random.choice(['Electrónica', 'Ropa', 'Hogar'], 1000),
    'monto': np.random.exponential(200, 1000)
})

# Tabla de contingencia
tabla_contingencia = pd.crosstab(datos_tabla['canal'], datos_tabla['categoria'])
print("=== TABLA DE CONTINGENCIA: CANAL vs CATEGORÍA ===")
print(tabla_contingencia)

# Tabla con porcentajes por fila
print("\n=== PORCENTAJES POR FILA ===")
print(pd.crosstab(datos_tabla['canal'], datos_tabla['categoria'], normalize='index') * 100)

# Tabla con totales
tabla_totales = pd.crosstab(datos_tabla['canal'], datos_tabla['categoria'], margins=True, margins_name='Total')
print("\n=== CON TOTALES ===")
print(tabla_totales)
```

---

## 3.8 Ejemplo: Análisis Descriptivo de Ventas

```python
# Dataset completo de ventas
np.random.seed(42)
n = 1000

ventas_df = pd.DataFrame({
    'fecha': pd.date_range('2024-01-01', periods=n, freq='D'),
    'producto': np.random.choice(['A', 'B', 'C', 'D', 'E'], n),
    'categoria': np.random.choice(['Premium', 'Estándar', 'Económico'], n, p=[0.2, 0.5, 0.3]),
    'region': np.random.choice(['Norte', 'Sur', 'Centro', 'Occidente'], n),
    'unidades': np.random.poisson(20, n),
    'precio': np.where(
        np.random.random(n) < 0.02,
        np.random.uniform(500, 1000, n),  # Outliers
        np.random.gamma(2, 50, n).clip(10, 400)
    ),
    'descuento': np.random.choice([0, 0.05, 0.10, 0.15, 0.20], n, p=[0.5, 0.2, 0.15, 0.1, 0.05])
})

ventas_df['ingreso'] = ventas_df['unidades'] * ventas_df['precio'] * (1 - ventas_df['descuento'])
ventas_df['ingreso_sin_descuento'] = ventas_df['unidades'] * ventas_df['precio']

print("=== ANÁLISIS DESCRIPTIVO COMPLETO ===")
print("Dimensiones:", ventas_df.shape)
print("\n=== TIPOS DE DATOS ===")
print(ventas_df.dtypes)

print("\n=== VALORES NULOS ===")
print(ventas_df.isnull().sum())

print("\n=== RESUMEN POR CATEGORÍA ===")
print(ventas_df.groupby('categoria')['ingreso'].describe())

print("\n=== VENTAS POR REGIÓN ===")
print(ventas_df.groupby('region').agg({
    'ingreso': ['sum', 'mean', 'std'],
    'unidades': 'sum'
}))

print("\n=== IMPACTO DEL DESCUENTO ===")
print(ventas_df.groupby('descuento').agg({
    'ingreso': 'mean',
    'unidades': 'mean'
}))
```

---

## 3.9 Ejemplo: Análisis Descriptivo de Inventarios (ABC)

```python
# Clasificación ABC basada en valor de inventario
np.random.seed(42)
n_skus = 200

inventario = pd.DataFrame({
    'sku': [f'SKU_{i:04d}' for i in range(n_skus)],
    'costo_unitario': np.random.uniform(10, 500, n_skus).round(2),
    'stock_actual': np.random.randint(0, 200, n_skus),
    'demanda_anual': np.random.randint(10, 5000, n_skus)
})

inventario['valor_inventario'] = inventario['costo_unitario'] * inventario['stock_actual']
inventario['valor_demanda_anual'] = inventario['costo_unitario'] * inventario['demanda_anual']

# Ordenamos por valor de demanda anual descendente
inventario = inventario.sort_values('valor_demanda_anual', ascending=False).reset_index(drop=True)

# Porcentaje acumulado
inventario['pct_acumulado'] = (inventario['valor_demanda_anual'].cumsum() / 
                                 inventario['valor_demanda_anual'].sum() * 100)

# Clasificación ABC
condiciones = [
    inventario['pct_acumulado'] <= 80,
    (inventario['pct_acumulado'] > 80) & (inventario['pct_acumulado'] <= 95),
    inventario['pct_acumulado'] > 95
]
clases = ['A', 'B', 'C']
inventario['clase_abc'] = np.select(condiciones, clases)

print("=== CLASIFICACIÓN ABC ===")
print(inventario.groupby('clase_abc').agg({
    'sku': 'count',
    'valor_inventario': 'sum',
    'stock_actual': 'mean'
}).round(2))

print("\n=== ESTADÍSTICAS DESCRIPTIVAS POR CLASE ===")
print(inventario.groupby('clase_abc')['costo_unitario'].describe())
```

---

## 3.10 Resumen

- La estadística descriptiva resume datos mediante tablas, gráficos y medidas numéricas
- Las tablas de frecuencia organizan datos categóricos y numéricos
- Los cuantiles (percentiles) permiten entender la distribución
- Los outliers se detectan mediante IQR o Z-Score
- Las tablas de contingencia revelan relaciones entre variables categóricas
- Pandas proporciona herramientas potentes para descriptiva

---

## Ejercicios Propuestos

1. **Ventas**: Calcula los percentiles 10, 25, 50, 75, 90 de ingresos por región
2. **Compras**: Crea una tabla de contingencia entre proveedor y calidad del producto
3. **Inventarios**: Aplica clasificación ABC a un conjunto de 50 SKUs simulados

---

[← Anterior](02-intro-probabilidad.md) | [Índice](index.md) | [Siguiente →](04-medidas-tendencia-central.md)
