# Capítulo 5: Medidas de Dispersión

[← Anterior](04-medidas-tendencia-central.md) | [Índice](index.md) | [Siguiente →](06-visualizacion-datos.md)

---

## 5.1 ¿Qué son las Medidas de Dispersión?

Mientras que las medidas de tendencia central indican el **centro** de los datos, las medidas de dispersión indican **cuán dispersos** están los datos alrededor de ese centro. Una baja dispersión significa datos homogéneos; alta dispersión significa datos heterogéneos.

### Importancia en Negocios

| Baja Dispersión | Alta Dispersión |
|-----------------|-----------------|
| Demanda predecible → Menos inventario de seguridad | Demanda volátil → Más stock de seguridad |
| Proveedores confiables → Lead time constante | Proveedores variables → Holgura en compras |
| Ventas estables → Pronósticos precisos | Ventas erráticas → Pronósticos inciertos |

---

## 5.2 Rango

### Definición

$$Rango = X_{máx} - X_{mín}$$

```python
import numpy as np
import pandas as pd
from scipy import stats

# Ventas diarias de dos productos
producto_A = [95, 97, 100, 102, 98, 101, 99, 103, 96, 100]  # Estable
producto_B = [50, 150, 80, 200, 30, 180, 60, 190, 40, 170]  # Volátil

rango_A = max(producto_A) - min(producto_A)
rango_B = max(producto_B) - min(producto_B)

print(f"Producto A - Rango: {rango_A} (ventas estables)")
print(f"Producto B - Rango: {rango_B} (ventas volátiles)")
print(f"El producto B tiene {rango_B/rango_A:.0f}x más variabilidad en el rango")
```

### Limitaciones del Rango
- Solo usa 2 valores (máximo y mínimo)
- Muy sensible a outliers
- No refleja la variabilidad interna

---

## 5.3 Rango Intercuartílico (IQR)

### Definición

$$IQR = Q_3 - Q_1$$

El IQR contiene el 50% central de los datos, siendo robusto ante outliers.

```python
# Comparación: rango vs IQR
datos_con_outlier = [10, 12, 11, 13, 12, 14, 11, 12, 13, 100]

Q1 = np.percentile(datos_con_outlier, 25)
Q3 = np.percentile(datos_con_outlier, 75)
IQR = Q3 - Q1
rango_total = max(datos_con_outlier) - min(datos_con_outlier)

print(f"Rango total: {rango_total}")
print(f"Q1: {Q1}")
print(f"Q3: {Q3}")
print(f"IQR: {IQR}")
print(f"El IQR ignora el outlier (100) y refleja la variabilidad real")

# Cinco números de Tukey
minimo = min(datos_con_outlier)
maximo_sin_outlier = Q3 + 1.5 * IQR

print(f"\nResumen de 5 números:")
print(f"  Mínimo: {minimo}")
print(f"  Q1: {Q1}")
print(f"  Mediana: {np.median(datos_con_outlier)}")
print(f"  Q3: {Q3}")
print(f"  Máximo (no-outlier): {maximo_sin_outlier:.1f}")
```

---

## 5.4 Varianza y Desviación Estándar

### Varianza

Mide la **dispersión promedio al cuadrado** alrededor de la media:

$$\sigma^2 = \frac{\sum_{i=1}^{N} (x_i - \mu)^2}{N} \quad \text{(Población)}$$

$$s^2 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n-1} \quad \text{(Muestra)}$$

### Desviación Estándar

Raíz cuadrada de la varianza, en las mismas unidades que los datos:

$$\sigma = \sqrt{\sigma^2} \quad \text{o} \quad s = \sqrt{s^2}$$

```python
def calcular_varianza_manual(datos, poblacion=False):
    """Cálculo manual de varianza"""
    media = np.mean(datos)
    diferencias = [(x - media) ** 2 for x in datos]
    if poblacion:
        return sum(diferencias) / len(datos)
    else:
        return sum(diferencias) / (len(datos) - 1)

# Datos de ventas semanales
ventas_semana = [12000, 12500, 11800, 12200, 12800, 11500, 12100, 12400]

varianza_muestral = np.var(ventas_semana, ddof=1)
varianza_poblacional = np.var(ventas_semana, ddof=0)
desv_estandar = np.std(ventas_semana, ddof=1)

print(f"Varianza muestral: {varianza_muestral:,.0f} $²")
print(f"Varianza poblacional: {varianza_poblacional:,.0f} $²")
print(f"Desviación estándar: {desv_estandar:,.0f} $")
print(f"Interpretación: Las ventas semanales varían típicamente ±${desv_estandar:,.0f}")
```

### Interpretación Empírica (Regla 68-95-99.7)

Para distribuciones aproximadamente normales:

- **68%** de los datos: $\bar{x} \pm 1s$
- **95%** de los datos: $\bar{x} \pm 2s$
- **99.7%** de los datos: $\bar{x} \pm 3s$

```python
# Validación con datos reales
np.random.seed(42)
ventas = np.random.normal(1000, 200, 10000)

media = np.mean(ventas)
std = np.std(ventas)

dentro_1s = np.mean(np.abs(ventas - media) <= 1 * std) * 100
dentro_2s = np.mean(np.abs(ventas - media) <= 2 * std) * 100
dentro_3s = np.mean(np.abs(ventas - media) <= 3 * std) * 100

print("=== REGLA EMPÍRICA ===")
print(f"Datos dentro de ±1σ: {dentro_1s:.1f}% (esperado: 68%)")
print(f"Datos dentro de ±2σ: {dentro_2s:.1f}% (esperado: 95%)")
print(f"Datos dentro de ±3σ: {dentro_3s:.1f}% (esperado: 99.7%)")

# Aplicación en inventarios
print(f"\nSi la demanda media es {media:.0f} con σ = {std:.0f}:")
print(f"  → 68% del tiempo la demanda está entre {media-std:.0f} y {media+std:.0f}")
print(f"  → 95% del tiempo está entre {media-2*std:.0f} y {media+2*std:.0f}")
print(f"  → Stock de seguridad para 95%: {1.96*std:.0f} unidades")
```

---

## 5.5 Coeficiente de Variación (CV)

Permite comparar la variabilidad entre conjuntos con diferentes medias:

$$CV = \frac{s}{\bar{x}}$$

```python
# Comparando variabilidad de diferentes productos
productos = {
    'Laptop': {'media': 15000, 'std': 3000},
    'Mouse': {'media': 200, 'std': 80},
    'Monitor': {'media': 5000, 'std': 1500},
    'Cable': {'media': 50, 'std': 30}
}

print("=== COEFICIENTE DE VARIACIÓN ===")
print(f"{'Producto':10s} {'Media':>8s} {'Std':>8s} {'CV':>8s}")
print("-" * 35)

for prod, vals in productos.items():
    cv = vals['std'] / vals['media']
    print(f"{prod:10s} {vals['media']:>8.0f} {vals['std']:>8.0f} {cv:>7.1%}")

print(f"\nAunque la laptop tiene mayor std, el cable tiene")
print(f"mayor variabilidad relativa (CV más alto)")
```

---

## 5.6 Desviación Media Absoluta (MAD)

Alternativa robusta a la desviación estándar:

$$MAD = \frac{1}{n} \sum_{i=1}^{n} |x_i - \bar{x}|$$

```python
def mad(datos):
    """Desviación media absoluta"""
    media = np.mean(datos)
    return np.mean(np.abs(datos - media))

# Comparación STD vs MAD con y sin outliers
datos_limpios = [100, 102, 98, 105, 101, 99, 103, 97, 104, 100]
datos_sucios = datos_limpios + [500]

print("=== STD vs MAD (robustez) ===")
print(f"Datos limpios - STD: {np.std(datos_limpios, ddof=1):.2f}, MAD: {mad(datos_limpios):.2f}")
print(f"Datos con outlier - STD: {np.std(datos_sucios, ddof=1):.2f}, MAD: {mad(datos_sucios):.2f}")
print(f"La MAD es menos afectada por outliers extremos")
```

---

## 5.7 Ejemplo: Ventas - Análisis de Variabilidad Comercial

```python
# Análisis de dispersión para fuerza de ventas
np.random.seed(42)

vendedores = pd.DataFrame({
    'vendedor': ['Ana', 'Carlos', 'María', 'Pedro', 'Laura'],
    'ventas_anuales': [
        np.random.normal(600000, 30000, 12).sum(),  # Ana: consistente
        np.random.normal(600000, 80000, 12).sum(),  # Carlos: volátil
        np.random.normal(550000, 20000, 12).sum(),  # María: baja pero estable
        np.random.normal(650000, 70000, 12).sum(),  # Pedro: alta pero volátil
        np.random.normal(580000, 25000, 12).sum(),  # Laura: estable media
    ],
    'std_mensual': [
        30000, 80000, 20000, 70000, 25000
    ],
    'media_mensual': [
        50000, 50000, 45833, 54167, 48333
    ]
})

vendedores['cv'] = vendedores['std_mensual'] / vendedores['media_mensual']

print("=== ANÁLISIS DE VARIABILIDAD COMERCIAL ===")
print(vendedores.to_string(index=False))

print(f"\n--- CONCLUSIONES ---")
print(f"Ana: Alto volumen, baja variabilidad → Vendedora estrella")
print(f"Carlos: Buen volumen pero errático → Necesita mentoring")
print(f"María: Bajo volumen pero muy estable → Cumple presupuesto")
print(f"Pedro: Mayor volumen pero volátil → Alto potencial, requiere soporte")
print(f"Laura: Rendimiento sólido y estable → Confiable")
```

---

## 5.8 Ejemplo: Compras - Evaluación de Proveedores

```python
# Evaluación de proveedores basada en consistencia de entrega
proveedores = pd.DataFrame({
    'proveedor': ['Proveedor A', 'Proveedor B', 'Proveedor C'],
    'dias_entrega_media': [5, 7, 6],
    'dias_entrega_std': [0.5, 3.0, 1.5],
    'precio_unitario': [100, 95, 98],
    'tasa_defectos': [0.01, 0.05, 0.02]
})

proveedores['cv_entrega'] = proveedores['dias_entrega_std'] / proveedores['dias_entrega_media']

print("=== EVALUACIÓN DE PROVEEDORES ===")
print(proveedores.to_string(index=False))

# Cálculo de stock de seguridad necesario
for _, prov in proveedores.iterrows():
    lead_time_prom = prov['dias_entrega_media']
    lead_time_std = prov['dias_entrega_std']
    demanda_diaria = 100  # unidades
    z = 1.65  # 95% nivel de servicio
    
    ss = z * demanda_diaria * lead_time_std  # Simplificado
    print(f"\n{prov['proveedor']}:")
    print(f"  Lead time: {lead_time_prom}±{lead_time_std} días (CV: {prov['cv_entrega']:.1%})")
    print(f"  Stock de seguridad necesario: {ss:.0f} unidades")
    print(f"  Costo de mantener SS: ${ss * 5:.0f}/año (costo unitario: $5)")
```

---

## 5.9 Ejemplo: Inventarios - Variabilidad de Demanda

```python
# Análisis de variabilidad para políticas de inventario
np.random.seed(42)

inventario = pd.DataFrame({
    'sku': [f'SKU_{i}' for i in range(20)],
    'demanda_media': np.random.uniform(10, 500, 20).round(),
    'demanda_std': np.random.uniform(2, 200, 20).round(),
    'costo_unitario': np.random.uniform(10, 200, 20).round(2),
    'lead_time': np.random.choice([2, 3, 5, 7], 20)
})

inventario['cv_demanda'] = inventario['demanda_std'] / inventario['demanda_media']
inventario['clasificacion'] = pd.cut(
    inventario['cv_demanda'],
    bins=[0, 0.3, 0.8, float('inf')],
    labels=['Baja var.', 'Media var.', 'Alta var.']
)

# Stock de seguridad (método simplificado)
Z = 1.65  # 95%服务水平
inventario['stock_seguridad'] = (Z * inventario['demanda_std'] * 
                                   np.sqrt(inventario['lead_time'])).round()
inventario['costo_ss'] = (inventario['stock_seguridad'] * inventario['costo_unitario']).round(2)

print("=== ANÁLISIS DE VARIABILIDAD DE INVENTARIO ===")
print(inventario.groupby('clasificacion').agg({
    'sku': 'count',
    'stock_seguridad': 'mean',
    'costo_ss': 'sum',
    'cv_demanda': 'mean'
}).round(2))

print("\n=== PRODUCTOS CON MAYOR VARIABILIDAD ===")
print(inventario.nlargest(5, 'cv_demanda')[['sku', 'demanda_media', 'demanda_std', 'cv_demanda', 'stock_seguridad']])

# Costo total de stock de seguridad
print(f"\nCosto total de stock de seguridad: ${inventario['costo_ss'].sum():,.2f}")
print(f"Costo evitable si se reduce variabilidad: significativo")
```

---

## 5.10 Desigualdad de Chebyshev

Para **cualquier** distribución (no solo normal):

$$P(|X - \mu| \geq k\sigma) \leq \frac{1}{k^2}$$

```python
def chebyshev(k):
    """Proporción máxima de datos fuera de k desviaciones"""
    return min(1, 1 / k**2)

print("=== DESIGUALDAD DE CHEBYSHEV ===")
for k in [1.5, 2, 2.5, 3]:
    print(f"P(|X-μ| ≥ {k}σ) ≤ {1/k**2:.4f} → Al menos {(1-1/k**2)*100:.1f}% dentro")

# Aplicación: límites de inventario
media_demanda = 100
std_demanda = 30

for k in [2, 3]:
    lim_sup = media_demanda + k * std_demanda
    print(f"\nCon Chebyshev, k={k}:")
    print(f"  Al menos {(1-1/k**2)*100:.1f}% de la demanda ≤ {lim_sup:.0f}")
    print(f"  Stock máximo recomendado: {lim_sup:.0f} unidades")
```

---

## 5.11 Medidas de Dispersión en Pandas

```python
# Resumen completo de dispersión con pandas
df = pd.DataFrame({
    'ventas': np.random.normal(10000, 2000, 100),
    'costos': np.random.normal(6000, 1000, 100),
    'margen': np.random.normal(4000, 800, 100)
})

print("=== MEDIDAS DE DISPERSIÓN COMPLETAS ===")
medidas = pd.DataFrame({
    'Ventas': [
        df['ventas'].std(),
        df['ventas'].var(),
        df['ventas'].mad(),
        df['ventas'].max() - df['ventas'].min(),
        df['ventas'].quantile(0.75) - df['ventas'].quantile(0.25),
        df['ventas'].std() / df['ventas'].mean(),
        stats.skew(df['ventas']),
        stats.kurtosis(df['ventas'])
    ],
    'Costos': [
        df['costos'].std(),
        df['costos'].var(),
        df['costos'].mad(),
        df['costos'].max() - df['costos'].min(),
        df['costos'].quantile(0.75) - df['costos'].quantile(0.25),
        df['costos'].std() / df['costos'].mean(),
        stats.skew(df['costos']),
        stats.kurtosis(df['costos'])
    ],
    'Margen': [
        df['margen'].std(),
        df['margen'].var(),
        df['margen'].mad(),
        df['margen'].max() - df['margen'].min(),
        df['margen'].quantile(0.75) - df['margen'].quantile(0.25),
        df['margen'].std() / df['margen'].mean(),
        stats.skew(df['margen']),
        stats.kurtosis(df['margen'])
    ]
}, index=['Desv. Estándar', 'Varianza', 'MAD', 'Rango', 'IQR', 'CV', 'Asimetría', 'Curtosis'])

print(medidas.round(4))
```

---

## 5.12 Resumen

| Medida | Unidades | Robusta | Uso Principal |
|--------|----------|---------|---------------|
| Rango | Originales | No | Visión rápida de extremos |
| IQR | Originales | **Sí** | Variabilidad central robusta |
| Varianza | Al cuadrado | No | Cálculos matemáticos |
| Desv. Estándar | Originales | No | Medida estándar de dispersión |
| MAD | Originales | **Sí** | Alternativa robusta |
| CV | Adimensional | - | Comparar heterogeneidad |
| Rango | Originales | No | Dispersión total |

---

## Ejercicios Propuestos

1. **Ventas**: Calcula el CV de 3 productos diferentes. ¿Cuál tiene la demanda más predecible?
2. **Compras**: Evalúa 2 proveedores: uno con lead time 5±0.5 días y otro 6±3 días. ¿Cuál elegirías?
3. **Inventarios**: Usando la regla empírica, determina el stock de seguridad para cubrir el 95% de la demanda

---

[← Anterior](04-medidas-tendencia-central.md) | [Índice](index.md) | [Siguiente →](06-visualizacion-datos.md)
