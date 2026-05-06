# Capítulo 4: Medidas de Tendencia Central

[← Anterior](03-estadistica-descriptiva.md) | [Índice](index.md) | [Siguiente →](05-medidas-dispersion.md)

---

## 4.1 ¿Qué son las Medidas de Tendencia Central?

Las medidas de tendencia central son valores que **representan el centro** de un conjunto de datos. Responden a la pregunta: "¿Cuál es el valor típico o representativo de estos datos?"

### Las Tres Medidas Principales

| Medida | Definición | Cuándo Usarla |
|--------|-----------|---------------|
| **Media** | Suma de valores ÷ número de observaciones | Datos simétricos, sin outliers |
| **Mediana** | Valor central cuando los datos se ordenan | Datos asimétricos o con outliers |
| **Moda** | Valor que más se repite | Datos categóricos o discretos |

---

## 4.2 Media Aritmética

### Definición

$$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$

### Implementación

```python
import numpy as np
import pandas as pd
from scipy import stats

# Datos de ventas diarias
ventas_diarias = [120, 135, 98, 145, 160, 112, 128, 95, 142, 155,
                  118, 130, 145, 108, 165, 122, 138, 150, 105, 140]

media = np.mean(ventas_diarias)
print(f"Media de ventas diarias: {media:.2f} unidades")
print(f"Interpretación: En promedio se venden {media:.0f} unidades por día")
```

### Media Ponderada

Útil cuando ciertas observaciones tienen más peso:

$$\bar{x}_p = \frac{\sum_{i=1}^{n} w_i x_i}{\sum_{i=1}^{n} w_i}$$

```python
# Precio promedio ponderado por volumen de compra
precios = np.array([100, 150, 200, 120])
unidades = np.array([500, 200, 100, 400])

precio_prom_ponderado = np.average(precios, weights=unidades)
precio_simple = np.mean(precios)

print(f"Precio promedio simple: ${precio_simple:.2f}")
print(f"Precio promedio ponderado: ${precio_prom_ponderado:.2f}")
print(f"El ponderado refleja mejor el costo real porque considera el volumen")
```

### Media Truncada (Trimmed Mean)

Elimina un porcentaje de valores extremos:

```python
# Datos con outliers
ventas_con_outliers = ventas_diarias + [500, 20, 600]

media_completa = np.mean(ventas_con_outliers)
media_truncada = stats.trim_mean(ventas_con_outliers, 0.1)  # 10% de cada extremo
mediana = np.median(ventas_con_outliers)

print(f"Media completa: {media_completa:.2f}")
print(f"Media truncada (10%): {media_truncada:.2f}")
print(f"Mediana: {mediana:.2f}")
print(f"La media truncada es robusta ante outliers extremos")
```

---

## 4.3 Mediana

### Definición

- Si $n$ es impar: $Mediana = x_{(n+1)/2}$
- Si $n$ es par: $Mediana = \frac{x_{n/2} + x_{(n/2)+1}}{2}$

```python
def calcular_mediana(datos):
    """Cálculo manual de la mediana"""
    datos_ord = sorted(datos)
    n = len(datos_ord)
    
    if n % 2 == 1:
        return datos_ord[n // 2]
    else:
        mid = n // 2
        return (datos_ord[mid - 1] + datos_ord[mid]) / 2

# Comparación media vs mediana
sueldos = [25000, 28000, 30000, 32000, 35000, 38000, 45000, 50000, 200000]

media_sueldos = np.mean(sueldos)
mediana_sueldos = np.median(sueldos)

print(f"Media salarial: ${media_sueldos:,.2f}")
print(f"Mediana salarial: ${mediana_sueldos:,.2f}")
print(f"La mediana es más representativa cuando hay valores extremos")
print(f"El salario típico (mediana) es ${mediana_sueldos:,.0f}")
```

### Cuándo usar la Mediana en Negocios

```python
# Precios de productos: unos muy caros distorsionan la media
precios_productos = [29, 39, 49, 59, 79, 99, 149, 199, 499, 999, 2499]

media_precios = np.mean(precios_productos)
mediana_precios = np.median(precios_productos)

print(f"Precio promedio: ${media_precios:.2f}")
print(f"Precio mediano: ${mediana_precios:.2f}")
print(f"El precio mediano representa mejor al 'producto típico'")
print(f"El {sum(1 for p in precios_productos if p < media_precios)} de {len(precios_productos)} productos")
print(f"están por debajo del precio promedio")
```

---

## 4.4 Moda

### Definición

Valor(es) que aparecen con mayor frecuencia en el conjunto de datos.

```python
from scipy import stats

# Moda en datos de categorías de venta
categorias = ['A', 'B', 'A', 'C', 'A', 'B', 'D', 'A', 'C', 'A', 'B', 'A', 'C', 'A', 'E']
moda_categorias = stats.mode(categorias)
print(f"Categoría modal: {moda_categorias.mode[0]} (frecuencia: {moda_categorias.count[0]})")

# Moda en datos numéricos
ventas_diarias = [100, 100, 120, 120, 120, 120, 130, 130, 140, 150, 150, 150]
moda_ventas = stats.mode(ventas_diarias)
print(f"Ventas modales: {moda_ventas.mode[0]} unidades ({moda_ventas.count[0]} ocurrencias)")

# Múltiples modas
datos_multimodales = [1, 1, 2, 2, 3, 3, 4, 5]
modas = stats.multimode(datos_multimodales)
print(f"Múltiples modas: {modas}")

# Distribución de frecuencias para encontrar la moda visualmente
```

---

## 4.5 Relación entre Media, Mediana y Moda

La forma de la distribución determina la relación:

```
Simétrica (Normal):        Media = Mediana = Moda
Asimetría Positiva:        Moda < Mediana < Media
Asimetría Negativa:        Media < Mediana < Moda
```

```python
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

# Generamos datos con asimetría positiva (ventas típicas)
ventas = np.random.exponential(scale=50, size=500) + 20

media = np.mean(ventas)
mediana = np.median(ventas)
moda_val = ventas[np.argmax(np.bincount(ventas.astype(int).clip(0, 300)))]

print(f"Media:  {media:.2f}")
print(f"Mediana: {mediana:.2f}")
print(f"Moda:   {moda_val:.2f}")
print(f"Relación: Moda ({moda_val:.1f}) < Mediana ({mediana:.1f}) < Media ({media:.1f})")
print("Esto indica una distribución con asimetría positiva (cola derecha)")
```

---

## 4.6 Media Geométrica

Útil para tasas de crecimiento, rendimientos, y datos multiplicativos:

$$G = \left(\prod_{i=1}^{n} x_i\right)^{1/n}$$

```python
from scipy import stats

# Crecimiento de ventas mes a mes (factores)
crecimiento = [1.10, 1.15, 1.08, 1.12, 1.05]  # 10%, 15%, 8%, 12%, 5%

media_geom = stats.gmean(crecimiento)
media_arit = np.mean(crecimiento)

print(f"Media aritmética del crecimiento: {(media_arit - 1) * 100:.2f}%")
print(f"Media geométrica del crecimiento: {(media_geom - 1) * 100:.2f}%")
print(f"Crecimiento real acumulado: {(np.prod(crecimiento) - 1) * 100:.2f}%")
print(f"Verificación: {(media_geom ** len(crecimiento) - 1) * 100:.2f}%")
```

### Ejemplo de Negocio

```python
# Ventas trimestrales con crecimiento
ventas_trim = [100, 115, 125, 140, 158, 175]
cambios = [ventas_trim[i] / ventas_trim[i-1] for i in range(1, len(ventas_trim))]

crecimiento_medio = stats.gmean(cambios)
print(f"Crecimiento trimestral medio: {(crecimiento_medio-1)*100:.2f}%")
print(f"Ventas iniciales: ${ventas_trim[0]:.0f}")
print(f"Ventas proyectadas con media geométrica: ${ventas_trim[0] * (crecimiento_medio**len(cambios)):.0f}")
print(f"Ventas reales finales: ${ventas_trim[-1]:.0f}")
```

---

## 4.7 Media Armónica

Útil para tasas y ratios (ej: velocidad, precio por unidad):

$$H = \frac{n}{\sum_{i=1}^{n} \frac{1}{x_i}}$$

```python
from scipy import stats

# Precio por unidad en diferentes lotes de compra
precio_unitario = [10.50, 10.25, 10.75, 10.00, 10.30]

media_arm = stats.hmean(precio_unitario)
media_arit = np.mean(precio_unitario)

print(f"Precio promedio (aritmética): ${media_arit:.4f}")
print(f"Precio promedio (armónica): ${media_arm:.4f}")
print(f"La media armónica es más apropiada cuando se promedian ratios")
```

---

## 4.8 Ejemplo: Ventas - KPI Comerciales

```python
# Análisis completo de tendencia central para reporting
np.random.seed(42)

# Ventas mensuales por vendedor
vendedores = {
    'Ana': np.random.normal(50000, 5000, 12),
    'Carlos': np.random.normal(45000, 8000, 12),
    'María': np.random.normal(55000, 3000, 12),
    'Pedro': np.random.normal(40000, 10000, 12),
    'Laura': np.random.normal(48000, 6000, 12)
}

print("=== REPORTE DE VENTAS - TENDENCIA CENTRAL ===")
print(f"{'Vendedor':10s} {'Media':>10s} {'Mediana':>10s} {'Moda':>10s} {'Min':>10s} {'Max':>10s}")
print("-" * 55)

for nombre, ventas in vendedores.items():
    moda_result = stats.mode(ventas.round(-2))
    print(f"{nombre:10s} {np.mean(ventas):>10.0f} {np.median(ventas):>10.0f} "
          f"{moda_result.mode[0]:>10.0f} {ventas.min():>10.0f} {ventas.max():>10.0f}")

# Vendedor del mes
print(f"\nVendedor con mayor media: {max(vendedores, key=lambda k: np.mean(vendedores[k]))}")
print(f"Vendedor más consistente (menor std): {min(vendedores, key=lambda k: np.std(vendedores[k]))}")
```

---

## 4.9 Ejemplo: Compras - Precio Promedio de Insumos

```python
# Diferentes métodos para calcular precio promedio de compras
compras = pd.DataFrame({
    'proveedor': ['Prov_A'] * 5 + ['Prov_B'] * 5,
    'producto': ['X'] * 10,
    'cantidad': [100, 200, 150, 300, 250, 50, 100, 80, 120, 60],
    'precio_unitario': [10, 9.5, 10.2, 8.9, 9.8, 11, 10.8, 11.5, 10.9, 12]
})

compras['total'] = compras['cantidad'] * compras['precio_unitario']

# Promedio simple del precio unitario
promedio_simple = compras['precio_unitario'].mean()

# Promedio ponderado por cantidad
promedio_ponderado = (compras['total'].sum()) / (compras['cantidad'].sum())

# Precio mediano
precio_mediano = compras['precio_unitario'].median()

print("=== ANÁLISIS DE PRECIOS DE COMPRA ===")
print(f"Promedio simple: ${promedio_simple:.2f}/unidad")
print(f"Promedio ponderado: ${promedio_ponderado:.2f}/unidad")
print(f"Mediana: ${precio_mediano:.2f}/unidad")
print(f"Gasto total: ${compras['total'].sum():,.2f}")
print(f"Unidades compradas: {compras['cantidad'].sum():,}")
print(f"\n⚠ El promedio ponderado ({promedio_ponderado:.2f}) es el")
print(f"verdadero costo promedio por unidad comprada")
```

---

## 4.10 Ejemplo: Inventarios - Punto de Reorden

```python
# Cálculo de demanda promedio para punto de reorden
np.random.seed(42)

# Demanda diaria de los últimos 60 días
demanda_diaria = np.random.poisson(lam=25, size=60)

# Añadimos algunos picos estacionales
demanda_diaria[20:35] = np.random.poisson(lam=40, size=15)  # Temporada alta

# Cálculos
demanda_promedio = np.mean(demanda_diaria)
demanda_mediana = np.median(demanda_diaria)
demanda_moda = stats.mode(demanda_diaria)

lead_time = 5  # días
dias_cobertura = 7  # días de seguridad

# Punto de reorden (método simple)
punto_reodor_media = demanda_promedio * (lead_time + dias_cobertura)
punto_reodor_mediana = demanda_mediana * (lead_time + dias_cobertura)

print("=== PUNTO DE REORDEN ===")
print(f"Demanda promedio diaria: {demanda_promedio:.1f} uds")
print(f"Demanda mediana diaria: {demanda_mediana:.1f} uds")
print(f"Demanda modal: {demanda_moda.mode[0]} uds")
print(f"Lead time: {lead_time} días")
print(f"Días de cobertura: {dias_cobertura} días")
print(f"Punto de reorden (media): {punto_reodor_media:.0f} unidades")
print(f"Punto de reorden (mediana): {punto_reodor_mediana:.0f} unidades")

# ¿Cuál usar?
print(f"\nRecomendación: Usar la mediana ({demanda_mediana:.0f})")
print(f"porque la media ({demanda_promedio:.0f}) está inflada por")
print(f"la temporada alta (15 días con demanda de 40 uds)")
```

---

## 4.11 Robustez y Sensibilidad

| Medida | Sensible a Outliers | Uso Recomendado |
|--------|-------------------|-----------------|
| Media | **MUY sensible** | Datos limpios y simétricos |
| Mediana | Robusta | Datos con outliers o asimétricos |
| Moda | Robusta | Datos categóricos o discretos |
| Media Truncada | Parcialmente robusta | Datos con algunos outliers |
| Media Geométrica | Sensible (no acepta ceros) | Tasas de crecimiento |

```python
# Demostración de sensibilidad
datos_base = [100, 102, 98, 105, 101, 99, 103, 97, 104, 100]
datos_con_outlier = datos_base + [500]

medidas_originales = {
    'media': np.mean(datos_base),
    'mediana': np.median(datos_base),
    'truncada_10': stats.trim_mean(datos_base, 0.1)
}

medidas_con_outlier = {
    'media': np.mean(datos_con_outlier),
    'mediana': np.median(datos_con_outlier),
    'truncada_10': stats.trim_mean(datos_con_outlier, 0.1)
}

print("=== EFECTO DE UN OUTLIER ===")
print(f"{'Medida':15s} {'Original':>10s} {'Con Outlier':>15s} {'Cambio %':>10s}")
print("-" * 50)
for m in medidas_originales:
    orig = medidas_originales[m]
    out = medidas_con_outlier[m]
    cambio = (out - orig) / orig * 100
    print(f"{m:15s} {orig:>10.2f} {out:>15.2f} {cambio:>9.1f}%")
```

---

## 4.12 Resumen

- La **media** es el promedio aritmético, sensible a outliers
- La **mediana** es el valor central, robusta
- La **moda** es el valor más frecuente
- La **media ponderada** da peso diferencial a observaciones
- La **media geométrica** es para tasas de crecimiento
- Elegir la medida correcta depende de la distribución y el contexto de negocio

---

## Ejercicios Propuestos

1. **Ventas**: Calcula el ticket promedio, mediano y modal. ¿Hay diferencias significativas? ¿Por qué?
2. **Compras**: Usando la media ponderada, calcula el costo real promedio de 3 lotes de compra con diferentes cantidades y precios
3. **Inventarios**: Determina si la media o mediana es más apropiada para calcular el punto de reorden en un producto con demanda estacional

---

[← Anterior](03-estadistica-descriptiva.md) | [Índice](index.md) | [Siguiente →](05-medidas-dispersion.md)
