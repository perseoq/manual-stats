# Capítulo 32: Muestreo

[← Anterior](31-control-calidad.md) | [Índice](index.md) | [Siguiente →](33-analisis-varianza.md)

---

## 32.1 Introducción

El **muestreo** es el proceso de seleccionar un subconjunto (muestra) de una población para hacer inferencias sobre toda la población. Un buen muestreo es clave para la validez de cualquier análisis estadístico.

### ¿Por qué muestrear?

1. **Costo**: Estudiar toda la población es caro
2. **Tiempo**: Más rápido que un censo
3. **Precisión**: Muestras bien diseñadas pueden ser más precisas que censos mal ejecutados
4. **Imposibilidad práctica**: Poblaciones infinitas o destructivas

### Error de Muestreo vs Error No Muestral

| Tipo | Definición | Ejemplo |
|------|-----------|---------|
| **Error de muestreo** | Diferencia entre muestra y población por azar | La media muestral difiere de la poblacional |
| **Sesgo de muestreo** | Error sistemático en la selección | Solo encuestar clientes que se quejan |
| **Error de medición** | Instrumento imperfecto | Balanza mal calibrada |
| **Error de cobertura** | Marco muestral incompleto | Lista de clientes desactualizada |

---

## 32.2 Tipos de Muestreo

### Probabilísticos (Cada elemento tiene probabilidad conocida)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Población de 10,000 clientes
poblacion = pd.DataFrame({
    'cliente_id': range(1, 10001),
    'segmento': np.random.choice(['Premium', 'Estándar', 'Económico'], 10000, 
                                  p=[0.2, 0.5, 0.3]),
    'region': np.random.choice(['Norte', 'Sur', 'Centro', 'Oeste'], 10000),
    'gasto_anual': np.random.gamma(5, 2000, 10000),
    'antiguedad': np.random.randint(1, 120, 10000)
})

print("=== POBLACIÓN ===")
print(f"Total clientes: {len(poblacion):,}")
print(f"Gasto medio: ${poblacion['gasto_anual'].mean():,.0f}")
```

### Muestreo Aleatorio Simple (MAS)

```python
def muestreo_aleatorio_simple(poblacion, n, semilla=42):
    """Selecciona n elementos aleatoriamente"""
    np.random.seed(semilla)
    indices = np.random.choice(len(poblacion), n, replace=False)
    return poblacion.iloc[indices]

# Ejemplo
muestra_mas = muestreo_aleatorio_simple(poblacion, 200)
print("=== MUESTREO ALEATORIO SIMPLE ===")
print(f"n = {len(muestra_mas)}")
print(f"Media muestral: ${muestra_mas['gasto_anual'].mean():,.0f}")
```

### Muestreo Estratificado

```python
def muestreo_estratificado(poblacion, col_estrato, n_total, semilla=42):
    """Muestreo estratificado proporcional"""
    np.random.seed(semilla)
    estratos = poblacion[col_estrato].value_counts(normalize=True)
    
    muestras = []
    for estrato, proporcion in estratos.items():
        n_estrato = int(np.ceil(n_total * proporcion))
        poblacion_estrato = poblacion[poblacion[col_estrato] == estrato]
        indices = np.random.choice(len(poblacion_estrato), 
                                   min(n_estrato, len(poblacion_estrato)), 
                                   replace=False)
        muestras.append(poblacion_estrato.iloc[indices])
    
    return pd.concat(muestras).sample(frac=1, random_state=semilla)

# Ejemplo
muestra_estrat = muestreo_estratificado(poblacion, 'segmento', 200)
print("\n=== MUESTREO ESTRATIFICADO ===")
print(f"n = {len(muestra_estrat)}")
print(f"Media muestral: ${muestra_estrat['gasto_anual'].mean():,.0f}")
print(f"Composición por segmento:")
print(muestra_estrat['segmento'].value_counts(normalize=True))
```

### Muestreo por Conglomerados

```python
def muestreo_conglomerados(poblacion, col_conglomerado, n_conglomerados, semilla=42):
    """Selecciona conglomerados completos"""
    np.random.seed(semilla)
    conglomerados = poblacion[col_conglomerado].unique()
    seleccionados = np.random.choice(conglomerados, n_conglomerados, replace=False)
    return poblacion[poblacion[col_conglomerado].isin(seleccionados)]

# Ejemplo
muestra_conglom = muestreo_conglomerados(poblacion, 'region', 2)
print("\n=== MUESTREO POR CONGLOMERADOS ===")
print(f"Regiones seleccionadas: {muestra_conglom['region'].unique()}")
print(f"n = {len(muestra_conglom)}")
```

### Muestreo Sistemático

```python
def muestreo_sistematico(poblacion, n, semilla=42):
    """Selecciona cada k-ésimo elemento"""
    np.random.seed(semilla)
    N = len(poblacion)
    k = N // n
    inicio = np.random.randint(0, k)
    indices = np.arange(inicio, N, k)[:n]
    return poblacion.iloc[indices]

# Ejemplo
muestra_sist = muestreo_sistematico(poblacion, 200)
print("\n=== MUESTREO SISTEMÁTICO ===")
print(f"n = {len(muestra_sist)}")
print(f"Media muestral: ${muestra_sist['gasto_anual'].mean():,.0f}")
```

---

## 32.3 Tamaño Muestral

### Para la Media

$$n = \frac{Z^2 \cdot \sigma^2}{E^2}$$

### Para la Proporción

$$n = \frac{Z^2 \cdot p(1-p)}{E^2}$$

```python
def calcular_tamano_muestral_media(margen, sigma, nivel_confianza=0.95):
    """Calcula n necesario para estimar la media"""
    z = stats.norm.ppf(1 - (1 - nivel_confianza) / 2)
    n = (z * sigma / margen) ** 2
    return np.ceil(n)

def calcular_tamano_muestral_proporcion(margen, p=0.5, nivel_confianza=0.95):
    """Calcula n necesario para estimar una proporción"""
    z = stats.norm.ppf(1 - (1 - nivel_confianza) / 2)
    n = (z**2 * p * (1-p)) / margen**2
    return np.ceil(n)

print("=== TAMAÑO MUESTRAL ===")
print(f"Para media (σ=$500, margen=$50): n = {calcular_tamano_muestral_media(50, 500)}")
print(f"Para media (σ=$500, margen=$100): n = {calcular_tamano_muestral_media(100, 500)}")
print(f"Para proporción (margen=3%): n = {calcular_tamano_muestral_proporcion(0.03)}")
print(f"Para proporción (margen=5%): n = {calcular_tamano_muestral_proporcion(0.05)}")
```

### Población Finita (Corrección)

```python
def tamano_muestral_finito(n_infinito, N):
    """Corrección para población finita"""
    return n_infinito / (1 + n_infinito / N)

# Ejemplo con población de 5,000
for N_poblacion in [500, 1000, 5000, 50000]:
    n_inf = calcular_tamano_muestral_media(50, 500)
    n_fin = tamano_muestral_finito(n_inf, N_poblacion)
    print(f"N={N_poblacion:>6,}: n_infinito={n_inf:.0f}, n_finito={n_fin:.0f}")
```

---

## 32.4 Error de Muestreo

```python
def simular_error_muestreo(poblacion, n, n_sim=1000):
    """Simula el error de muestreo para diferentes n"""
    errores = []
    medias_pob = poblacion['gasto_anual'].mean()
    
    for _ in range(n_sim):
        muestra = muestreo_aleatorio_simple(poblacion, n)
        error = muestra['gasto_anual'].mean() - medias_pob
        errores.append(error)
    
    return np.array(errores)

# Comparar errores para diferentes tamaños
medias_pob = poblacion['gasto_anual'].mean()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, n in enumerate([30, 100, 500]):
    errores = simular_error_muestreo(poblacion, n, 2000)
    axes[i].hist(errores, bins=30, density=True, alpha=0.7, 
                 color='steelblue', edgecolor='white')
    axes[i].axvline(0, color='red', linestyle='--')
    axes[i].set_title(f'n = {n}, SE = {errores.std():.0f}', fontweight='bold')
    axes[i].set_xlabel('Error de muestreo')

plt.tight_layout()
plt.savefig('error_muestreo.png', dpi=150, bbox_inches='tight')
plt.show()

print("=== ERROR DE MUESTREO ===")
for n in [30, 100, 500]:
    errores = simular_error_muestreo(poblacion, n, 2000)
    print(f"n={n:3d}: SE={errores.std():.0f}, Error máx 95%={np.percentile(np.abs(errores), 95):.1f}")
```

---

## 32.5 Ejemplo en Ventas: Encuesta de Satisfacción

```python
# Diseño de encuesta de satisfacción
print("=== ENCUESTA DE SATISFACCIÓN ===")
p_cliente_satis = 0.85  # Estimación inicial
margen_deseado = 0.03  # ±3%
n_encuestas = calcular_tamano_muestral_proporcion(margen_deseado, p_cliente_satis)
print(f"Clientes totales: 15,000")
print(f"Satisfacción estimada: {p_cliente_satis:.0%}")
print(f"Tamaño muestral necesario: {n_encuestas:.0f}")
print(f"Corregido (población finita): {tamano_muestral_finito(n_encuestas, 15000):.0f}")
```

---

## 32.6 Ejemplo en Compras: Inspección de Lotes

```python
# Plan de muestreo para aceptación de lotes
print("=== PLAN DE MUESTREO PARA ACEPTACIÓN ===")
N_lote = 1000
nivel_calidad_aceptable = 0.02  # AQL = 2%
nivel_calidad_rechazable = 0.08  # LTPD = 8%
riesgo_productor = 0.05  # α
riesgo_consumidor = 0.10  # β

# Plan de muestreo simple (simplificado)
n_muestra = 80
c_aceptacion = 3  # máximo defectos para aceptar

# Curva OC (Operating Characteristic)
def probabilidad_aceptacion(p, n, c):
    """Probabilidad de aceptar un lote con calidad p"""
    return stats.binom.cdf(c, n, p)

print(f"Plan: n={n_muestra}, c={c_aceptacion}")
print(f"\nCurva OC:")
for p in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]:
    pa = probabilidad_aceptacion(p, n_muestra, c_aceptacion)
    print(f"  p={p:.0%}: P(aceptar)={pa:.3f}")
```

---

## 32.7 Ejemplo en Inventarios: Conteo Cíclico

```python
# Diseño de muestreo para conteo cíclico de inventario
print("=== MUESTREO PARA CONTEO CÍCLICO ===")

# Población de SKUs
n_skus = 5000
precision_deseada = 0.02  # ±2%
var_estimada = 0.15  # 15% de variabilidad

n_conteo = calcular_tamano_muestral_media(precision_deseada, var_estimada)
print(f"SKUs totales: {n_skus}")
print(f"Precisión deseada: ±{precision_deseada:.0%}")
print(f"SKUs a contar: {n_conteo:.0f} por período")
print(f"Días por conteo: {n_conteo / 20:.0f} SKUs/día (20 días hábiles)")

# Asignación por categoría ABC
categoria_a = n_skus * 0.2  # 20% de SKU
print(f"\nEstrategia: Contar categoría A cada mes, B cada 3 meses, C cada 6 meses")
```

---

## 32.8 Resumen

| Método | Ventajas | Desventajas |
|--------|----------|-------------|
| **Aleatorio Simple** | Imparcial, fácil de analizar | Requiere marco muestral completo |
| **Estratificado** | Reduce error, asegura representación | Requiere conocer estratos |
| **Conglomerados** | Bajo costo operativo | Mayor error de muestreo |
| **Sistemático** | Simple, buena distribución espacial | Puede ocultar periodicidad |

---

## Ejercicios Propuestos

1. **Ventas**: Calcula el tamaño muestral para estimar el gasto promedio con margen ±$50 (σ=$400)
2. **Compras**: Diseña un plan de muestreo para inspeccionar lotes de 2000 unidades con AQL=1.5%
3. **Inventarios**: Determina cuántos SKUs contar si hay 10,000 SKUs y quieres precisión de ±1%

---

[← Anterior](31-control-calidad.md) | [Índice](index.md) | [Siguiente →](33-analisis-varianza.md)
