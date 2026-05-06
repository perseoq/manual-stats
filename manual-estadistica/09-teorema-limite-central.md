# Capítulo 9: Teorema del Límite Central

[← Anterior](08-distribuciones-continuas.md) | [Índice](index.md) | [Siguiente →](10-ley-grandes-numeros.md)

---

## 9.1 El Teorema Fundamental de la Estadística

El **Teorema del Límite Central (TLC)** es, sin duda, el resultado más importante de la estadística. Establece que:

> **La suma (o promedio) de un gran número de variables aleatorias independientes e idénticamente distribuidas (i.i.d.) se aproxima a una distribución Normal, independientemente de la distribución de las variables originales.**

### Implicaciones Revolucionarias

1. **Universalidad**: No importa la distribución original (Binomial, Poisson, Uniforme, Exponencial...), el promedio de muchas observaciones será aproximadamente Normal
2. **Predictibilidad**: Permite hacer inferencia sobre poblaciones sin conocer su forma exacta
3. **Aplicabilidad**: Justifica el uso de la Normal en prácticamente todos los métodos estadísticos

### Formalmente

Sea $X_1, X_2, ..., X_n$ una secuencia de variables aleatorias i.i.d. con media $\mu$ y varianza $\sigma^2$ finitas. Entonces:

$$\frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} N(0, 1)$$

Donde $\bar{X}_n = \frac{1}{n}\sum_{i=1}^{n} X_i$ es la media muestral.

---

## 9.2 Demostración Visual con Simulación

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

np.random.seed(42)

def demostrar_tlc(distribucion, nombre, n_muestras=10000, tamanos=[1, 2, 5, 10, 30, 100]):
    """Demuestra el TLC: la media muestral se vuelve Normal sin importar la distribución original"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, n in enumerate(tamanos):
        # Generamos n_muestras, cada una con n observaciones
        muestras = distribucion.rvs(size=(n_muestras, n))
        medias_muestrales = muestras.mean(axis=1)
        
        # Graficamos
        axes[i].hist(medias_muestrales, bins=50, density=True, 
                     alpha=0.7, color='steelblue', edgecolor='white')
        
        # Superponemos la Normal teórica
        mu_teorica = distribucion.mean()
        sigma_teorica = distribucion.std() / np.sqrt(n)
        x = np.linspace(medias_muestrales.min(), medias_muestrales.max(), 100)
        axes[i].plot(x, stats.norm.pdf(x, mu_teorica, sigma_teorica), 
                    'r-', linewidth=2, label='Normal teórica')
        
        axes[i].set_title(f'n = {n}', fontweight='bold')
        axes[i].set_xlabel('Media muestral')
        axes[i].set_ylabel('Densidad')
        axes[i].legend()
    
    fig.suptitle(f'TLC - Distribución Original: {nombre}', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'tlc_{nombre.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()

# --- Demostración 1: Uniforme ---
print("=== TLC: DISTRIBUCIÓN UNIFORME ===")
demostrar_tlc(stats.uniform(loc=0, scale=1), "Uniforme(0,1)")
```

```python
# --- Demostración 2: Exponencial (muy asimétrica) ---
print("=== TLC: DISTRIBUCIÓN EXPONENCIAL ===")
demostrar_tlc(stats.expon(scale=1), "Exponencial(1)")
```

```python
# --- Demostración 3: Binomial (discreta) ---
print("=== TLC: DISTRIBUCIÓN BINOMIAL (n=10, p=0.2) ===")
demostrar_tlc(stats.binom(n=10, p=0.2), "Binomial(10, 0.2)")
```

---

## 9.3 ¿Qué tan Grande debe ser n?

La velocidad de convergencia depende de la **asimetría** de la distribución original:

| Distribución Original | n mínimo recomendado |
|-----------------------|---------------------|
| Normal | 1 (ya es Normal) |
| Simétrica (Uniforme) | 5-10 |
| Moderadamente asimétrica | 15-20 |
| Muy asimétrica (Exponencial) | 30-50 |
| Extremadamente asimétrica (Log-normal) | 100+ |

```python
def evaluar_convergencia(distribucion, nombre, n_max=200, n_muestras=5000):
    """Evalúa qué tan rápido converge la media muestral a Normal"""
    errores = []
    
    for n in range(1, n_max + 1):
        muestras = distribucion.rvs(size=(n_muestras, n))
        medias = muestras.mean(axis=1)
        
        # Prueba de normalidad de Shapiro-Wilk (limitada a n≤5000)
        if n_muestras <= 5000:
            stat, p_valor = stats.shapiro(medias[:min(5000, len(medias))])
            errores.append(1 - p_valor)  # 1-p: qué tan lejos de normal
    
    return errores

# Comparación de convergencia
distribuciones = [
    (stats.uniform(loc=0, scale=1), "Uniforme"),
    (stats.expon(scale=1), "Exponencial"),
    (stats.binom(n=10, p=0.2), "Binomial(10, 0.2)"),
    (stats.poisson(mu=3), "Poisson(3)")
]

plt.figure(figsize=(12, 6))
for dist, nombre in distribuciones:
    errores = evaluar_convergencia(dist, nombre, n_max=100)
    plt.plot(range(1, 101), errores, label=nombre, linewidth=2)

plt.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5, label='p=0.05')
plt.xlabel('Tamaño de muestra (n)')
plt.ylabel('1 - p_valor (Shapiro-Wilk)')
plt.title('Velocidad de Convergencia al TLC', fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('convergencia_tlc.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 9.4 Error Estándar de la Media

El error estándar mide la **precisión de la media muestral** como estimador de la media poblacional:

$$SE = \frac{\sigma}{\sqrt{n}}$$

```python
print("=== ERROR ESTÁNDAR DE LA MEDIA ===")
sigma_poblacional = 100  # desviación estándar de la población

for n in [10, 25, 50, 100, 250, 500, 1000]:
    se = sigma_poblacional / np.sqrt(n)
    margen_error_95 = 1.96 * se
    print(f"n={n:4d}: SE={se:6.2f}, Margen error 95%={margen_error_95:±.2f}")

print("\nCuadruplicar la muestra reduce el error a la mitad")
```

### Relación Muestra-Precisión

```python
# Para un estudio de mercado
sigma_estimado = 30  # desviación típica de gasto mensual
margen_deseado = 5   # margen de error deseado (±$5)
z = 1.96  # 95% confianza

n_requerido = (z * sigma_estimado / margen_deseado) ** 2
print(f"=== TAMAÑO MUESTRAL NECESARIO ===")
print(f"Sigma estimado: ${sigma_estimado}")
print(f"Margen deseado: ±${margen_deseado}")
print(f"n requerido: {np.ceil(n_requerido):.0f} encuestas")
```

---

## 9.5 Ejemplo en Ventas: Error en Pronóstico

```python
# El error de pronóstico de ventas tiene distribución asimétrica
np.random.seed(42)

# Error de pronóstico (real - pronosticado) ~ Laplace(0, 100)
errores_individuales = stats.laplace.rvs(loc=0, scale=100, size=100000)

print("=== DISTRIBUCIÓN DEL ERROR DE PRONÓSTICO ===")
print(f"Media de errores individuales: {errores_individuales.mean():.2f}")
print(f"Desv. estándar: {errores_individuales.std():.2f}")

# Aplicamos TLC: promedios semanales (n=7) y mensuales (n=30)
for n_periodo, nombre in [(7, 'semanal'), (30, 'mensual')]:
    # Simulamos muchos períodos
    errores_reshape = errores_individuales[:len(errores_individuales)//n_periodo * n_periodo]
    errores_reshape = errores_reshape.reshape(-1, n_periodo)
    errores_promedio = errores_reshape.mean(axis=1)
    
    print(f"\nError de pronóstico {nombre} (n={n_periodo}):")
    print(f"  Media: {errores_promedio.mean():.2f}")
    print(f"  Desv. estándar: {errores_promedio.std():.2f}")
    print(f"  Simetría (skew): {stats.skew(errores_promedio):.3f}")
    print(f"  ¿Normal? Shapiro p-valor: {stats.shapiro(errores_promedio[:5000])[1]:.4f}")
```

---

## 9.6 Ejemplo en Compras: Precio Promedio de Insumos

```python
# Los precios individuales de un insumo tienen distribución asimétrica
np.random.seed(42)
precios_individuales = stats.gamma.rvs(a=2, scale=50, size=100000)

print("=== PRECIO DE INSUMOS - APLICACIÓN DEL TLC ===")
print(f"Distribución original (Gamma): media={precios_individuales.mean():.1f}, "
      f"mediana={np.median(precios_individuales):.1f}, skew={stats.skew(precios_individuales):.2f}")

# ¿Cuántas cotizaciones necesitamos para que el precio promedio sea confiable?
for n_cotizaciones in [1, 5, 10, 20, 50]:
    cotizaciones = precios_individuales[:n_cotizaciones * 10000].reshape(-1, n_cotizaciones)
    precios_promedio = cotizaciones.mean(axis=1)
    
    z = 1.96
    se = precios_promedio.std()
    margen = z * se
    
    print(f"\nCotizaciones promedio (n={n_cotizaciones:2d}):")
    print(f"  Media: ${precios_promedio.mean():.1f}")
    print(f"  SE: ${se:.1f}")
    print(f"  IC 95%: [${precios_promedio.mean() - margen:.1f}, "
          f"${precios_promedio.mean() + margen:.1f}]")

print("\n→ Con más cotizaciones, el precio promedio es más preciso")
print("→ El TLC permite usar la Normal aunque los precios individuales no lo sean")
```

---

## 9.7 Ejemplo en Inventarios: Demanda Promedio y Stock de Seguridad

```python
# La demanda diaria tiene distribución Poisson (discreta, asimétrica)
np.random.seed(42)

lambda_demanda = 20
dias_simulacion = 5000

# Demanda diaria individual
demanda_diaria = stats.poisson.rvs(mu=lambda_demanda, size=dias_simulacion)

print("=== DEMANDA DIARIA (POISSON) ===")
print(f"Distribución original: Poisson(λ={lambda_demanda})")
print(f"Media: {demanda_diaria.mean():.1f}, Varianza: {demanda_diaria.var():.1f}")

# Aplicación del TLC: demanda promedio semanal
demanda_semanal = demanda_diaria[:len(demanda_diaria)//7 * 7].reshape(-1, 7)
promedio_semanal = demanda_semanal.mean(axis=1)

print(f"\nDemanda promedio semanal (n=7):")
print(f"  Media: {promedio_semanal.mean():.2f}")
print(f"  Desv. estándar: {promedio_semanal.std():.2f}")
print(f"  Error estándar (SE): {demanda_diaria.std() / np.sqrt(7):.2f}")

# Stock de seguridad usando TLC
demanda_semanal_total = demanda_semanal.sum(axis=1)
print(f"\nDemanda semanal TOTAL:")
print(f"  Media: {demanda_semanal_total.mean():.1f}")
print(f"  Desv. estándar: {demanda_semanal_total.std():.1f}")

# Stock de seguridad para 95% nivel de servicio
z = 1.645
ss = z * demanda_semanal_total.std()
print(f"\n  Stock seguridad (95%): {ss:.0f} unidades")
print(f"  Punto reorden: {demanda_semanal_total.mean() + ss:.0f} unidades")
```

---

## 9.8 El TLC en la Práctica Empresarial

### Aplicaciones Concretas

| Aplicación | Variable Original | Variable Transformada | n típico |
|------------|------------------|----------------------|----------|
| Control de calidad | Dimensiones de piezas | Diámetro promedio | 5-30 |
| Pronóstico de ventas | Ventas diarias | Ventas semanales/mensuales | 7-30 |
| Encuestas de satisfacción | Respuestas Likert | Puntaje promedio | 30-500 |
| Muestreo de inventarios | Valor de cada SKU | Valor promedio del lote | 50-200 |
| Evaluación de proveedores | Días de entrega | Lead time promedio | 10-50 |

### El TLC NO aplica cuando:

1. **Dependencia**: Las observaciones no son independientes (ej: datos de series temporales con autocorrelación)
2. **Varianza infinita**: Distribuciones de Cauchy o Pareto con $\alpha \leq 2$
3. **Muestra muy pequeña**: n < 5-10 en la práctica
4. **Outliers extremos**: Pueden distorsionar el promedio

```python
# Verificación de robustez del TLC
np.random.seed(42)

# Distribución con outliers ocasionales (mezcla de normales)
def generar_con_outliers(n, p_outlier=0.02, factor=10):
    """Genera datos normales con algunos outliers extremos"""
    datos = np.random.normal(100, 15, n)
    mascara_outliers = np.random.random(n) < p_outlier
    datos[mascara_outliers] = datos[mascara_outliers] * factor
    return datos

datos_limpios = np.random.normal(100, 15, 1000)
datos_sucios = generar_con_outliers(1000)

print("=== ROBUSTEZ DEL TLC ANTE OUTLIERS ===")
for datos, nombre in [(datos_limpios, 'Limpios'), (datos_sucios, 'Con outliers')]:
    # Distribución de medias con n=30
    medias = [np.mean(np.random.choice(datos, 30, replace=True)) for _ in range(1000)]
    _, p_normal = stats.shapiro(medias[:500])
    print(f"\n{nombre}:")
    print(f"  Media global: {datos.mean():.2f}")
    print(f"  Shapiro p-valor (normalidad de medias): {p_normal:.4f}")
    print(f"  {'Aprox. Normal' if p_normal > 0.05 else 'NO Normal'}")
```

---

## 9.9 Teorema de De Moivre-Laplace (Caso Binomial)

Caso especial del TLC: la Binomial se aproxima a Normal cuando $n$ es grande:

$$X \sim Binomial(n,p) \implies \frac{X - np}{\sqrt{np(1-p)}} \xrightarrow{d} N(0,1)$$

```python
print("=== APROXIMACIÓN NORMAL DE LA BINOMIAL ===")
n, p = 100, 0.3
binomial = stats.binom(n, p)
normal_aprox = stats.norm(n*p, np.sqrt(n*p*(1-p)))

for k in range(20, 41, 5):
    prob_binom = binomial.pmf(k)
    prob_norm = normal_aprox.pdf(k)  # Aproximación
    print(f"P(X={k:2d}): Binomial={prob_binom:.4f}, Normal≈{prob_norm:.4f}")

print(f"\nP(X ≤ 25) - Binomial: {binomial.cdf(25):.4f}")
print(f"P(X ≤ 25) - Normal:   {normal_aprox.cdf(25.5):.4f} (con corrección de continuidad)")
```

---

## 9.10 Resumen

- El **TLC** es la razón por la que la Normal aparece en casi toda la estadística
- La **media muestral** de cualquier distribución se vuelve Normal al aumentar n
- El **error estándar** ($\sigma/\sqrt{n}$) mide la precisión de la estimación
- n ≥ 30 es una regla general, pero depende de la asimetría original
- El TLC es la base de intervalos de confianza, pruebas de hipótesis y control de calidad

---

## Ejercicios Propuestos

1. **Ventas**: Simula 10,000 muestras de tamaño n=30 de una distribución Exponencial(λ=0.01) y verifica que las medias son aproximadamente Normales
2. **Compras**: Si el precio de un insumo tiene distribución Gamma(α=2, β=25), ¿cuántas cotizaciones necesitas para que el precio promedio tenga un error estándar < $3?
3. **Inventarios**: Demuestra con simulación que la demanda mensual (30 días) de un producto Poisson(λ=15/día) se aproxima a Normal

---

[← Anterior](08-distribuciones-continuas.md) | [Índice](index.md) | [Siguiente →](10-ley-grandes-numeros.md)
