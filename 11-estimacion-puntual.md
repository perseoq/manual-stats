# Capítulo 11: Estimación Puntual

[← Anterior](10-ley-grandes-numeros.md) | [Índice](index.md) | [Siguiente →](12-intervalos-confianza.md)

---

## 11.1 Introducción a la Estimación

La **estimación puntual** consiste en usar datos de una muestra para calcular un único valor (el estimador) que aproxime un parámetro poblacional desconocido.

### Parámetro vs Estimador

| Concepto | Población | Muestra |
|----------|-----------|---------|
| **Media** | $\mu$ (parámetro) | $\bar{x}$ (estimador) |
| **Varianza** | $\sigma^2$ (parámetro) | $s^2$ (estimador) |
| **Proporción** | $p$ (parámetro) | $\hat{p}$ (estimador) |
| **Desv. Estándar** | $\sigma$ (parámetro) | $s$ (estimador) |

### Propiedades de los Buenos Estimadores

1. **Insesgado (Sesgo = 0)**: $E[\hat{\theta}] = \theta$
2. **Consistente**: $\hat{\theta}_n \xrightarrow{p} \theta$ cuando $n \to \infty$
3. **Eficiente**: Mínima varianza entre todos los estimadores insesgados
4. **Suficiente**: Utiliza toda la información relevante en la muestra

---

## 11.2 Método de Momentos (MM)

Iguala los momentos muestrales con los teóricos:

- 1er momento: $\frac{1}{n}\sum X_i = E[X]$
- 2do momento: $\frac{1}{n}\sum X_i^2 = E[X^2]$

```python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)

class EstimadorMomentos:
    """Estimación por método de momentos para diferentes distribuciones"""
    
    @staticmethod
    def estimar_normal(datos):
        """Estimar μ y σ de una Normal"""
        mu_est = np.mean(datos)
        sigma_est = np.std(datos, ddof=0)  # Momentos usa n, no n-1
        return {'μ_hat': mu_est, 'σ_hat': sigma_est}
    
    @staticmethod
    def estimar_poisson(datos):
        """Estimar λ de una Poisson"""
        # E[X] = λ, E[X²] = λ + λ²
        # Ambos momentos dan λ = media
        lambda_est = np.mean(datos)
        return {'λ_hat': lambda_est}
    
    @staticmethod
    def estimar_exponencial(datos):
        """Estimar λ de una Exponencial"""
        # E[X] = 1/λ
        lambda_est = 1 / np.mean(datos)
        return {'λ_hat': lambda_est}
    
    @staticmethod
    def estimar_gamma(datos):
        """Estimar α y β de una Gamma"""
        # E[X] = αβ, E[X²] = α(α+1)β²
        media = np.mean(datos)
        varianza = np.var(datos, ddof=0)
        alpha_est = media**2 / varianza
        beta_est = varianza / media
        return {'α_hat': alpha_est, 'β_hat': beta_est}

# --- Ejemplo: Estimación de parámetros de ventas ---
ventas_mensuales = stats.norm.rvs(loc=100, scale=20, size=50)
resultado = EstimadorMomentos.estimar_normal(ventas_mensuales)
print("=== ESTIMACIÓN POR MOMENTOS - VENTAS ===")
print(f"μ real = 100, μ estimado = {resultado['μ_hat']:.2f}")
print(f"σ real = 20, σ estimado = {resultado['σ_hat']:.2f}")
```

---

## 11.3 Máxima Verosimilitud (MLE)

Encuentra los parámetros que **maximizan la probabilidad** de observar los datos.

$$L(\theta | datos) = \prod_{i=1}^{n} f(x_i | \theta)$$
$$\hat{\theta}_{MLE} = \arg\max_{\theta} L(\theta | datos)$$

```python
# Implementación manual de MLE para Normal
def mle_normal(datos):
    """MLE para distribución Normal"""
    n = len(datos)
    mu_mle = np.mean(datos)
    sigma_mle = np.sqrt(np.sum((datos - mu_mle)**2) / n)
    return mu_mle, sigma_mle

# Comparación MLE con SciPy
ventas = stats.norm.rvs(loc=50000, scale=8000, size=100)
mu_mle, sigma_mle = mle_normal(ventas)

print("=== MLE - DISTRIBUCIÓN NORMAL ===")
print(f"Parámetros reales: μ=50000, σ=8000")
print(f"MLE estimados: μ={mu_mle:.1f}, σ={sigma_mle:.1f}")
print(f"SciPy MLE: μ={stats.norm.fit(ventas)[0]:.1f}, σ={stats.norm.fit(ventas)[1]:.1f}")

# MLE para Poisson
def mle_poisson(datos):
    """MLE para Poisson: λ = media muestral"""
    return np.mean(datos)

demanda = stats.poisson.rvs(mu=25, size=50)
lambda_mle = mle_poisson(demanda)
print(f"\nMLE Poisson - λ real=25, λ estimado={lambda_mle:.2f}")
```

### MLE con Optimización Numérica

```python
from scipy.optimize import minimize

# MLE por optimización para cualquier distribución
def mle_optimizacion(datos, funcion_neg_loglik, params_iniciales):
    """MLE mediante optimización numérica"""
    def neg_loglik(params):
        return funcion_neg_loglik(datos, *params)
    
    resultado = minimize(neg_loglik, params_iniciales, method='Nelder-Mead')
    return resultado.x

# Ejemplo: distribución Gamma
def neg_loglik_gamma(datos, alpha, beta):
    """Log-verosimilitud negativa para Gamma"""
    if alpha <= 0 or beta <= 0:
        return np.inf
    return -np.sum(stats.gamma.logpdf(datos, a=alpha, scale=beta))

# Datos simulados
datos_gamma = stats.gamma.rvs(a=5, scale=10, size=200, random_state=42)
alpha_est, beta_est = mle_optimizacion(datos_gamma, neg_loglik_gamma, [3, 8])

print("=== MLE POR OPTIMIZACIÓN - GAMMA ===")
print(f"Parámetros reales: α=5, β=10")
print(f"MLE estimados: α={alpha_est:.2f}, β={beta_est:.2f}")
```

---

## 11.4 Estimación de Proporciones

$$\hat{p} = \frac{X}{n} \quad \text{donde X = número de éxitos}$$

```python
# Estimación de tasa de conversión
n_leads = 200
conversiones = 54  # 54 ventas de 200 leads

p_hat = conversiones / n_leads
print("=== ESTIMACIÓN DE PROPORCIÓN ===")
print(f"Tasa de conversión estimada: {p_hat:.2%}")
print(f"Error estándar (SE): {np.sqrt(p_hat * (1-p_hat) / n_leads):.4f}")
```

---

## 11.5 Insesgamiento y Sesgo

$$Bias(\hat{\theta}) = E[\hat{\theta}] - \theta$$

```python
def demostrar_sesgo(n_sim=10000, n_muestra=20, mu_real=100, sigma_real=15):
    """Demuestra el sesgo en estimación de varianza"""
    estimadores_n = []
    estimadores_n1 = []
    
    for _ in range(n_sim):
        muestra = np.random.normal(mu_real, sigma_real, n_muestra)
        # Varianza con n (sesgada)
        estimadores_n.append(np.var(muestra, ddof=0))
        # Varianza con n-1 (insesgada)
        estimadores_n1.append(np.var(muestra, ddof=1))
    
    varianza_real = sigma_real**2
    
    print("=== DEMOSTRACIÓN DE INSESGAMIENTO ===")
    print(f"Varianza real (σ²): {varianza_real}")
    print(f"Media estimador con n: {np.mean(estimadores_n):.2f} (SESGADO)")
    print(f"Media estimador con n-1: {np.mean(estimadores_n1):.2f} (INSESGADO)")
    print(f"Sesgo (n): {np.mean(estimadores_n) - varianza_real:.2f}")
    print(f"Sesgo (n-1): {np.mean(estimadores_n1) - varianza_real:.2f}")

demostrar_sesgo()
```

---

## 11.6 Error Cuadrático Medio (MSE)

$$MSE(\hat{\theta}) = E[(\hat{\theta} - \theta)^2] = Var(\hat{\theta}) + Bias(\hat{\theta})^2$$

```python
def calcular_mse(estimador, parametro_real, n_sim=10000):
    """Calcula el MSE de un estimador"""
    errores_cuad = [(est - parametro_real)**2 for est in estimador]
    return np.mean(errores_cuad)

# Comparación de estimadores
np.random.seed(42)
n_sim = 50000
n_muestra = 10

estim_media = []
estim_mediana = []
estim_truncada = []

for _ in range(n_sim):
    muestra = np.random.exponential(scale=5, size=n_muestra)
    estim_media.append(np.mean(muestra))
    estim_mediana.append(np.median(muestra))
    estim_truncada.append(stats.trim_mean(muestra, 0.1))

mu_real = 5  # Media de Exp(5)

print("=== COMPARACIÓN DE MSE ===")
print(f"Parámetro real (μ de Exponencial): {mu_real}")
print(f"MSE de la media: {calcular_mse(estim_media, mu_real):.4f}")
print(f"MSE de la mediana: {calcular_mse(estim_mediana, mu_real):.4f}")
print(f"MSE de media truncada: {calcular_mse(estim_truncada, mu_real):.4f}")
```

---

## 11.7 Estimación Robusta

Los estimadores robustos son menos sensibles a outliers.

```python
from scipy.stats import trim_mean

# Datos limpios vs contaminados
datos_limpios = stats.norm.rvs(loc=100, scale=15, size=100)
datos_contaminados = np.append(datos_limpios, [500, 600, -100, -200])

print("=== ESTIMACIÓN ROBUSTA ===")
print(f"{'Métrica':15s} {'Limpios':>10s} {'Contaminados':>15s}")
print("-" * 42)
print(f"{'Media':15s} {np.mean(datos_limpios):>10.2f} {np.mean(datos_contaminados):>15.2f}")
print(f"{'Mediana':15s} {np.median(datos_limpios):>10.2f} {np.median(datos_contaminados):>15.2f}")
print(f"{'Truncada 10%':15s} {trim_mean(datos_limpios, 0.1):>10.2f} {trim_mean(datos_contaminados, 0.1):>15.2f}")
print(f"{'Winsorizada':15s} {stats.mstats.winsorize(datos_limpios, 0.1).mean():>10.2f} "
      f"{stats.mstats.winsorize(datos_contaminados, 0.1).mean():>15.2f}")
```

---

## 11.8 Ejemplo en Ventas: Estimación de Ticket Promedio

```python
# Estimación del ticket promedio de compra
np.random.seed(42)

# Datos reales
tickets_reales = stats.gamma.rvs(a=3, scale=30, size=1000) + 20
ticket_real_medio = tickets_reales.mean()

# Muestreo
for n_muestra in [10, 30, 50, 100, 500]:
    muestra = np.random.choice(tickets_reales, n_muestra, replace=False)
    media_est = np.mean(muestra)
    mediana_est = np.median(muestra)
    error_media = abs(media_est - ticket_real_medio)
    error_mediana = abs(mediana_est - ticket_real_medio)
    
    print(f"n={n_muestra:3d}: Media={media_est:6.2f} (error={error_media:5.2f}), "
          f"Mediana={mediana_est:6.2f} (error={error_mediana:5.2f})")

print(f"\nTicket real medio: ${ticket_real_medio:.2f}")
```

---

## 11.9 Ejemplo en Compras: Costo Unitario Estimado

```python
# Estimación del costo unitario de un insumo
np.random.seed(42)

# Historial de compras
costos = [95, 102, 98, 110, 97, 105, 93, 108, 100, 96,
          101, 104, 99, 106, 94, 107, 103, 92, 109, 111]

print("=== ESTIMACIÓN DE COSTO UNITARIO ===")
print(f"Costo medio estimado: ${np.mean(costos):.2f}")
print(f"Error estándar: ${np.std(costos, ddof=1)/np.sqrt(len(costos)):.2f}")
print(f"IC 95% aproximado: ${np.mean(costos) - 1.96*np.std(costos, ddof=1)/np.sqrt(len(costos)):.2f} "
      f"a ${np.mean(costos) + 1.96*np.std(costos, ddof=1)/np.sqrt(len(costos)):.2f}")
```

---

## 11.10 Ejemplo en Inventarios: Estimación de Demanda

```python
# Estimación de parámetros de demanda
np.random.seed(42)

# La demanda sigue una distribución Poisson
lambda_real = 45
demanda_observada = stats.poisson.rvs(mu=lambda_real, size=60)

# Estimación puntual
lambda_est = np.mean(demanda_observada)

print("=== ESTIMACIÓN DE DEMANDA ===")
print(f"λ real: {lambda_real}")
print(f"λ estimado (media muestral): {lambda_est:.2f}")
print(f"Error de estimación: {abs(lambda_est - lambda_real):.2f}")
print(f"SE: {np.sqrt(lambda_est / len(demanda_observada)):.2f}")

# Implicación: stock de seguridad
z = 1.645
ss = z * np.sqrt(lambda_est)
print(f"\nStock de seguridad recomendado (95%): {ss:.0f} unidades")
print(f"Punto de reorden: {lambda_est * 3 + ss:.0f} unidades")
```

---

## 11.11 Comparación de Métodos de Estimación

| Método | Ventajas | Desventajas |
|--------|----------|-------------|
| **Momentos** | Simple, siempre computable | Puede ser ineficiente |
| **MLE** | Eficiente, insesgado asintóticamente | Requiere optimización numérica |
| **Bayesiano** | Incorpora información previa | Requiere prior |
| **Robusto** | Resistente a outliers | Menos eficiente con datos limpios |

```python
# Comparación completa
np.random.seed(42)

n_sim = 5000
resultados = {'media': [], 'mediana': [], 'truncada': [], 'mle': []}

for _ in range(n_sim):
    datos = stats.norm.rvs(loc=100, scale=15, size=30)
    resultados['media'].append(np.mean(datos))
    resultados['mediana'].append(np.median(datos))
    resultados['truncada'].append(stats.trim_mean(datos, 0.1))
    resultados['mle'].append(stats.norm.fit(datos)[0])

print("=== COMPARACIÓN DE ESTIMADORES (n=30, Normal(100,15)) ===")
print(f"{'Método':12s} {'Media':>8s} {'Std':>8s} {'MSE':>8s} {'Sesgo':>8s}")
print("-" * 44)
for nombre, vals in resultados.items():
    media = np.mean(vals)
    std = np.std(vals)
    mse = np.mean((np.array(vals) - 100)**2)
    sesgo = media - 100
    print(f"{nombre:12s} {media:>8.3f} {std:>8.3f} {mse:>8.3f} {sesgo:>8.3f}")
```

---

## 11.12 Resumen

- La **estimación puntual** produce un único valor para un parámetro desconocido
- Buenos estimadores son **insesgados, consistentes, eficientes**
- **MLE** es el método más usado (eficiente asintóticamente)
- El **MSE** balancea varianza y sesgo
- La **estimación robusta** protege contra outliers

---

## Ejercicios Propuestos

1. **Ventas**: Estima la tasa de conversión de 250 leads (85 conversiones). Calcula SE y sesgo
2. **Compras**: Usando el método de momentos, estima α y β de una distribución Gamma para costos de proveedor
3. **Inventarios**: Con 90 días de demanda observada ([Poisson λ=30]), estima λ, calcula SE y determina stock de seguridad

---

[← Anterior](10-ley-grandes-numeros.md) | [Índice](index.md) | [Siguiente →](12-intervalos-confianza.md)
