# Capítulo 12: Intervalos de Confianza

[← Anterior](11-estimacion-puntual.md) | [Índice](index.md) | [Siguiente →](13-pruebas-hipotesis.md)

---

## 12.1 Introducción

Un **intervalo de confianza (IC)** es un rango de valores que, con un nivel de confianza determinado, contiene el verdadero valor del parámetro poblacional.

### IC vs Estimación Puntual

| Estimación Puntual | Intervalo de Confianza |
|-------------------|----------------------|
| Un solo número | Un rango de valores |
| "La media es 100" | "La media está entre 95 y 105" |
| Certidumbre falsa | Incertidumbre explícita |
| Sin medida de precisión | Con nivel de confianza |

### Interpretación Correcta

> "Si repitiéramos el muestreo muchas veces, el **95%** de los intervalos calculados contendrían el verdadero parámetro."

NO es: "Hay 95% de probabilidad de que el parámetro esté en este intervalo" (esa es la interpretación bayesiana).

---

## 12.2 Estructura General

$$IC = \text{Estimación} \pm \text{Margen de Error}$$

$$\text{Margen de Error} = \text{Valor Crítico} \times \text{Error Estándar}$$

### Valores Críticos Comunes

| Nivel Confianza | z (Normal) | t (gl=30) | t (gl=100) |
|----------------|------------|-----------|------------|
| 90% | 1.645 | 1.697 | 1.660 |
| 95% | **1.960** | **2.042** | **1.984** |
| 99% | 2.576 | 2.750 | 2.626 |

---

## 12.3 IC para la Media (σ conocida - Z)

$$IC: \bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$

```python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

def ic_media_z(datos, sigma_poblacional, nivel_confianza=0.95):
    """Intervalo de confianza para la media (σ conocida)"""
    n = len(datos)
    media = np.mean(datos)
    se = sigma_poblacional / np.sqrt(n)
    z = stats.norm.ppf(1 - (1 - nivel_confianza) / 2)
    
    margen = z * se
    return media - margen, media + margen, media, margen

np.random.seed(42)
# Ventas diarias con σ conocida = 30
ventas = stats.norm.rvs(loc=150, scale=30, size=50)
li, ls, media, margen = ic_media_z(ventas, 30, 0.95)

print("=== IC PARA MEDIA (σ CONOCIDA) ===")
print(f"Media muestral: {media:.2f}")
print(f"Error estándar: {30/np.sqrt(50):.2f}")
print(f"IC 95%: [{li:.2f}, {ls:.2f}]")
print(f"Margen de error: ±{margen:.2f}")
```

---

## 12.4 IC para la Media (σ desconocida - t)

$$IC: \bar{x} \pm t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}}$$

```python
def ic_media_t(datos, nivel_confianza=0.95):
    """IC para la media (σ desconocida, usa t-Student)"""
    n = len(datos)
    media = np.mean(datos)
    s = np.std(datos, ddof=1)
    se = s / np.sqrt(n)
    t = stats.t.ppf(1 - (1 - nivel_confianza) / 2, df=n-1)
    
    margen = t * se
    return media - margen, media + margen, media, margen, s

# Ejemplo: evaluación de ingreso promedio
np.random.seed(42)
ingresos = stats.norm.rvs(loc=50000, scale=12000, size=30)
li, ls, media, margen, s = ic_media_t(ingresos)

print("=== IC PARA MEDIA (σ DESCONOCIDA) ===")
print(f"Media: ${media:,.1f}")
print(f"Desv. estándar muestral: ${s:,.1f}")
print(f"n = {len(ingresos)}")
print(f"IC 95%: [${li:,.1f}, ${ls:,.1f}]")
print(f"Margen: ±${margen:,.1f}")

# Efecto del tamaño muestral
print(f"\nEfecto de n:")
for n_muestra in [10, 30, 50, 100, 500]:
    muestra = np.random.choice(ingresos, n_muestra, replace=True) if n_muestra > len(ingresos) \
              else ingresos[:n_muestra]
    li_n, ls_n, _, margen_n, _ = ic_media_t(muestra)
    print(f"  n={n_muestra:3d}: IC=[${li_n:,.0f}, ${ls_n:,.0f}], Margen=±${margen_n:,.0f}")
```

---

## 12.5 IC para Proporciones

$$IC: \hat{p} \pm z_{\alpha/2} \cdot \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$$

```python
def ic_proporcion(x, n, nivel_confianza=0.95):
    """IC para una proporción"""
    p_hat = x / n
    z = stats.norm.ppf(1 - (1 - nivel_confianza) / 2)
    se = np.sqrt(p_hat * (1 - p_hat) / n)
    margen = z * se
    
    li = max(0, p_hat - margen)
    ls = min(1, p_hat + margen)
    return li, ls, p_hat, margen

# Tasa de conversión
n_leads = 500
conversiones = 135

li, ls, p_hat, margen = ic_proporcion(conversiones, n_leads)

print("=== IC PARA PROPORCIÓN ===")
print(f"Tasa estimada: {p_hat:.2%}")
print(f"IC 95%: [{li:.2%}, {ls:.2%}]")
print(f"Margen: ±{margen:.2%}")

# Efecto del tamaño muestral
print(f"\n¿Cuántos leads necesito para margen < 2%?")
for n_test in [200, 500, 1000, 2000, 5000]:
    _, _, _, m = ic_proporcion(int(n_test * 0.27), n_test)  # p≈27%
    print(f"  n={n_test:4d}: margen=±{m:.2%}")
```

---

## 12.6 IC para la Varianza

$$IC: \left[\frac{(n-1)s^2}{\chi^2_{\alpha/2, n-1}}, \frac{(n-1)s^2}{\chi^2_{1-\alpha/2, n-1}}\right]$$

```python
def ic_varianza(datos, nivel_confianza=0.95):
    """IC para la varianza poblacional"""
    n = len(datos)
    s2 = np.var(datos, ddof=1)
    chi2_inf = stats.chi2.ppf((1 - nivel_confianza) / 2, df=n-1)
    chi2_sup = stats.chi2.ppf(1 - (1 - nivel_confianza) / 2, df=n-1)
    
    li = (n-1) * s2 / chi2_sup
    ls = (n-1) * s2 / chi2_inf
    return li, ls, s2

# Evaluación de variabilidad en lead time
lead_times = stats.norm.rvs(loc=7, scale=2, size=25)
li_var, ls_var, s2 = ic_varianza(lead_times)

print("=== IC PARA VARIANZA ===")
print(f"Varianza muestral: {s2:.2f}")
print(f"IC 95% varianza: [{li_var:.2f}, {ls_var:.2f}]")
print(f"IC 95% desv. est.: [{np.sqrt(li_var):.2f}, {np.sqrt(ls_var):.2f}]")
```

---

## 12.7 IC para la Diferencia de Medias

### Muestras Independientes

$$IC: (\bar{x}_1 - \bar{x}_2) \pm t_{\alpha/2, \nu} \cdot \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}$$

```python
def ic_diferencia_medias(datos1, datos2, nivel_confianza=0.95):
    """IC para la diferencia de dos medias independientes"""
    n1, n2 = len(datos1), len(datos2)
    media1, media2 = np.mean(datos1), np.mean(datos2)
    s1, s2 = np.std(datos1, ddof=1), np.std(datos2, ddof=1)
    
    # Welch t-test (varianzas desiguales)
    se = np.sqrt(s1**2/n1 + s2**2/n2)
    df_numer = (s1**2/n1 + s2**2/n2)**2
    df_denom = (s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1)
    df = df_numer / df_denom
    t = stats.t.ppf(1 - (1 - nivel_confianza)/2, df=df)
    
    dif = media1 - media2
    margen = t * se
    return dif - margen, dif + margen, dif, margen

# Comparación: ventas online vs tienda física
np.random.seed(42)
online = stats.norm.rvs(loc=120, scale=30, size=100)
tienda = stats.norm.rvs(loc=100, scale=25, size=80)

li_dif, ls_dif, dif, margen_dif = ic_diferencia_medias(online, tienda)

print("=== IC PARA DIFERENCIA DE MEDIAS ===")
print(f"Media Online: ${np.mean(online):.1f}")
print(f"Media Tienda: ${np.mean(tienda):.1f}")
print(f"Diferencia: ${dif:.1f}")
print(f"IC 95%: [${li_dif:.1f}, ${ls_dif:.1f}]")
print(f"Margen: ±${margen_dif:.1f}")
```

---

## 12.8 IC Bootstrap (Método No Paramétrico)

Cuando no podemos asumir normalidad:

```python
def ic_bootstrap(datos, estadistico=np.mean, n_bootstrap=10000, nivel_confianza=0.95):
    """IC mediante bootstrap (percentil)"""
    estimaciones = []
    n = len(datos)
    
    for _ in range(n_bootstrap):
        muestra_boot = np.random.choice(datos, n, replace=True)
        estimaciones.append(estadistico(muestra_boot))
    
    alpha = 1 - nivel_confianza
    li = np.percentile(estimaciones, alpha/2 * 100)
    ls = np.percentile(estimaciones, (1 - alpha/2) * 100)
    
    return li, ls, np.mean(estimaciones), np.std(estimaciones)

# Datos asimétricos (precios de productos)
np.random.seed(42)
precios = stats.gamma.rvs(a=2, scale=50, size=100) + 10

li_boot, ls_boot, media_boot, se_boot = ic_bootstrap(precios, np.mean)
li_t, ls_t, media_t, margen_t, _ = ic_media_t(precios)

print("=== IC BOOTSTRAP (NO PARAMÉTRICO) ===")
print(f"Media: ${media_boot:.1f}")
print(f"IC Bootstrap 95%: [${li_boot:.1f}, ${ls_boot:.1f}]")
print(f"IC t-Student 95%: [${li_t:.1f}, ${ls_t:.1f}]")
print(f"\nLos IC son similares cuando n es grande")
print(f"Bootstrap no asume normalidad (útil para datos asimétricos)")
```

---

## 12.9 Tamaño Muestral para IC

### Para la Media

$$n = \left(\frac{z_{\alpha/2} \cdot \sigma}{ME}\right)^2$$

### Para Proporciones

$$n = \frac{z_{\alpha/2}^2 \cdot p(1-p)}{ME^2}$$

```python
def tamano_muestral_media(margen_error, sigma, nivel_confianza=0.95):
    """Tamaño muestral necesario para IC de media"""
    z = stats.norm.ppf(1 - (1 - nivel_confianza) / 2)
    n = (z * sigma / margen_error) ** 2
    return np.ceil(n)

def tamano_muestral_proporcion(margen_error, p_estimado=0.5, nivel_confianza=0.95):
    """Tamaño muestral para IC de proporción. p=0.5 da max n"""
    z = stats.norm.ppf(1 - (1 - nivel_confianza) / 2)
    n = (z**2 * p_estimado * (1 - p_estimado)) / margen_error**2
    return np.ceil(n)

print("=== CÁLCULO DE TAMAÑO MUESTRAL ===")
print("\nPara media (σ estimada = 2000, margen ±500):")
print(f"n = {tamano_muestral_media(500, 2000):.0f}")

print("\nPara proporción (margen ±3%):")
for p in [0.1, 0.3, 0.5]:
    print(f"  p_est={p:.0%}: n = {tamano_muestral_proporcion(0.03, p):.0f}")

print("\nPara proporción sin estimación (p=0.5 = máximo):")
for margen in [0.01, 0.02, 0.03, 0.05]:
    n = tamano_muestral_proporcion(margen, 0.5)
    print(f"  Margen ±{margen:.0%}: n = {n:.0f}")
```

---

## 12.10 Ejemplo en Ventas: IC del Ticket Promedio

```python
np.random.seed(42)

tickets = stats.gamma.rvs(a=3, scale=30, size=200) + 20
li, ls, media, margen, std = ic_media_t(tickets)

print("=== IC DEL TICKET PROMEDIO DE VENTA ===")
print(f"Ticket promedio: ${media:.2f}")
print(f"IC 95%: [${li:.2f}, ${ls:.2f}]")
print(f"Margen: ±${margen:.2f}")
print(f"\nConclusión: Con 95% de confianza, el ticket promedio")
print(f"está entre ${li:.2f} y ${ls:.2f}")

# IC para la mediana (bootstrap)
li_med, ls_med, med_boot, se_med = ic_bootstrap(tickets, np.median)
print(f"\nMediana bootstrap 95%: [${li_med:.2f}, ${ls_med:.2f}]")
print(f"Mediana muestral: ${np.median(tickets):.2f}")
```

---

## 12.11 Ejemplo en Compras: IC del Lead Time

```python
# Lead time de 3 proveedores
np.random.seed(42)
proveedores = {
    'A': stats.expon.rvs(scale=5, size=30) + 1,
    'B': stats.expon.rvs(scale=3, size=25) + 1,
    'C': stats.expon.rvs(scale=7, size=20) + 1
}

print("=== IC DEL LEAD TIME POR PROVEEDOR ===")
print(f"{'Prov':6s} {'Media':>8s} {'IC 95%':>25s} {'Amplitud':>10s}")
print("-" * 50)
for prov, datos in proveedores.items():
    li, ls, media, margen, _ = ic_media_t(datos)
    print(f"{prov:6s} {media:>8.2f} [{li:>8.2f}, {ls:>8.2f}] {(ls-li):>10.2f}")

# Diferencia entre proveedores
print(f"\nDiferencia Proveedor A - B:")
li_d, ls_d, dif, _ = ic_diferencia_medias(proveedores['A'], proveedores['B'])
print(f"IC 95%: [{li_d:.2f}, {ls_d:.2f}]")
if li_d > 0 or ls_d < 0:
    print("→ Diferencia significativa")
else:
    print("→ No hay diferencia significativa")
```

---

## 12.12 Ejemplo en Inventarios: IC del Stock de Seguridad

```python
# Determinación del stock de seguridad con IC
np.random.seed(42)

demanda_diaria = stats.poisson.rvs(mu=50, size=90)
li_dem, ls_dem, media_dem, margen_dem, _ = ic_media_t(demanda_diaria)

print("=== IC PARA STOCK DE SEGURIDAD ===")
print(f"Demanda media diaria: {media_dem:.1f}")
print(f"IC 95% demanda: [{li_dem:.1f}, {ls_dem:.1f}]")

lead_time = 5  # días
z = 1.645

# Stock de seguridad basado en límite superior
ss_conservador = z * np.std(demanda_diaria, ddof=1) * np.sqrt(lead_time)
ss_optimista = z * np.std(demanda_diaria, ddof=1) * np.sqrt(lead_time) * 0.8

print(f"\nStock de seguridad (conservador): {ss_conservador:.0f} unidades")
print(f"Stock de seguridad (optimista): {ss_optimista:.0f} unidades")
print(f"Punto reorden (conservador): {media_dem * lead_time + ss_conservador:.0f}")
```

---

## 12.13 Resumen

| Parámetro | IC (95%) | Fórmula |
|-----------|----------|---------|
| Media (σ conocida) | $\bar{x} \pm 1.96 \cdot \sigma/\sqrt{n}$ | Z |
| Media (σ desc.) | $\bar{x} \pm t_{0.025,n-1} \cdot s/\sqrt{n}$ | t |
| Proporción | $\hat{p} \pm 1.96 \cdot \sqrt{\hat{p}(1-\hat{p})/n}$ | Z |
| Varianza | $\left[\frac{(n-1)s^2}{\chi^2_{sup}}, \frac{(n-1)s^2}{\chi^2_{inf}}\right]$ | $\chi^2$ |
| Diferencia medias | $(\bar{x}_1 - \bar{x}_2) \pm t \cdot SE$ | t |

---

## Ejercicios Propuestos

1. **Ventas**: Con una muestra de 80 tickets (media=$85, s=$23), calcula IC 95% para el ticket promedio
2. **Compras**: Compara los lead times de 2 proveedores con IC de la diferencia. ¿Son significativamente diferentes?
3. **Inventarios**: Calcula el IC para la varianza de la demanda. ¿Cómo afecta esto al stock de seguridad?

---

[← Anterior](11-estimacion-puntual.md) | [Índice](index.md) | [Siguiente →](13-pruebas-hipotesis.md)
