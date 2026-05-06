# Capítulo 34: Distribuciones Muestrales

[← Anterior](33-analisis-varianza.md) | [Índice](index.md) | [Siguiente →](35-analisis-supervivencia.md)

---

## 34.1 Introducción

Una **distribución muestral** es la distribución de un estadístico (media, proporción, varianza) calculado a partir de todas las muestras posibles de tamaño n extraídas de una población.

### Concepto Fundamental

| Concepto | Distribución | Parámetros |
|----------|-------------|------------|
| **Población** | $X \sim (\mu, \sigma^2)$ | $\mu, \sigma^2$ |
| **Media muestral** | $\bar{X} \sim (\mu, \frac{\sigma^2}{n})$ | $\mu_{\bar{x}} = \mu$, $\sigma^2_{\bar{x}} = \sigma^2/n$ |
| **Proporción muestral** | $\hat{p} \sim (p, \frac{p(1-p)}{n})$ | $\mu_{\hat{p}} = p$, $\sigma^2_{\hat{p}} = p(1-p)/n$ |

---

## 34.2 Distribución de la Media Muestral

Por el **Teorema del Límite Central**: $\bar{X} \approx N(\mu, \sigma/\sqrt{n})$

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

np.random.seed(42)

def demostrar_dist_media_muestral(dist_poblacion, nombre, n_muestras=10000, 
                                    tamanos=[5, 10, 30, 100]):
    """Demuestra la distribución de la media muestral para diferentes n"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    
    mu_pob = dist_poblacion.mean()
    sigma_pob = dist_poblacion.std()
    
    for i, n in enumerate(tamanos):
        # Generar n_muestras de tamaño n
        muestras = dist_poblacion.rvs(size=(n_muestras, n))
        medias = muestras.mean(axis=1)
        
        # Histograma
        axes[i].hist(medias, bins=50, density=True, alpha=0.7, 
                     color='steelblue', edgecolor='white')
        
        # Normal teórica
        x = np.linspace(medias.min(), medias.max(), 100)
        axes[i].plot(x, stats.norm.pdf(x, mu_pob, sigma_pob/np.sqrt(n)),
                    'r-', linewidth=2, label='Normal teórica')
        
        axes[i].set_title(f'n = {n}', fontweight='bold')
        axes[i].set_xlabel('Media muestral')
        axes[i].set_ylabel('Densidad')
        axes[i].legend()
    
    fig.suptitle(f'Distribución de la Media Muestral - Población: {nombre}', 
                 fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'dist_media_{nombre.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()

# Demostración con distribución exponencial (muy asimétrica)
print("=== DISTRIBUCIÓN DE LA MEDIA MUESTRAL ===")
print("Población: Exponencial(λ=0.5) - MUY asimétrica")
demostrar_dist_media_muestral(stats.expon(scale=2), "Exponencial(λ=0.5)")

# Verificar propiedades
poblacion_exp = stats.expon(scale=2)
mu_pob = poblacion_exp.mean()
sigma_pob = poblacion_exp.std()

print(f"\nPropiedades de la población:")
print(f"  μ = {mu_pob:.2f}")
print(f"  σ = {sigma_pob:.2f}")

for n in [5, 10, 30, 100]:
    muestras = poblacion_exp.rvs(size=(10000, n))
    medias = medias = muestras.mean(axis=1)
    print(f"\nn={n:3d}:")
    print(f"  Media de medias = {medias.mean():.2f} (esperado: {mu_pob:.2f})")
    print(f"  SE empírico = {medias.std():.2f} (teórico: {sigma_pob/np.sqrt(n):.2f})")
```

---

## 34.3 Distribución de la Proporción Muestral

$$\hat{p} \sim N\left(p, \sqrt{\frac{p(1-p)}{n}}\right)$$

```python
def demostrar_dist_proporcion(p_real=0.3, n_muestras=10000, tamanos=[20, 50, 100, 500]):
    """Demuestra la distribución de la proporción muestral"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    
    for i, n in enumerate(tamanos):
        muestras = np.random.binomial(n, p_real, n_muestras)
        proporciones = muestras / n
        
        axes[i].hist(proporciones, bins=30, density=True, alpha=0.7,
                     color='forestgreen', edgecolor='white')
        
        # Normal aproximada
        x = np.linspace(proporciones.min(), proporciones.max(), 100)
        se = np.sqrt(p_real * (1-p_real) / n)
        axes[i].plot(x, stats.norm.pdf(x, p_real, se), 'r-', linewidth=2, label='Normal aprox.')
        
        axes[i].set_title(f'n = {n}', fontweight='bold')
        axes[i].set_xlabel('Proporción muestral')
        axes[i].set_ylabel('Densidad')
        axes[i].legend()
    
    fig.suptitle(f'Distribución de la Proporción Muestral (p={p_real})', 
                 fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig('dist_proporcion.png', dpi=150, bbox_inches='tight')
    plt.show()

print("=== DISTRIBUCIÓN DE LA PROPORCIÓN MUESTRAL ===")
demostrar_dist_proporcion(p_real=0.3)
```

---

## 34.4 Distribución de la Varianza Muestral

$$\frac{(n-1)s^2}{\sigma^2} \sim \chi^2_{n-1}$$

```python
def demostrar_dist_varianza(n_muestras=10000, n_muestra=30):
    """Demuestra la distribución de la varianza muestral"""
    sigma_real = 15
    varianzas = np.zeros(n_muestras)
    
    for i in range(n_muestras):
        muestra = np.random.normal(100, sigma_real, n_muestra)
        varianzas[i] = np.var(muestra, ddof=1)
    
    # Chi-cuadrado transformada
    chi2_obs = (n_muestra - 1) * varianzas / sigma_real**2
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Varianzas
    axes[0].hist(varianzas, bins=40, density=True, alpha=0.7, 
                 color='coral', edgecolor='white')
    axes[0].axvline(sigma_real**2, color='red', linestyle='--', linewidth=2, label=f'σ²={sigma_real**2}')
    axes[0].set_title('Distribución de la Varianza Muestral (s²)', fontweight='bold')
    axes[0].set_xlabel('s²')
    axes[0].legend()
    
    # Chi-cuadrado
    axes[1].hist(chi2_obs, bins=40, density=True, alpha=0.7,
                 color='steelblue', edgecolor='white')
    x = np.linspace(chi2_obs.min(), chi2_obs.max(), 100)
    axes[1].plot(x, stats.chi2.pdf(x, n_muestra-1), 'r-', linewidth=2, 
                label=f'χ²({n_muestra-1})')
    axes[1].set_title('Transformación Chi-Cuadrado', fontweight='bold')
    axes[1].set_xlabel('(n-1)s²/σ²')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('dist_varianza.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("=== DISTRIBUCIÓN DE LA VARIANZA MUESTRAL ===")
    print(f"Media de s²: {varianzas.mean():.1f} (esperado: {sigma_real**2})")
    print(f"Media de χ²: {chi2_obs.mean():.1f} (esperado: {n_muestra-1})")

demostrar_dist_varianza(n_muestra=30)
```

---

## 34.5 Distribución t de Student vs Normal

```python
def comparar_t_normal(gl=[2, 5, 30]):
    """Compara la distribución t con la Normal estándar"""
    x = np.linspace(-4, 4, 200)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, stats.norm.pdf(x), 'k-', linewidth=2, label='N(0,1)')
    
    for df in gl:
        ax.plot(x, stats.t.pdf(x, df), '--', linewidth=1.5, label=f't({df})')
    
    ax.set_xlabel('Valor', fontweight='bold')
    ax.set_ylabel('Densidad', fontweight='bold')
    ax.set_title('Distribución t-Student vs Normal Estándar', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.savefig('t_vs_normal.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("=== t-STUDENT vs NORMAL ===")
    print(f"{'gl':>5s} {'P95':>8s} {'P97.5':>8s} {'P99':>8s}")
    print("-" * 30)
    for df in [2, 5, 10, 30, 100]:
        t = stats.t(df)
        print(f"{df:>5d} {t.ppf(0.95):>8.3f} {t.ppf(0.975):>8.3f} {t.ppf(0.99):>8.3f}")
    print(f"{'N(0,1)':>5s} {stats.norm.ppf(0.95):>8.3f} {stats.norm.ppf(0.975):>8.3f} {stats.norm.ppf(0.99):>8.3f}")
    
    print("\n→ Con más gl, t se aproxima a Normal")
    print("→ Con pocos gl, t tiene colas más pesadas")

comparar_t_normal()
```

---

## 34.6 Distribución F de Fisher

```python
def demostrar_dist_f():
    """Demuestra la distribución F (cociente de varianzas)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(0, 5, 200)
    
    for d1, d2 in [(3, 5), (5, 10), (10, 20), (20, 30)]:
        ax.plot(x, stats.f.pdf(x, d1, d2), linewidth=1.5, label=f'F({d1},{d2})')
    
    ax.set_title('Distribución F de Fisher', fontweight='bold')
    ax.set_xlabel('Valor F')
    ax.set_ylabel('Densidad')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.savefig('dist_f.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("=== DISTRIBUCIÓN F ===")
    for d1, d2 in [(3, 5), (5, 10), (10, 20)]:
        f_dist = stats.f(d1, d2)
        print(f"F({d1},{d2}): Media={f_dist.mean():.2f}, P95={f_dist.ppf(0.95):.2f}, P99={f_dist.ppf(0.99):.2f}")

demostrar_dist_f()
```

---

## 34.7 Ejemplo en Ventas: Precisión de la Media

```python
# ¿Qué tan precisa es la media de ventas estimada?
np.random.seed(42)

# Población de ventas
ventas_poblacion = stats.gamma.rvs(a=3, scale=2000, size=50000)
mu_real = ventas_poblacion.mean()
sigma_real = ventas_poblacion.std()

print("=== PRECISIÓN DE LA MEDIA MUESTRAL (VENTAS) ===")
print(f"Media poblacional real: ${mu_real:,.0f}")
print(f"Sigma poblacional: ${sigma_real:,.0f}")

for n in [10, 30, 50, 100, 500]:
    se_teorico = sigma_real / np.sqrt(n)
    margen_95 = 1.96 * se_teorico
    print(f"\nn={n:3d}:")
    print(f"  SE = ${se_teorico:,.0f}")
    print(f"  Margen error 95% = ±${margen_95:,.0f}")
    print(f"  IC 95% esperado = [${mu_real - margen_95:,.0f}, ${mu_real + margen_95:,.0f}]")
```

---

## 34.8 Ejemplo en Compras: Proporción de Defectos

```python
# Distribución muestral de la tasa de defectos
p_defectos = 0.04  # 4% de defectos

print("=== DISTRIBUCIÓN MUESTRAL DE DEFECTOS ===")
for n_lote in [50, 100, 200, 500]:
    se = np.sqrt(p_defectos * (1-p_defectos) / n_lote)
    margen = 1.96 * se
    print(f"n={n_lote:3d}: SE={se:.2%}, Margen 95%=±{margen:.2%}")
```

---

## 34.9 Ejemplo en Inventarios: Error de Pronóstico

```python
# Distribución del error de pronóstico
np.random.seed(42)

# Error de pronóstico (~ Normal(0, 200))
errores_pronostico = stats.norm.rvs(loc=0, scale=200, size=10000)

print("=== DISTRIBUCIÓN DEL ERROR DE PRONÓSTICO ===")
print(f"Media: {errores_pronostico.mean():.1f}")
print(f"Std: {errores_pronostico.std():.1f}")
print(f"Error MAPE: ${np.abs(errores_pronostico).mean():.1f}")
print(f"P(Error > $500) = {(np.abs(errores_pronostico) > 500).mean():.2%}")

# Error de pronóstico promedio (n períodos)
for n_periodos in [1, 7, 30]:
    medio = errores_pronostico[:len(errores_pronostico)//n_periodos * n_periodos]
    medio = medio.reshape(-1, n_periodos).mean(axis=1)
    print(f"\nError promedio ({n_periodos} períodos):")
    print(f"  Media: {medio.mean():.1f}")
    print(f"  SE: {medio.std():.1f}")
```

---

## 34.10 Resumen

| Estadístico | Distribución | Media | Error Estándar |
|------------|-------------|-------|---------------|
| **Media** $\bar{x}$ | Normal (TLC) | $\mu$ | $\sigma/\sqrt{n}$ |
| **Proporción** $\hat{p}$ | Normal (aprox.) | $p$ | $\sqrt{p(1-p)/n}$ |
| **Varianza** $s^2$ | $\chi^2_{n-1}$ | $\sigma^2$ | $\sigma^2\sqrt{2/(n-1)}$ |
| **Diferencia** $\bar{x}_1-\bar{x}_2$ | Normal/t | $\mu_1-\mu_2$ | $\sqrt{\sigma_1^2/n_1 + \sigma_2^2/n_2}$ |

---

## Ejercicios Propuestos

1. **Ventas**: Para una población con μ=500 y σ=100, ¿cuál es la probabilidad de que una muestra de n=25 tenga media > 520?
2. **Compras**: Si la tasa real de defectos es 3%, ¿cuál es la probabilidad de que en una muestra de 200 haya más del 5%?
3. **Inventarios**: Demuestra con simulación que la media de 30 observaciones de demanda Poisson(λ=20) se distribuye aproximadamente Normal

---

[← Anterior](33-analisis-varianza.md) | [Índice](index.md) | [Siguiente →](35-analisis-supervivencia.md)
