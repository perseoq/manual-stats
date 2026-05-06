# Capítulo 14: Pruebas Paramétricas

[← Anterior](13-pruebas-hipotesis.md) | [Índice](index.md) | [Siguiente →](15-pruebas-no-parametricas.md)

---

## 14.1 Introducción

Las **pruebas paramétricas** asumen que los datos provienen de una distribución conocida (típicamente Normal) y hacen inferencias sobre sus parámetros (media, varianza).

### Supuestos Comunes
1. **Normalidad**: Los datos siguen distribución Normal
2. **Independencia**: Las observaciones son independientes
3. **Homogeneidad de varianzas**: Varianzas similares entre grupos (para comparaciones)

---

## 14.2 Prueba t de una Muestra

Compara la media de una muestra contra un valor de referencia.

$$t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}} \sim t_{n-1}$$

```python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

def t_unamuestra(datos, mu0, alternativa='bilateral'):
    """Prueba t de una muestra"""
    n = len(datos)
    media = np.mean(datos)
    s = np.std(datos, ddof=1)
    se = s / np.sqrt(n)
    t_stat = (media - mu0) / se
    gl = n - 1
    
    if alternativa == 'bilateral':
        p_valor = 2 * (1 - stats.t.cdf(abs(t_stat), gl))
    elif alternativa == 'mayor':
        p_valor = 1 - stats.t.cdf(t_stat, gl)
    else:  # menor
        p_valor = stats.t.cdf(t_stat, gl)
    
    # IC 95%
    t_crit = stats.t.ppf(0.975, gl)
    ic = (media - t_crit * se, media + t_crit * se)
    
    return t_stat, p_valor, gl, media, ic

# Ejemplo: ¿El gasto promedio es diferente de $500?
gastos = stats.norm.rvs(loc=520, scale=80, size=40)
t, p, gl, media, ic = t_unamuestra(gastos, 500)

print("=== PRUEBA t DE UNA MUESTRA ===")
print(f"Gasto promedio: ${media:.2f}")
print(f"H₀: μ = $500")
print(f"t({gl}) = {t:.3f}")
print(f"p-valor = {p:.4f}")
print(f"IC 95%: [${ic[0]:.2f}, ${ic[1]:.2f}]")
print(f"Conclusión: {'Rechazamos H₀' if p < 0.05 else 'No rechazamos H₀'}")
```

---

## 14.3 Prueba t de Dos Muestras Independientes

Compara las medias de dos grupos independientes.

$$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$

```python
def t_dosmuestras(datos1, datos2, varianzas_iguales=True):
    """Prueba t para dos muestras independientes"""
    n1, n2 = len(datos1), len(datos2)
    media1, media2 = np.mean(datos1), np.mean(datos2)
    s1, s2 = np.std(datos1, ddof=1), np.std(datos2, ddof=1)
    
    if varianzas_iguales:
        # t de Student (varianzas iguales)
        s_pooled = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1 + n2 - 2))
        se = s_pooled * np.sqrt(1/n1 + 1/n2)
        gl = n1 + n2 - 2
    else:
        # t de Welch (varianzas desiguales)
        se = np.sqrt(s1**2/n1 + s2**2/n2)
        gl_num = (s1**2/n1 + s2**2/n2)**2
        gl_den = (s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1)
        gl = gl_num / gl_den
    
    t_stat = (media1 - media2) / se
    p_valor = 2 * (1 - stats.t.cdf(abs(t_stat), gl))
    
    return t_stat, p_valor, gl, media1, media2

# Ejemplo: Ventas Online vs Tienda Física
np.random.seed(42)
online = stats.norm.rvs(loc=250, scale=60, size=100)
tienda = stats.norm.rvs(loc=220, scale=50, size=80)

t, p, gl, m1, m2 = t_dosmuestras(online, tienda)
t_std, p_std = stats.ttest_ind(online, tienda)

print("=== PRUEBA t DE DOS MUESTRAS INDEPENDIENTES ===")
print(f"Online: media=${m1:.1f}, n={len(online)}")
print(f"Tienda: media=${m2:.1f}, n={len(tienda)}")
print(f"Diferencia: ${m1 - m2:.1f}")
print(f"t({gl:.0f}) = {t:.3f}")
print(f"p-valor = {p:.4f}")
print(f"SciPy: t={t_std:.3f}, p={p_std:.4f}")
```

---

## 14.4 Prueba t Pareada (Muestras Relacionadas)

Compara mediciones **antes/después** en los mismos sujetos.

$$t = \frac{\bar{d}}{s_d / \sqrt{n}}, \quad d_i = x_{i,después} - x_{i,antes}$$

```python
def t_pareada(antes, despues):
    """Prueba t pareada (antes/después)"""
    diferencias = np.array(despues) - np.array(antes)
    n = len(diferencias)
    media_d = np.mean(diferencias)
    s_d = np.std(diferencias, ddof=1)
    se = s_d / np.sqrt(n)
    t_stat = media_d / se
    gl = n - 1
    p_valor = 2 * (1 - stats.t.cdf(abs(t_stat), gl))
    
    return t_stat, p_valor, gl, media_d, diferencias

# Ejemplo: Efecto de capacitación en ventas
np.random.seed(42)
ventas_antes = stats.norm.rvs(loc=100, scale=15, size=30)
ventas_despues = ventas_antes + stats.norm.rvs(loc=8, scale=5, size=30)  # mejora ≈8

t, p, gl, mejora, difs = t_pareada(ventas_antes, ventas_despues)

print("=== PRUEBA t PAREADA - CAPACITACIÓN ===")
print(f"Antes: media={np.mean(ventas_antes):.1f}")
print(f"Después: media={np.mean(ventas_despues):.1f}")
print(f"Mejora media: {mejora:.2f}")
print(f"t({gl}) = {t:.3f}")
print(f"p-valor = {p:.6f}")
print(f"¿La capacitación fue efectiva? {'SÍ' if p < 0.05 else 'NO'}")
```

---

## 14.5 ANOVA de un Factor

Compara las medias de **tres o más grupos** simultáneamente.

### Hipótesis
- H₀: $\mu_1 = \mu_2 = ... = \mu_k$ (todas las medias son iguales)
- H₁: Al menos una media es diferente

### Descomposición de la Varianza

$$SS_{total} = SS_{entre} + SS_{dentro}$$
$$F = \frac{MS_{entre}}{MS_{dentro}} \sim F_{k-1, n-k}$$

```python
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols

np.random.seed(42)

# Ventas por región (3 regiones)
norte = stats.norm.rvs(loc=120, scale=20, size=30)
sur = stats.norm.rvs(loc=110, scale=20, size=28)
centro = stats.norm.rvs(loc=130, scale=20, size=25)

# ANOVA con SciPy
f_stat, p_valor = f_oneway(norte, sur, centro)

print("=== ANOVA DE UN FACTOR ===")
print(f"Región Norte: media={np.mean(norte):.1f}, n={len(norte)}")
print(f"Región Sur: media={np.mean(sur):.1f}, n={len(sur)}")
print(f"Región Centro: media={np.mean(centro):.1f}, n={len(centro)}")
print(f"F = {f_stat:.3f}")
print(f"p-valor = {p_valor:.4f}")
print(f"¿Diferencias significativas entre regiones? {'SÍ' if p_valor < 0.05 else 'NO'}")

# ANOVA con StatsModels (más detallado)
df_ventas = pd.DataFrame({
    'ventas': np.concatenate([norte, sur, centro]),
    'region': ['Norte']*30 + ['Sur']*28 + ['Centro']*25
})

modelo = ols('ventas ~ C(region)', data=df_ventas).fit()
tabla_anova = sm.stats.anova_lm(modelo, typ=2)
print(f"\nTabla ANOVA (StatsModels):")
print(tabla_anova)
```

---

## 14.6 Comparaciones Post-Hoc

Si el ANOVA es significativo, ¿qué grupos difieren?

```python
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Tukey HSD
tukey = pairwise_tukeyhsd(df_ventas['ventas'], df_ventas['region'], alpha=0.05)
print("=== TUKEY HSD - COMPARACIONES POST-HOC ===")
print(tukey)

# Visualización
fig = tukey.plot_simultaneous()
plt.title('Tukey HSD - Diferencias entre Regiones', fontweight='bold')
plt.savefig('tukey_hsd.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 14.7 Prueba Z para Proporciones

Compara una proporción observada contra un valor de referencia.

$$z = \frac{\hat{p} - p_0}{\sqrt{p_0(1-p_0)/n}}$$

```python
def z_proporcion(x, n, p0, alternativa='bilateral'):
    """Prueba Z para una proporción"""
    p_hat = x / n
    se = np.sqrt(p0 * (1-p0) / n)
    z_stat = (p_hat - p0) / se
    
    if alternativa == 'bilateral':
        p_valor = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    elif alternativa == 'mayor':
        p_valor = 1 - stats.norm.cdf(z_stat)
    else:
        p_valor = stats.norm.cdf(z_stat)
    
    return z_stat, p_valor, p_hat

# Tasa de conversión histórica: 25%. Nueva campaña: 65/200 = 32.5%
z, p, p_hat = z_proporcion(65, 200, 0.25, alternativa='mayor')

print("=== PRUEBA Z PARA PROPORCIÓN ===")
print(f"Tasa histórica: 25%")
print(f"Tasa observada: {p_hat:.1%}")
print(f"z = {z:.3f}")
print(f"p-valor = {p:.4f}")
print(f"¿La campaña mejoró la conversión? {'SÍ (p<0.05)' if p < 0.05 else 'NO (p≥0.05)'}")
```

---

## 14.8 Comparación de Dos Proporciones

$$z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1-\hat{p})(1/n_1 + 1/n_2)}}$$

donde $\hat{p} = \frac{x_1 + x_2}{n_1 + n_2}$

```python
def z_dos_proporciones(x1, n1, x2, n2):
    """Prueba Z para dos proporciones independientes"""
    p1, p2 = x1/n1, x2/n2
    p_pooled = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pooled * (1-p_pooled) * (1/n1 + 1/n2))
    z_stat = (p1 - p2) / se
    p_valor = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    return z_stat, p_valor, p1, p2

# Conversión online vs presencial
z, p, p_online, p_tienda = z_dos_proporciones(80, 320, 45, 250)

print("=== COMPARACIÓN DE DOS PROPORCIONES ===")
print(f"Online: {80}/320 = {p_online:.1%}")
print(f"Tienda: {45}/250 = {p_tienda:.1%}")
print(f"z = {z:.3f}")
print(f"p-valor = {p:.4f}")
print(f"¿Hay diferencia significativa? {'SÍ' if p < 0.05 else 'NO'}")
```

---

## 14.9 Supuestos y Diagnósticos

### Verificación de Normalidad

```python
def diagnosticar_normalidad(datos, nombre="Datos"):
    """Diagnóstico de normalidad"""
    # Shapiro-Wilk
    stat_sw, p_sw = stats.shapiro(datos)
    
    # Kolmogorov-Smirnov (comparación con Normal)
    stat_ks, p_ks = stats.kstest(datos, 'norm', args=(np.mean(datos), np.std(datos, ddof=1)))
    
    # Asimetría y curtosis
    skew = stats.skew(datos)
    kurt = stats.kurtosis(datos)
    
    print(f"=== DIAGNÓSTICO DE NORMALIDAD - {nombre} ===")
    print(f"Shapiro-Wilk: W={stat_sw:.4f}, p={p_sw:.4f}")
    print(f"KS: stat={stat_ks:.4f}, p={p_ks:.4f}")
    print(f"Asimetría: {skew:.3f} {'✓' if abs(skew) < 1 else '⚠'}")
    print(f"Curtosis: {kurt:.3f} {'✓' if abs(kurt) < 2 else '⚠'}")
    print(f"Conclusión: {'Aprox. Normal' if p_sw > 0.05 else 'NO Normal'}")

datos_normales = stats.norm.rvs(loc=100, scale=15, size=100)
datos_exponenciales = stats.expon.rvs(scale=1, size=100)

diagnosticar_normalidad(datos_normales, "Normal")
print()
diagnosticar_normalidad(datos_exponenciales, "Exponencial")
```

### Verificación de Homogeneidad de Varianzas

```python
from scipy.stats import levene, bartlett, fligner

def diagnosticar_varianzas(*grupos):
    """Pruebas de homogeneidad de varianzas"""
    nombres = [f'Grupo{i+1}' for i in range(len(grupos))]
    
    stat_l, p_l = levene(*grupos)
    stat_b, p_b = bartlett(*grupos)
    stat_f, p_f = fligner(*grupos)
    
    print("=== HOMOGENEIDAD DE VARIANZAS ===")
    for i, (g, n) in enumerate(zip(grupos, nombres)):
        print(f"{n}: media={np.mean(g):.2f}, var={np.var(g, ddof=1):.2f}")
    print(f"\nLevene: stat={stat_l:.3f}, p={p_l:.4f}")
    print(f"Bartlett: stat={stat_b:.3f}, p={p_b:.4f}")
    print(f"Fligner: stat={stat_f:.3f}, p={p_f:.4f}")
    print(f"¿Varianzas homogéneas? {'SÍ' if p_l > 0.05 else 'NO'}")

grupo1 = stats.norm.rvs(loc=100, scale=15, size=30)
grupo2 = stats.norm.rvs(loc=110, scale=15, size=30)
grupo3 = stats.norm.rvs(loc=105, scale=30, size=30)  # Varianza mayor

diagnosticar_varianzas(grupo1, grupo2, grupo3)
```

---

## 14.10 Alternativas cuando los Supuestos Fallan

| Supuesto violado | Alternativa |
|-----------------|-------------|
| Normalidad | Prueba no paramétrica (Mann-Whitney, Kruskal-Wallis) |
| Varianzas desiguales | t de Welch (varianzas desiguales) |
| Datos pareados pero no normales | Wilcoxon signed-rank |
| Múltiples grupos, no normales | Kruskal-Wallis + Dunn post-hoc |

---

## 14.11 Ejemplo Integrador: Análisis Completo

```python
# Análisis completo: ¿El tipo de descuento afecta las ventas?
np.random.seed(42)

# 4 tipos de descuento
sin_desc = stats.norm.rvs(loc=100, scale=20, size=40)
desc_10 = stats.norm.rvs(loc=115, scale=22, size=38)
desc_20 = stats.norm.rvs(loc=125, scale=25, size=35)
desc_30 = stats.norm.rvs(loc=118, scale=35, size=30)

print("=== ANÁLISIS COMPLETO: EFECTO DEL DESCUENTO ===")
print(f"{'Descuento':12s} {'Media':>8s} {'n':>5s} {'Std':>8s}")
print("-" * 35)
for nombre, datos in [('Sin desc.', sin_desc), ('10%', desc_10), ('20%', desc_20), ('30%', desc_30)]:
    print(f"{nombre:12s} {np.mean(datos):>8.1f} {len(datos):>5d} {np.std(datos, ddof=1):>8.1f}")

# 1. ¿Varianzas homogéneas?
_, p_var = stats.levene(sin_desc, desc_10, desc_20, desc_30)
print(f"\n1. Homogeneidad varianzas (Levene): p = {p_var:.4f}")

# 2. ANOVA
f_stat, p_anova = stats.f_oneway(sin_desc, desc_10, desc_20, desc_30)
print(f"2. ANOVA: F = {f_stat:.3f}, p = {p_anova:.4f}")

# 3. Post-hoc si ANOVA significativo
if p_anova < 0.05:
    df_desc = pd.DataFrame({
        'ventas': np.concatenate([sin_desc, desc_10, desc_20, desc_30]),
        'descuento': ['Sin desc.']*40 + ['10%']*38 + ['20%']*35 + ['30%']*30
    })
    tukey = pairwise_tukeyhsd(df_desc['ventas'], df_desc['descuento'], alpha=0.05)
    print(f"3. Tukey HSD:")
    print(tukey)
```

---

## 14.12 Resumen

| Prueba | Uso | Estadístico |
|--------|-----|-------------|
| **t una muestra** | Media vs valor referencia | t |
| **t independiente** | Dos grupos independientes | t |
| **t pareada** | Antes/Después | t |
| **ANOVA** | Tres o más grupos | F |
| **Z proporción** | Proporción vs referencia | Z |
| **Z dos proporciones** | Comparar dos proporciones | Z |

---

## Ejercicios Propuestos

1. **Ventas**: Usa t pareada para evaluar si una capacitación mejoró las ventas de 15 vendedores
2. **Compras**: Con ANOVA, determina si 3 proveedores tienen diferentes lead times. Haz post-hoc Tukey
3. **Inventarios**: Prueba Z para verificar si el nivel de servicio supera el 97% en 180 días de operación

---

[← Anterior](13-pruebas-hipotesis.md) | [Índice](index.md) | [Siguiente →](15-pruebas-no-parametricas.md)
