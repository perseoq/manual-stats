# Capítulo 13: Pruebas de Hipótesis

[← Anterior](12-intervalos-confianza.md) | [Índice](index.md) | [Siguiente →](14-pruebas-parametricas.md)

---

## 13.1 Marco Conceptual

Las **pruebas de hipótesis** son procedimientos estadísticos para tomar decisiones basadas en datos. Permiten determinar si la evidencia es suficiente para rechazar una afirmación sobre la población.

### Elementos Clave

| Elemento | Definición | Ejemplo |
|----------|-----------|---------|
| **H₀ (Nula)** | Afirmación actual/status quo | "El tiempo medio de entrega es 5 días" |
| **H₁ (Alternativa)** | Lo que queremos demostrar | "El tiempo medio es ≠ 5 días" |
| **Estadístico de prueba** | Valor calculado de los datos | t = 2.34 |
| **p-valor** | Probabilidad de observar datos tan extremos si H₀ es cierta | p = 0.023 |
| **α (alfa)** | Nivel de significancia (error tipo I que toleramos) | α = 0.05 |

### Tipos de Hipótesis Alternativas

| Tipo | H₀ | H₁ | ¿Cuándo usarlo? |
|------|-----|-----|-----------------|
| **Bilateral** | μ = μ₀ | μ ≠ μ₀ | "El proceso cambió" |
| **Unilateral derecha** | μ ≤ μ₀ | μ > μ₀ | "Las ventas mejoraron" |
| **Unilateral izquierda** | μ ≥ μ₀ | μ < μ₀ | "Los costos se redujeron" |

---

## 13.2 Errores Tipo I y Tipo II

| Decisión | H₀ Verdadera | H₀ Falsa |
|----------|-------------|----------|
| **No rechazar H₀** | ✓ Correcto (1-α) | ✗ Error Tipo II (β) |
| **Rechazar H₀** | ✗ Error Tipo I (α) | ✓ Potencia (1-β) |

### Consecuencias en Negocios

| Error | Consecuencia |
|-------|-------------|
| **Tipo I (α)** | Invertir en un cambio que no mejora nada (falso positivo) |
| **Tipo II (β)** | Dejar pasar una oportunidad de mejora (falso negativo) |

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Visualización de errores tipo I y II
def visualizar_errores(mu0=100, mu1=105, sigma=15, alpha=0.05, n=30):
    """Visualiza α y β en una prueba de hipótesis"""
    se = sigma / np.sqrt(n)
    z_critico = stats.norm.ppf(1 - alpha/2)  # Bilateral
    limite_inf = mu0 - z_critico * se
    limite_sup = mu0 + z_critico * se
    
    x = np.linspace(mu0 - 4*se, mu1 + 4*se, 200)
    
    # Distribución bajo H₀
    y0 = stats.norm.pdf(x, mu0, se)
    
    # Distribución bajo H₁ (potencia)
    y1 = stats.norm.pdf(x, mu1, se)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, y0, 'b-', linewidth=2, label='H₀: μ = 100')
    ax.plot(x, y1, 'r-', linewidth=2, label='H₁: μ = 105')
    ax.axvline(limite_inf, color='gray', linestyle='--', alpha=0.7)
    ax.axvline(limite_sup, color='gray', linestyle='--', alpha=0.7, label='Límites críticos')
    
    # Área α (Error Tipo I)
    x_alpha = x[x > limite_sup]
    ax.fill_between(x_alpha, 0, stats.norm.pdf(x_alpha, mu0, se), 
                     color='red', alpha=0.3, label=f'α = {alpha}')
    
    # Área β (Error Tipo II)
    x_beta = x[(x >= limite_inf) & (x <= limite_sup)]
    ax.fill_between(x_beta, 0, stats.norm.pdf(x_beta, mu1, se), 
                     color='orange', alpha=0.3, label='β')
    
    ax.set_xlabel('Media muestral', fontweight='bold')
    ax.set_ylabel('Densidad', fontweight='bold')
    ax.set_title('Errores Tipo I (α) y Tipo II (β)', fontweight='bold')
    ax.legend()
    plt.savefig('errores_tipo1_tipo2.png', dpi=150, bbox_inches='tight')
    plt.show()

visualizar_errores()
```

---

## 13.3 Potencia Estadística (1-β)

La **potencia** es la probabilidad de detectar un efecto real. Depende de:

1. **Tamaño del efecto**: Qué tan grande es la diferencia real
2. **Tamaño muestral (n)**: Más datos = más potencia
3. **α**: Mayor α = mayor potencia (pero más errores tipo I)
4. **Variabilidad (σ)**: Menos variabilidad = más potencia

```python
def calcular_potencia(mu0, mu1, sigma, n, alpha=0.05):
    """Calcula la potencia estadística para una prueba de medias"""
    se = sigma / np.sqrt(n)
    z_critico = stats.norm.ppf(1 - alpha/2)
    limite = mu0 + z_critico * se
    
    # Potencia = P(rechazar H₀ | H₁ es cierta)
    potencia = 1 - stats.norm.cdf(limite, mu1, se) + stats.norm.cdf(-limite, mu1, se)
    # Corrección para unilateral:
    potencia_unilateral = 1 - stats.norm.cdf(mu0 + stats.norm.ppf(1-alpha) * se, mu1, se)
    return potencia, potencia_unilateral

print("=== POTENCIA ESTADÍSTICA ===")
for efecto in [3, 5, 8, 10, 15]:
    pot_bilateral, pot_unilateral = calcular_potencia(100, 100+efecto, 15, 30)
    print(f"Diferencia={efecto:2d}: Potencia bilateral={pot_bilateral:.2%}, "
          f"unilateral={pot_unilateral:.2%}")

print("\nEfecto del tamaño muestral (diferencia=5):")
for n in [10, 20, 30, 50, 100, 500]:
    pot, _ = calcular_potencia(100, 105, 15, n)
    print(f"  n={n:3d}: potencia={pot:.2%}")
```

---

## 13.4 p-valor

El **p-valor** es la probabilidad de obtener un resultado tan extremo o más que el observado, **asumiendo que H₀ es cierta**.

### Interpretación del p-valor

| p-valor | Evidencia contra H₀ |
|---------|---------------------|
| p > 0.10 | Insuficiente |
| 0.05 < p ≤ 0.10 | Marginal/débil |
| 0.01 < p ≤ 0.05 | Moderada |
| 0.001 < p ≤ 0.01 | Fuerte |
| p ≤ 0.001 | Muy fuerte |

### ADVERTENCIA: Malas Interpretaciones Comunes

| ⚠ Incorrecto | ✓ Correcto |
|-------------|-----------|
| "El p-valor es la probabilidad de que H₀ sea cierta" | "El p-valor es P(datos extremos | H₀ cierta)" |
| "p > 0.05 significa que H₀ es verdadera" | "p > 0.05 significa que no hay suficiente evidencia para rechazar H₀" |
| "p = 0.01 significa que H₁ tiene 99% de probabilidad" | "p = 0.01 significa que hay evidencia fuerte contra H₀" |

---

## 13.5 Procedimiento General

1. **Definir H₀ y H₁**
2. **Elegir α** (típicamente 0.05)
3. **Calcular estadístico de prueba**
4. **Calcular p-valor**
5. **Tomar decisión**: Si p-valor < α, rechazar H₀

```python
def prueba_hipotesis_media(datos, mu0, alpha=0.05, alternativa='bilateral'):
    """
    Prueba de hipótesis para la media de una población
    alternativa: 'bilateral', 'mayor', 'menor'
    """
    n = len(datos)
    media = np.mean(datos)
    s = np.std(datos, ddof=1)
    se = s / np.sqrt(n)
    
    # Estadístico t
    t_stat = (media - mu0) / se
    gl = n - 1
    
    # p-valor según alternativa
    if alternativa == 'bilateral':
        p_valor = 2 * (1 - stats.t.cdf(abs(t_stat), gl))
    elif alternativa == 'mayor':
        p_valor = 1 - stats.t.cdf(t_stat, gl)
    elif alternativa == 'menor':
        p_valor = stats.t.cdf(t_stat, gl)
    
    rechazar = p_valor < alpha
    
    return {
        't_stat': t_stat,
        'gl': gl,
        'p_valor': p_valor,
        'alpha': alpha,
        'media_muestral': media,
        'rechazar_H0': rechazar,
        'alternativa': alternativa,
        'conclusion': 'Se rechaza H₀' if rechazar else 'No se rechaza H₀'
    }

# Ejemplo: ¿El ticket promedio es diferente de $100?
np.random.seed(42)
tickets = stats.norm.rvs(loc=105, scale=20, size=50)  # Realmente μ=105

resultado = prueba_hipotesis_media(tickets, mu0=100, alpha=0.05)

print("=== PRUEBA DE HIPÓTESIS ===")
print(f"H₀: μ = 100")
print(f"H₁: μ ≠ 100")
print(f"Media muestral: {resultado['media_muestral']:.2f}")
print(f"t = {resultado['t_stat']:.3f}")
print(f"gl = {resultado['gl']}")
print(f"p-valor = {resultado['p_valor']:.4f}")
print(f"α = {resultado['alpha']}")
print(f"Decisión: {resultado['conclusion']}")
```

---

## 13.6 Relación entre IC y Prueba de Hipótesis

```python
# Un IC del 95% que NO contiene μ₀ es equivalente a rechazar H₀ con α=0.05
np.random.seed(42)

datos = stats.norm.rvs(loc=105, scale=20, size=50)
media = np.mean(datos)
s = np.std(datos, ddof=1)
se = s / np.sqrt(len(datos))
t = stats.t.ppf(0.975, len(datos)-1)

ic_li = media - t * se
ic_ls = media + t * se

print("=== RELACIÓN IC - PRUEBA DE HIPÓTESIS ===")
print(f"H₀: μ = 100")
print(f"IC 95%: [{ic_li:.2f}, {ic_ls:.2f}]")
print(f"¿100 está dentro del IC? {100 >= ic_li and 100 <= ic_ls}")
print(f"Conclusión: {'No rechazar H₀' if (100 >= ic_li and 100 <= ic_ls) else 'Rechazar H₀'}")
```

---

## 13.7 Ejemplo en Ventas: ¿La campaña funcionó?

```python
np.random.seed(42)

# Ventas antes y después de campaña
antes = stats.norm.rvs(loc=50000, scale=8000, size=20)
despues = stats.norm.rvs(loc=55000, scale=8000, size=20)

# Prueba: ¿Las ventas después son mayores?
resultado = prueba_hipotesis_media(despues, mu0=np.mean(antes), 
                                    alpha=0.05, alternativa='mayor')

print("=== EVALUACIÓN DE CAMPAÑA DE VENTAS ===")
print(f"Ventas antes: media={np.mean(antes):,.0f}")
print(f"Ventas después: media={np.mean(despues):,.0f}")
print(f"Diferencia: ${np.mean(despues) - np.mean(antes):,.0f}")
print(f"p-valor: {resultado['p_valor']:.4f}")
print(f"¿La campaña aumentó ventas? {'SÍ' if resultado['rechazar_H0'] else 'NO'}")
```

---

## 13.8 Ejemplo en Compras: ¿Un proveedor es más rápido?

```python
# Lead times de dos proveedores
np.random.seed(42)
prov_a = stats.expon.rvs(scale=5, size=25) + 2  # media ≈ 7
prov_b = stats.expon.rvs(scale=3, size=30) + 2   # media ≈ 5

# Prueba t de dos muestras
t_stat, p_valor = stats.ttest_ind(prov_a, prov_b, alternative='greater')

print("=== COMPARACIÓN DE PROVEEDORES ===")
print(f"Proveedor A: media={np.mean(prov_a):.2f} días")
print(f"Proveedor B: media={np.mean(prov_b):.2f} días")
print(f"t = {t_stat:.3f}")
print(f"p-valor = {p_valor:.4f}")
print(f"¿A es más lento que B? {'SÍ (p < 0.05)' if p_valor < 0.05 else 'NO (p ≥ 0.05)'}")
```

---

## 13.9 Ejemplo en Inventarios: ¿El nivel de servicio es adecuado?

```python
# Prueba de proporción: ¿El nivel de servicio es ≥ 95%?
np.random.seed(42)

dias = 200
stockouts_observados = 6  # 6 días con stockout en 200
exitos = dias - stockouts_observados
nivel_servicio_obs = exitos / dias

# H₀: p ≥ 0.95 (nivel de servicio adecuado)
# H₁: p < 0.95 (nivel de servicio inadecuado)
p0 = 0.95
p_hat = nivel_servicio_obs

# Estadístico Z para proporción
se = np.sqrt(p0 * (1-p0) / dias)
z_stat = (p_hat - p0) / se
p_valor = stats.norm.cdf(z_stat)  # Unilateral izquierda

print("=== PRUEBA DE NIVEL DE SERVICIO ===")
print(f"Días evaluados: {dias}")
print(f"Stockouts: {stockouts_observados}")
print(f"Nivel de servicio observado: {nivel_servicio_obs:.1%}")
print(f"H₀: Nivel de servicio ≥ 95%")
print(f"z = {z_stat:.3f}")
print(f"p-valor = {p_valor:.4f}")
print(f"¿Nivel de servicio inadecuado? {'SÍ' if p_valor < 0.05 else 'NO'}")
```

---

## 13.10 Tamaño del Efecto

El p-valor solo no basta. El **tamaño del efecto** mide la magnitud de la diferencia:

### d de Cohen

$$d = \frac{\bar{x}_1 - \bar{x}_2}{s_{pooled}}$$

```python
def cohens_d(datos1, datos2):
    """Calcula d de Cohen (tamaño del efecto)"""
    n1, n2 = len(datos1), len(datos2)
    media1, media2 = np.mean(datos1), np.mean(datos2)
    s1, s2 = np.std(datos1, ddof=1), np.std(datos2, ddof=1)
    
    s_pooled = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1 + n2 - 2))
    d = (media1 - media2) / s_pooled
    return d

# Interpretación de d de Cohen
def interpretar_d(d):
    if abs(d) < 0.2:
        return "Efecto MUY pequeño (despreciable)"
    elif abs(d) < 0.5:
        return "Efecto PEQUEÑO" 
    elif abs(d) < 0.8:
        return "Efecto MEDIANO"
    else:
        return "Efecto GRANDE"

np.random.seed(42)
grupo_control = stats.norm.rvs(loc=100, scale=15, size=50)
grupo_tratamiento = stats.norm.rvs(loc=108, scale=15, size=50)

d = cohens_d(grupo_tratamiento, grupo_control)
_, p = stats.ttest_ind(grupo_tratamiento, grupo_control)

print("=== TAMAÑO DEL EFECTO ===")
print(f"d de Cohen = {d:.3f}")
print(f"Interpretación: {interpretar_d(d)}")
print(f"p-valor = {p:.4f}")
print(f"NOTA: p-valor significativo NO implica efecto grande")
```

---

## 13.11 Problema de Múltiples Comparaciones

Cuando realizamos muchas pruebas, aumentamos la probabilidad de errores tipo I.

### Corrección de Bonferroni

$$\alpha_{ajustado} = \frac{\alpha}{k}$$

Donde $k$ es el número de comparaciones.

```python
def bonferroni(p_valores, alpha=0.05):
    """Aplica corrección de Bonferroni"""
    k = len(p_valores)
    alpha_adj = alpha / k
    rechazos = [p < alpha_adj for p in p_valores]
    return alpha_adj, rechazos

# Ejemplo: Comparar 5 productos contra un control
np.random.seed(42)
control = stats.norm.rvs(loc=100, scale=15, size=30)
productos = [stats.norm.rvs(loc=100 + np.random.choice([-5, 0, 5, 10]), scale=15, size=30) 
             for _ in range(5)]

p_valores = [stats.ttest_ind(p, control)[1] for p in productos]

print("=== MÚLTIPLES COMPARACIONES ===")
print("Comparaciones sin ajustar:")
for i, p in enumerate(p_valores):
    print(f"  Producto {i+1}: p = {p:.4f} {'*' if p < 0.05 else ''}")

alpha_adj, rechazos = bonferroni(p_valores)
print(f"\nBonferroni: α ajustado = {alpha_adj:.4f}")
for i, (p, r) in enumerate(zip(p_valores, rechazos)):
    print(f"  Producto {i+1}: p = {p:.4f} {'*' if r else ''}")
```

---

## 13.12 Resumen

- Las pruebas de hipótesis ayudan a tomar decisiones basadas en datos
- **Error Tipo I** (α): Falso positivo (rechazar H₀ verdadera)
- **Error Tipo II** (β): Falso negativo (no rechazar H₀ falsa)
- El **p-valor** no es la probabilidad de H₀
- El **tamaño del efecto** es importante además del p-valor
- Las **múltiples comparaciones** requieren ajuste

---

## Ejercicios Propuestos

1. **Ventas**: Prueba si una campaña publicitaria aumentó las ventas (H₁: μ_después > μ_antes) con α=0.05
2. **Compras**: ¿Hay diferencia significativa en lead time entre 2 proveedores? Calcula también d de Cohen
3. **Inventarios**: Prueba si el nivel de servicio real es ≥ 98% (200 días, observa stockouts)

---

[← Anterior](12-intervalos-confianza.md) | [Índice](index.md) | [Siguiente →](14-pruebas-parametricas.md)
