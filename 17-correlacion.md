# Capítulo 17: Correlación

[← Anterior](16-chi-cuadrado.md) | [Índice](index.md) | [Siguiente →](18-regresion-lineal.md)

---

## 17.1 Introducción

La **correlación** mide la **fuerza y dirección** de la relación lineal entre dos variables cuantitativas.

### Tipos de Correlación

| Tipo | Rango | Uso |
|------|-------|-----|
| **Pearson (r)** | [-1, 1] | Relación lineal, datos normales |
| **Spearman (ρ)** | [-1, 1] | Relación monótona, datos no normales |
| **Kendall (τ)** | [-1, 1] | Relación monótona, muestras pequeñas |

### Interpretación de r de Pearson

| Valor | Fuerza |
|-------|--------|
| 0.00 - 0.19 | Muy débil |
| 0.20 - 0.39 | Débil |
| 0.40 - 0.59 | Moderada |
| 0.60 - 0.79 | Fuerte |
| 0.80 - 1.00 | Muy fuerte |

**Signo (+)**: Relación directa (↑ X → ↑ Y)
**Signo (-)**: Relación inversa (↑ X → ↓ Y)

---

## 17.2 Coeficiente de Correlación de Pearson

$$r = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2 \sum_{i=1}^{n} (y_i - \bar{y})^2}}$$

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

np.random.seed(42)

def correlacion_pearson_detallada(x, y, nombre_x="X", nombre_y="Y"):
    """Calcula correlación de Pearson con todos los detalles"""
    n = len(x)
    r, p_valor = stats.pearsonr(x, y)
    
    # Intervalo de confianza (transformación z de Fisher)
    z = np.arctanh(r)
    se_z = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(0.975)
    ic_z = (z - z_crit * se_z, z + z_crit * se_z)
    ic_r = (np.tanh(ic_z[0]), np.tanh(ic_z[1]))
    
    # t-statistic
    t_stat = r * np.sqrt((n-2) / (1-r**2))
    
    print(f"=== CORRELACIÓN DE PEARSON: {nombre_x} vs {nombre_y} ===")
    print(f"r = {r:.4f}")
    print(f"IC 95%: [{ic_r[0]:.4f}, {ic_r[1]:.4f}]")
    print(f"t({n-2}) = {t_stat:.3f}")
    print(f"p-valor = {p_valor:.6f}")
    print(f"n = {n}")
    print(f"Interpretación: ", end="")
    if abs(r) < 0.2: print("Correlación muy débil")
    elif abs(r) < 0.4: print("Correlación débil")
    elif abs(r) < 0.6: print("Correlación moderada")
    elif abs(r) < 0.8: print("Correlación fuerte")
    else: print("Correlación muy fuerte")
    print(f"Dirección: {'Directa (+)' if r > 0 else 'Inversa (-)'}")
    print(f"Significativa: {'SÍ (p<0.05)' if p_valor < 0.05 else 'NO (p≥0.05)'}")
    
    return r, p_valor

# Ejemplo: Relación entre gasto en publicidad y ventas
np.random.seed(42)
n = 50
publicidad = np.random.uniform(10, 100, n)
ventas = 500 + 8 * publicidad + np.random.normal(0, 50, n)

correlacion_pearson_detallada(publicidad, ventas, "Publicidad ($K)", "Ventas ($)")
```

### Matriz de Correlación con Pandas

```python
# Dataset de ventas
df = pd.DataFrame({
    'publicidad': publicidad,
    'ventas': ventas,
    'precio': np.random.uniform(20, 100, n),
    'descuento': np.random.choice([0, 5, 10, 15, 20], n),
    'satisfaccion': np.random.uniform(3, 5, n).round(1)
})

print("=== MATRIZ DE CORRELACIÓN ===")
corr_matrix = df.corr()
print(corr_matrix.round(3))

print("\n=== MATRIZ DE P-VALORES ===")
p_matrix = pd.DataFrame(np.zeros_like(corr_matrix), 
                        index=corr_matrix.index, 
                        columns=corr_matrix.columns)
for i in corr_matrix.index:
    for j in corr_matrix.columns:
        if i != j:
            _, p = stats.pearsonr(df[i], df[j])
            p_matrix.loc[i, j] = p
print(p_matrix.round(4))
```

---

## 17.3 Heatmap de Correlación

```python
fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', 
            center=0, vmin=-1, vmax=1, linewidths=1, ax=ax,
            mask=np.triu(np.ones_like(corr_matrix, dtype=bool)))
ax.set_title('Matriz de Correlación - Variables de Negocio', fontweight='bold')

plt.tight_layout()
plt.savefig('heatmap_correlacion.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 17.4 Correlación de Spearman (Rango)

Alternativa no paramétrica cuando:
- Los datos no son normales
- La relación es monótona pero no lineal
- Hay outliers

$$\rho = 1 - \frac{6\sum d_i^2}{n(n^2-1)}$$

```python
def correlacion_spearman_detallada(x, y, nombre_x="X", nombre_y="Y"):
    """Correlación de Spearman detallada"""
    rho, p_valor = stats.spearmanr(x, y)
    
    print(f"=== CORRELACIÓN DE SPEARMAN: {nombre_x} vs {nombre_y} ===")
    print(f"ρ = {rho:.4f}")
    print(f"p-valor = {p_valor:.4f}")
    print(f"Comparación con Pearson: r = {stats.pearsonr(x, y)[0]:.4f}")
    
    return rho, p_valor

# Datos con relación no lineal (monótona)
np.random.seed(42)
x = np.linspace(1, 10, 100)
y = np.exp(x * 0.5) + np.random.normal(0, 10, 100)  # Relación exponencial

print("Datos con relación EXPONENCIAL (no lineal):")
r, _ = correlacion_pearson_detallada(x, y, "X", "Y (exponencial)")
print()
correlacion_spearman_detallada(x, y, "X", "Y (exponencial)")
print("\n⚠ Spearman captura mejor relaciones monótonas no lineales")
```

---

## 17.5 Correlación de Kendall Tau

$$ \tau = \frac{(\text{pares concordantes}) - (\text{pares discordantes})}{\frac{1}{2}n(n-1)}$$

```python
# Correlación de Kendall
tau, p_valor = stats.kendalltau(x, y)

print("=== CORRELACIÓN DE KENDALL τ ===")
print(f"τ = {tau:.4f}")
print(f"p-valor = {p_valor:.4f}")

# Comparación de los 3 coeficientes
print("\n=== COMPARACIÓN DE COEFICIENTES ===")
print(f"{'Método':12s} {'Coef':>8s} {'p-valor':>8s}")
print("-" * 30)
r_p, p_p = stats.pearsonr(x, y)
rho_s, p_s = stats.spearmanr(x, y)
tau_k, p_k = stats.kendalltau(x, y)
print(f"{'Pearson':12s} {r_p:>8.4f} {p_p:>8.4f}")
print(f"{'Spearman':12s} {rho_s:>8.4f} {p_s:>8.4f}")
print(f"{'Kendall':12s} {tau_k:>8.4f} {p_k:>8.4f}")
```

---

## 17.6 Correlación Parcial

Mide la correlación entre dos variables **controlando** por una tercera.

$$r_{xy|z} = \frac{r_{xy} - r_{xz}r_{yz}}{\sqrt{(1-r_{xz}^2)(1-r_{yz}^2)}}$$

```python
def correlacion_parcial(x, y, z):
    """Correlación parcial controlando por z"""
    r_xy = stats.pearsonr(x, y)[0]
    r_xz = stats.pearsonr(x, z)[0]
    r_yz = stats.pearsonr(y, z)[0]
    
    r_xy_z = (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    
    print("=== CORRELACIÓN PARCIAL ===")
    print(f"Correlación simple r(X,Y) = {r_xy:.4f}")
    print(f"Correlación parcial r(X,Y | Z) = {r_xy_z:.4f}")
    
    return r_xy_z

# Ejemplo: Correlación entre precio y ventas, controlando por publicidad
precio = np.random.normal(50, 10, 100)
publicidad = np.random.normal(100, 20, 100)
ventas = 500 - 3 * precio + 2 * publicidad + np.random.normal(0, 30, 100)

print("Correlación simple:")
r_pv, _ = stats.pearsonr(precio, ventas)
r_pp, _ = stats.pearsonr(precio, publicidad)
r_pubv, _ = stats.pearsonr(publicidad, ventas)
print(f"  r(precio, ventas) = {r_pv:.4f}")
print(f"  r(precio, publicidad) = {r_pp:.4f}")
print(f"  r(publicidad, ventas) = {r_pubv:.4f}")
print()

correlacion_parcial(precio, ventas, publicidad)
print("\n→ Al controlar por publicidad, el efecto negativo del precio es más claro")
```

---

## 17.7 Correlación vs Causalidad

> **CORRELACIÓN NO IMPLICA CAUSALIDAD**

```python
# Demostración de correlaciones espurias
np.random.seed(42)

n = 50
tiempo = np.arange(n)

# Dos variables sin relación causal pero correlacionadas
ventas_helado = 100 + 0.5 * tiempo + np.random.normal(0, 10, n)
ahogados_playa = 50 + 0.3 * tiempo + np.random.normal(0, 8, n)

r_espurio, p_espurio = stats.pearsonr(ventas_helado, ahogados_playa)

print("=== CORRELACIONES ESPURIAS ===")
print(f"Ventas de helado vs Ahogados en playa:")
print(f"r = {r_espurio:.4f}, p = {p_espurio:.4f}")
print(f"\n⚠ AMBAS VARIABLES AUMENTAN CON EL TIEMPO (verano)")
print("⚠ NO hay relación causal directa")
print("⚠ La correlación puede deberse a una tercera variable (temperatura)")
```

---

## 17.8 Ejemplo en Ventas: Correlación Publicidad-Ventas

```python
np.random.seed(42)

# Datos de marketing
marketing_df = pd.DataFrame({
    'tv': np.random.uniform(10, 100, 100),
    'redes': np.random.uniform(5, 80, 100),
    'email': np.random.uniform(1, 50, 100),
    'radio': np.random.uniform(2, 30, 100),
    'ventas': np.zeros(100)
})

# Las ventas dependen principalmente de TV y redes
marketing_df['ventas'] = (2000 + 30 * marketing_df['tv'] + 
                          15 * marketing_df['redes'] + 
                          5 * marketing_df['email'] + 
                          np.random.normal(0, 200, 100))

print("=== CORRELACIÓN: INVERSIÓN PUBLICITARIA vs VENTAS ===")
corr_mkt = marketing_df.corr()['ventas'].sort_values(ascending=False)
print(corr_mkt)

print(f"\nCanal más efectivo: {corr_mkt.index[1]} (r={corr_mkt.values[1]:.3f})")
print(f"Canal menos efectivo: {corr_mkt.index[-1]} (r={corr_mkt.values[-1]:.3f})")
```

---

## 17.9 Ejemplo en Compras: Correlación Costo-Volumen

```python
# Relación entre volumen de compra y precio unitario
np.random.seed(42)
n = 60

volumen = np.random.randint(100, 10000, n)
# Descuentos por volumen: a mayor volumen, menor precio unitario
precio_unitario = 50 - 0.002 * volumen + np.random.normal(0, 5, n)
precio_unitario = np.clip(precio_unitario, 10, 60)

r_vol_precio, p_vp = stats.pearsonr(volumen, precio_unitario)

print("=== CORRELACIÓN: VOLUMEN vs PRECIO UNITARIO ===")
print(f"r = {r_vol_precio:.4f}")
print(f"p = {p_vp:.4f}")
print(f"Interpretación: Correlación {'negativa' if r_vol_precio < 0 else 'positiva'} {'débil' if abs(r_vol_precio) < 0.3 else 'moderada' if abs(r_vol_precio) < 0.6 else 'fuerte'}")

# Ahorro estimado por volumen
if r_vol_precio < -0.5:
    print("\n→ Hay evidencia de descuentos por volumen significativos")
    print("→ Estrategia: consolidar compras para mejor precio")
```

---

## 17.10 Ejemplo en Inventarios: Correlación Stock-Demanda

```python
# Relación entre stock y demanda (puede haber correlación espuria)
np.random.seed(42)
n = 80

stock = np.random.randint(0, 200, n)
demanda_real = np.random.poisson(30, n)
# Stockout afecta ventas: si stock < demanda, se pierden ventas
ventas = np.minimum(stock, demanda_real) + np.random.normal(0, 2, n)

r_stock_ventas, _ = stats.pearsonr(stock, ventas)
r_stock_demanda, _ = stats.pearsonr(stock, demanda_real)
r_demanda_ventas, _ = stats.pearsonr(demanda_real, ventas)

print("=== CORRELACIÓN EN INVENTARIOS ===")
print(f"r(stock, ventas) = {r_stock_ventas:.4f}")
print(f"r(stock, demanda_real) = {r_stock_demanda:.4f}")
print(f"r(demanda_real, ventas) = {r_demanda_ventas:.4f}")
print(f"\n⚠ La correlación stock-ventas puede estar limitada por stockouts")
print("⚠ No siempre más stock significa más ventas")
```

---

## 17.11 Transformación de Fisher (IC para r)

```python
def ic_pearson(r, n, nivel_confianza=0.95):
    """Intervalo de confianza para r de Pearson (Fisher Z)"""
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - (1 - nivel_confianza) / 2)
    
    ic_z = (z - z_crit * se, z + z_crit * se)
    ic_r = (np.tanh(ic_z[0]), np.tanh(ic_z[1]))
    
    return ic_r

# Ejemplo
r_obs = 0.65
n_obs = 50
ic = ic_pearson(r_obs, n_obs)

print("=== IC PARA r DE PEARSON ===")
print(f"r observado: {r_obs}")
print(f"IC 95%: [{ic[0]:.4f}, {ic[1]:.4f}]")
print(f"Anchura: {ic[1]-ic[0]:.4f}")

# Efecto de n
print(f"\nEfecto del tamaño muestral (r = 0.65):")
for n_test in [10, 30, 50, 100, 500]:
    ic_test = ic_pearson(0.65, n_test)
    print(f"  n={n_test:3d}: IC=[{ic_test[0]:.3f}, {ic_test[1]:.3f}], anchura={ic_test[1]-ic_test[0]:.3f}")
```

---

## 17.12 Resumen

- **Pearson (r)**: Relación lineal, datos normales
- **Spearman (ρ)**: Relación monótona, cualquier distribución
- **Kendall (τ)**: Relación monótona, muestras pequeñas, muchos empates
- **Correlación NO es causalidad**
- **Matriz de correlación**: Visión general de relaciones múltiples
- **Correlación parcial**: Aísla el efecto de una tercera variable

---

## Ejercicios Propuestos

1. **Ventas**: Calcula la correlación entre inversión en publicidad y ventas mensuales de 12 meses
2. **Compras**: ¿Hay correlación entre el volumen de pedido y el precio unitario? Calcula Pearson y Spearman
3. **Inventarios**: Analiza la correlación entre el stock de seguridad y el nivel de servicio. ¿Qué coeficiente usas y por qué?

---

[← Anterior](16-chi-cuadrado.md) | [Índice](index.md) | [Siguiente →](18-regresion-lineal.md)
