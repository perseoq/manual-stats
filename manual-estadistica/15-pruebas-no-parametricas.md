# Capítulo 15: Pruebas No Paramétricas

[← Anterior](14-pruebas-parametricas.md) | [Índice](index.md) | [Siguiente →](16-chi-cuadrado.md)

---

## 15.1 Introducción

Las **pruebas no paramétricas** (o de distribución libre) no requieren supuestos sobre la distribución de los datos. Son ideales cuando:

1. Los datos no siguen una distribución Normal
2. El tamaño muestral es pequeño
3. Los datos son ordinales o de rango
4. Hay outliers extremos

### Ventajas y Desventajas

| Ventajas | Desventajas |
|----------|-------------|
| No requieren normalidad | Menos potencia (si se cumplen supuestos paramétricos) |
| Robustas a outliers | No usan toda la información numérica |
| Funcionan con datos ordinales | Más difíciles de interpretar |
| Válidas con muestras pequeñas | Menos sensibles a diferencias sutiles |

### Equivalencia: Paramétrica ⇔ No Paramétrica

| Paramétrica | No Paramétrica | Tipo de Dato |
|-------------|---------------|--------------|
| t pareada | Wilcoxon signed-rank | Dos mediciones pareadas |
| t independiente | Mann-Whitney U | Dos grupos independientes |
| ANOVA | Kruskal-Wallis | Tres+ grupos independientes |
| ANOVA medidas repetidas | Friedman | Tres+ mediciones pareadas |
| Correlación Pearson | Correlación Spearman | Asociación entre variables |

---

## 15.2 Prueba de Mann-Whitney U (Wilcoxon Rank-Sum)

Alternativa no paramétrica a la **t de dos muestras independientes**. Compara si dos grupos provienen de la misma distribución.

### Estadístico U

$$U = \sum_{i=1}^{n_1} \sum_{j=1}^{n_2} [x_i > y_j]$$

Donde $[x_i > y_j]$ es 1 si el valor del grupo 1 supera al del grupo 2, y 0 en caso contrario. Intuitivamente, U cuenta cuántas veces un valor del primer grupo es mayor que uno del segundo. Si las distribuciones son idénticas, el valor esperado es:

$$E[U] = \frac{n_1 n_2}{2}$$

Para muestras grandes ($n_1, n_2 > 20$), la distribución de U se aproxima a una Normal:

$$Z = \frac{U - \frac{n_1 n_2}{2}}{\sqrt{\frac{n_1 n_2 (n_1 + n_2 + 1)}{12}}} \sim N(0,1)$$

```python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

def mann_whitney_detallado(datos1, datos2, nombre1="Grupo 1", nombre2="Grupo 2"):
    """Prueba de Mann-Whitney U con detalles"""
    # Prueba
    u_stat, p_valor = stats.mannwhitneyu(datos1, datos2, alternative='two-sided')
    
    # Medianas
    med1, med2 = np.median(datos1), np.median(datos2)
    
    print(f"=== MANN-WHITNEY U: {nombre1} vs {nombre2} ===")
    print(f"{'':12s} {'n':>5s} {'Mediana':>10s} {'Media':>10s}")
    print("-" * 38)
    print(f"{nombre1:12s} {len(datos1):5d} {med1:10.2f} {np.mean(datos1):10.2f}")
    print(f"{nombre2:12s} {len(datos2):5d} {med2:10.2f} {np.mean(datos2):10.2f}")
    print(f"\nU = {u_stat:.1f}")
    print(f"p-valor = {p_valor:.4f}")
    print(f"¿Diferencia significativa? {'SÍ' if p_valor < 0.05 else 'NO'}")
    
    return u_stat, p_valor

# Ejemplo: Ventas con distribución no normal (exponencial)
ventas_a = stats.expon.rvs(scale=100, size=25) + 20  # skew positiva
ventas_b = stats.expon.rvs(scale=130, size=30) + 20

# Primero verificamos normalidad
_, p_norm_a = stats.shapiro(ventas_a)
_, p_norm_b = stats.shapiro(ventas_b)

print("=== VERIFICACIÓN DE NORMALIDAD ===")
print(f"Grupo A: Shapiro p = {p_norm_a:.4f} {'(Normal)' if p_norm_a > 0.05 else '(NO Normal)'}")
print(f"Grupo B: Shapiro p = {p_norm_b:.4f} {'(Normal)' if p_norm_b > 0.05 else '(NO Normal)'}")
print("→ Datos NO normales → Usamos Mann-Whitney")

# Aplicamos Mann-Whitney
print()
mann_whitney_detallado(ventas_a, ventas_b, "Producto A", "Producto B")
```

---

## 15.3 Prueba de Wilcoxon Signed-Rank

Alternativa no paramétrica a la **t pareada**. Compara dos mediciones relacionadas.

### Estadístico W

$$W = \sum_{i=1}^{n} \text{sign}(d_i) \cdot R(|d_i|)$$

Donde $d_i = y_i - x_i$ son las diferencias entre mediciones pareadas, $R(|d_i|)$ es el rango de la diferencia absoluta, y $\text{sign}(d_i)$ es el signo (+1 o -1). Bajo H₀ (no hay diferencia), el valor esperado de W es cero. Para muestras grandes, se aproxima a una Normal:

$$Z = \frac{W}{\sqrt{n(n+1)(2n+1)/6}} \sim N(0,1)$$

```python
def wilcoxon_detallado(antes, despues, nombre="Prueba"):
    """Prueba de Wilcoxon signed-rank detallada"""
    diferencias = np.array(despues) - np.array(antes)
    w_stat, p_valor = stats.wilcoxon(antes, despues, alternative='two-sided')
    
    med_antes, med_despues = np.median(antes), np.median(despues)
    
    print(f"=== WILCOXON SIGNED-RANK: {nombre} ===")
    print(f"{'':12s} {'n':>5s} {'Mediana':>10s} {'Media':>10s}")
    print("-" * 38)
    print(f"{'Antes':12s} {len(antes):5d} {med_antes:10.2f} {np.mean(antes):10.2f}")
    print(f"{'Después':12s} {len(despues):5d} {med_despues:10.2f} {np.mean(despues):10.2f}")
    print(f"\nDiferencias positivas: {np.sum(diferencias > 0)}")
    print(f"Diferencias negativas: {np.sum(diferencias < 0)}")
    print(f"W = {w_stat:.1f}")
    print(f"p-valor = {p_valor:.4f}")
    print(f"¿Cambio significativo? {'SÍ (p<0.05)' if p_valor < 0.05 else 'NO (p≥0.05)'}")
    
    return w_stat, p_valor

# Ejemplo: Evaluación de satisfacción antes/después (escala Likert)
np.random.seed(42)
satisfaccion_antes = np.random.choice([1, 2, 3, 4, 5], 20, p=[0.1, 0.2, 0.4, 0.2, 0.1])
satisfaccion_despues = satisfaccion_antes + np.random.choice([0, 1, 2], 20, p=[0.3, 0.5, 0.2])
satisfaccion_despues = np.clip(satisfaccion_despues, 1, 5)

print("Datos ordinales (escala Likert 1-5) → Wilcoxon")
wilcoxon_detallado(satisfaccion_antes, satisfaccion_despues, "Satisfacción cliente")
```

---

## 15.4 Prueba de Kruskal-Wallis

Alternativa no paramétrica al **ANOVA de un factor**. Compara tres o más grupos independientes.

### Estadístico H

$$H = \frac{12}{N(N+1)} \sum_{i=1}^{k} \frac{R_i^2}{n_i} - 3(N+1)$$

Donde $R_i$ es la suma de rangos del grupo i, $n_i$ es el tamaño del grupo i, $N = \sum n_i$ es el tamaño total, y $k$ es el número de grupos. Bajo H₀, $H \sim \chi^2_{k-1}$ aproximadamente.

```python
def kruskal_detallado(*grupos, nombres=None):
    """Prueba de Kruskal-Wallis detallada"""
    if nombres is None:
        nombres = [f'Grupo {i+1}' for i in range(len(grupos))]
    
    h_stat, p_valor = stats.kruskal(*grupos)
    
    print("=== KRUSKAL-WALLIS ===")
    for nombre, datos in zip(nombres, grupos):
        print(f"{nombre:15s}: n={len(datos):3d}, Mediana={np.median(datos):8.2f}, "
              f"IQR={stats.iqr(datos):8.2f}")
    print(f"\nH = {h_stat:.3f}")
    print(f"p-valor = {p_valor:.4f}")
    print(f"¿Diferencias entre grupos? {'SÍ' if p_valor < 0.05 else 'NO'}")
    
    # Post-hoc: comparaciones Dunn (si es significativo)
    if p_valor < 0.05 and len(grupos) > 2:
        print("\n→ Se requieren comparaciones post-hoc (Dunn)")
    
    return h_stat, p_valor

# Ejemplo: Ventas por canal (3+ canales, datos no normales)
np.random.seed(42)
canal_a = stats.expon.rvs(scale=50, size=25) + 10
canal_b = stats.expon.rvs(scale=65, size=30) + 10
canal_c = stats.expon.rvs(scale=45, size=28) + 10
canal_d = stats.expon.rvs(scale=70, size=22) + 10

kruskal_detallado(canal_a, canal_b, canal_c, canal_d, 
                   nombres=['Online', 'Tienda', 'Catálogo', 'Mayoreo'])
```

---

## 15.5 Prueba de Friedman

Alternativa no paramétrica al **ANOVA de medidas repetidas**. Compara tres o más mediciones relacionadas.

```python
def friedman_detallado(*mediciones, nombres=None):
    """Prueba de Friedman para medidas repetidas"""
    if nombres is None:
        nombres = [f'Medición {i+1}' for i in range(len(mediciones))]
    
    # Friedman requiere que todos tengan el mismo tamaño
    q_stat, p_valor = stats.friedmanchisquare(*mediciones)
    
    print("=== FRIEDMAN (MEDIDAS REPETIDAS) ===")
    for nombre, datos in zip(nombres, mediciones):
        print(f"{nombre:15s}: Mediana={np.median(datos):8.2f}, IQR={stats.iqr(datos):8.2f}")
    print(f"\nQ = {q_stat:.3f}")
    print(f"p-valor = {p_valor:.4f}")
    print(f"¿Diferencias entre mediciones? {'SÍ' if p_valor < 0.05 else 'NO'}")
    
    return q_stat, p_valor

# Ejemplo: Productividad de vendedores en 4 trimestres
np.random.seed(42)
n_vendedores = 15
trim1 = stats.norm.rvs(loc=100, scale=20, size=n_vendedores)
trim2 = trim1 + np.random.normal(5, 10, n_vendedores)
trim3 = trim1 + np.random.normal(8, 12, n_vendedores)
trim4 = trim1 + np.random.normal(3, 15, n_vendedores)

friedman_detallado(trim1, trim2, trim3, trim4, 
                   nombres=['Trim1', 'Trim2', 'Trim3', 'Trim4'])
```

---

## 15.6 Prueba de Kolmogorov-Smirnov (KS)

Compara si dos muestras provienen de la **misma distribución** (no solo igualdad de medias).

$$D = \sup_x |F_1(x) - F_2(x)|$$

```python
def ks_detallado(datos1, datos2, nombre1="Muestra 1", nombre2="Muestra 2"):
    """Prueba KS de dos muestras"""
    d_stat, p_valor = stats.ks_2samp(datos1, datos2)
    
    print("=== KOLMOGOROV-SMIRNOV (DOS MUESTRAS) ===")
    print(f"D = {d_stat:.4f}")
    print(f"p-valor = {p_valor:.4f}")
    print(f"¿Misma distribución? {'SÍ (p≥0.05)' if p_valor >= 0.05 else 'NO (p<0.05)'}")

# Ejemplo: ¿Dos proveedores tienen la misma distribución de lead times?
np.random.seed(42)
lead_a = stats.expon.rvs(scale=5, size=30) + 2
lead_b = stats.gamma.rvs(a=3, scale=2, size=25) + 2  # Distribución diferente

ks_detallado(lead_a, lead_b, "Proveedor A", "Proveedor B")
```

---

## 15.7 Comparación de Eficiencia: Paramétrica vs No Paramétrica

```python
# Simulación: datos normales vs no normales
np.random.seed(42)

def comparar_pruebas(datos1, datos2, parametrica=True):
    """Compara resultados de prueba paramétrica y no paramétrica"""
    if parametrica:
        stat, p = stats.ttest_ind(datos1, datos2)
        nombre = "t-test"
    else:
        stat, p = stats.mannwhitneyu(datos1, datos2)
        nombre = "Mann-Whitney"
    return p

# Escenario 1: Datos normales
norm_a = stats.norm.rvs(loc=100, scale=15, size=30)
norm_b = stats.norm.rvs(loc=108, scale=15, size=30)

p_t = comparar_pruebas(norm_a, norm_b, True)
p_mw = comparar_pruebas(norm_a, norm_b, False)

# Escenario 2: Datos exponenciales (no normales)
exp_a = stats.expon.rvs(scale=50, size=30) + 10
exp_b = stats.expon.rvs(scale=65, size=30) + 10

p_t_exp = comparar_pruebas(exp_a, exp_b, True)
p_mw_exp = comparar_pruebas(exp_a, exp_b, False)

print("=== EFICIENCIA: PARAMÉTRICA vs NO PARAMÉTRICA ===")
print(f"{'Escenario':20s} {'t-test p':>10s} {'M-W p':>10s} {'¿Coinciden?':>12s}")
print("-" * 52)
print(f"{'Normales':20s} {p_t:>10.4f} {p_mw:>10.4f} {'SÍ' if (p_t<0.05)==(p_mw<0.05) else 'NO':>12s}")
print(f"{'Exponenciales':20s} {p_t_exp:>10.4f} {p_mw_exp:>10.4f} {'SÍ' if (p_t_exp<0.05)==(p_mw_exp<0.05) else 'NO':>12s}")

print("\n⚠ Con datos normales, t-test es más potente (p menor)")
print("⚠ Con datos no normales, Mann-Whitney es más confiable")
```

---

## 15.8 Ejemplo en Ventas: Segmentación de Clientes

```python
# Comparar gasto entre segmentos de clientes (datos muy asimétricos)
np.random.seed(42)

# Gasto mensual por segmento (distribuciones muy asimétricas)
premium = stats.lognorm.rvs(s=0.8, scale=500, size=40)
estandar = stats.lognorm.rvs(s=0.6, scale=300, size=50)
economico = stats.lognorm.rvs(s=0.5, scale=150, size=60)

print("=== SEGMENTACIÓN DE CLIENTES (No paramétrica) ===")
print(f"{'Segmento':12s} {'n':>5s} {'Mediana':>10s} {'Media':>10s}")
print("-" * 38)
for nombre, datos in [('Premium', premium), ('Estándar', estandar), ('Económico', economico)]:
    print(f"{nombre:12s} {len(datos):5d} {np.median(datos):10.2f} {np.mean(datos):10.2f}")

# Kruskal-Wallis
h_stat, p_valor = stats.kruskal(premium, estandar, economico)
print(f"\nKruskal-Wallis: H = {h_stat:.3f}, p = {p_valor:.4f}")

# Comparaciones pairwise con Bonferroni
from scipy.stats import mannwhitneyu
parejas = [('Premium', 'Estándar'), ('Premium', 'Económico'), ('Estándar', 'Económico')]
datos_dict = {'Premium': premium, 'Estándar': estandar, 'Económico': economico}

print("\nComparaciones pairwise (Mann-Whitney con Bonferroni):")
alpha_adj = 0.05 / len(parejas)
for g1, g2 in parejas:
    _, p = mannwhitneyu(datos_dict[g1], datos_dict[g2])
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"  {g1:12s} vs {g2:12s}: p={p:.4f} {sig}")
```

---

## 15.9 Ejemplo en Compras: Evaluación de Proveedores

```python
# Evaluación de 3 proveedores (datos ordinales de calidad)
np.random.seed(42)

# Calificación de calidad (1-5, ordinal)
prov_a = np.random.choice([1, 2, 3, 4, 5], 20, p=[0.05, 0.10, 0.30, 0.35, 0.20])
prov_b = np.random.choice([1, 2, 3, 4, 5], 22, p=[0.10, 0.20, 0.35, 0.25, 0.10])
prov_c = np.random.choice([1, 2, 3, 4, 5], 18, p=[0.15, 0.25, 0.35, 0.15, 0.10])

print("=== EVALUACIÓN DE CALIDAD - PROVEEDORES ===")
print(f"{'Proveedor':12s} {'Mediana':>8s} {'Media':>8s}")
print("-" * 30)
for nombre, datos in [('Prov A', prov_a), ('Prov B', prov_b), ('Prov C', prov_c)]:
    print(f"{nombre:12s} {np.median(datos):8.1f} {np.mean(datos):8.2f}")

# Datos ordinales → Kruskal-Wallis
h, p_kw = stats.kruskal(prov_a, prov_b, prov_c)
print(f"\nKruskal-Wallis: H = {h:.3f}, p = {p_kw:.4f}")
```

---

## 15.10 Ejemplo en Inventarios: Rotación por Categoría

```python
# Rotación de inventario (días) por categoría ABC
np.random.seed(42)

cat_a = stats.expon.rvs(scale=5, size=50) + 1   # rápida rotación
cat_b = stats.expon.rvs(scale=15, size=40) + 2  # media rotación
cat_c = stats.expon.rvs(scale=30, size=35) + 3  # lenta rotación

print("=== ROTACIÓN DE INVENTARIO POR CATEGORÍA ===")
print(f"{'Categoría':12s} {'n':>5s} {'Mediana':>10s} {'IQR':>10s}")
print("-" * 38)
for nombre, datos in [('A', cat_a), ('B', cat_b), ('C', cat_c)]:
    print(f"{nombre:12s} {len(datos):5d} {np.median(datos):10.1f} {stats.iqr(datos):10.1f}")

# Kruskal-Wallis (rotación no es normal)
h, p = stats.kruskal(cat_a, cat_b, cat_c)
print(f"\nKruskal-Wallis: H = {h:.3f}, p = {p:.4f}")

# Post-hoc pairwise
print("\nComparaciones por pares:")
for g1, g2, n1, n2 in [('A', 'B', cat_a, cat_b), ('A', 'C', cat_a, cat_c), ('B', 'C', cat_b, cat_c)]:
    u, p_mw = stats.mannwhitneyu(n1, n2)
    print(f"  {g1} vs {g2}: U={u:.0f}, p={p_mw:.4f}")
```

---

## 15.11 Cuándo Usar Cada Prueba

```python
def guia_prueba(tipo_comparacion, tipo_datos, n_muestras, supuestos_cumplidos=False):
    """Guía para seleccionar la prueba estadística adecuada"""
    print("=== GUÍA DE SELECCIÓN DE PRUEBA ESTADÍSTICA ===")
    print(f"Comparación: {tipo_comparacion}")
    print(f"Tipo de datos: {tipo_datos}")
    print(f"Muestras: {n_muestras}")
    
    if tipo_datos == "categóricos":
        print("→ Prueba Chi-cuadrado")
    elif tipo_datos == "numéricos":
        if supuestos_cumplidos:
            if n_muestras == 1:
                print("→ t de una muestra (si compara con referencia)")
            elif tipo_comparacion == "independiente":
                print("→ t de Student (2 grupos) o ANOVA (3+ grupos)")
            else:
                print("→ t pareada (2) o ANOVA medidas repetidas (3+)")
        else:
            if n_muestras == 1:
                print("→ Wilcoxon signed-rank (si es pareada)")
            elif tipo_comparacion == "independiente":
                if n_muestras <= 2:
                    print("→ Mann-Whitney U")
                else:
                    print("→ Kruskal-Wallis")
            else:
                if n_muestras <= 2:
                    print("→ Wilcoxon signed-rank")
                else:
                    print("→ Friedman")
    print()

guia_prueba("independiente", "numéricos", 2, supuestos_cumplidos=False)
guia_prueba("independiente", "numéricos", 2, supuestos_cumplidos=True)
guia_prueba("pareada", "numéricos", 2, supuestos_cumplidos=False)
guia_prueba("independiente", "numéricos", 3, supuestos_cumplidos=False)
```

---

## 15.12 Resumen

| Prueba No Paramétrica | Equivalente Paramétrica | Uso |
|----------------------|------------------------|-----|
| Mann-Whitney U | t independiente | 2 grupos independientes |
| Wilcoxon signed-rank | t pareada | 2 mediciones pareadas |
| Kruskal-Wallis | ANOVA un factor | 3+ grupos independientes |
| Friedman | ANOVA medidas repetidas | 3+ mediciones pareadas |
| KS test | — | Igualdad de distribuciones |
| Spearman | Pearson | Correlación no lineal |

---

## Ejercicios Propuestos

1. **Ventas**: Usa Mann-Whitney para comparar ventas de dos equipos (datos no normales: exponenciales)
2. **Compras**: Con Kruskal-Wallis, compara 3 proveedores en calidad de servicio (escala 1-5)
3. **Inventarios**: Aplica Wilcoxon signed-rank para evaluar si un nuevo sistema redujo los días de inventario

---

[← Anterior](14-pruebas-parametricas.md) | [Índice](index.md) | [Siguiente →](16-chi-cuadrado.md)
