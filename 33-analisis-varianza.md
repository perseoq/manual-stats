# Capítulo 33: Análisis de Varianza (ANOVA)

[← Anterior](32-muestreo.md) | [Índice](index.md) | [Siguiente →](34-distribuciones-muestrales.md)

---

## 33.1 Introducción

El **Análisis de Varianza (ANOVA)** compara las medias de **tres o más grupos** para determinar si existen diferencias significativas. Mientras que la prueba t compara dos grupos, ANOVA los compara simultáneamente.

### Hipótesis
- **H₀**: $\mu_1 = \mu_2 = ... = \mu_k$ (todas las medias son iguales)
- **H₁**: Al menos una media es diferente

### Partición de la Varianza

$$SS_{Total} = SS_{Entre} + SS_{Dentro}$$

| Fuente | Suma de Cuadrados | gl | Cuadrados Medios | F |
|--------|------------------|----|-----------------|---|
| **Entre grupos** | $SS_{entre}$ | $k-1$ | $MS_{entre} = \frac{SS_{entre}}{k-1}$ | $F = \frac{MS_{entre}}{MS_{dentro}}$ |
| **Dentro grupos** | $SS_{dentro}$ | $n-k$ | $MS_{dentro} = \frac{SS_{dentro}}{n-k}$ | |
| **Total** | $SS_{total}$ | $n-1$ | | |

---

## 33.2 ANOVA de un Factor

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns

np.random.seed(42)

# Ventas por región (4 regiones)
norte = stats.norm.rvs(loc=150, scale=20, size=35)
sur = stats.norm.rvs(loc=135, scale=20, size=30)
centro = stats.norm.rvs(loc=160, scale=20, size=32)
oeste = stats.norm.rvs(loc=140, scale=20, size=28)

# ANOVA con SciPy
f_stat, p_valor = stats.f_oneway(norte, sur, centro, oeste)

print("=== ANOVA DE UN FACTOR: VENTAS POR REGIÓN ===")
print(f"{'Región':10s} {'n':>5s} {'Media':>10s} {'Std':>10s}")
print("-" * 35)
for nombre, datos in [('Norte', norte), ('Sur', sur), ('Centro', centro), ('Oeste', oeste)]:
    print(f"{nombre:10s} {len(datos):5d} {np.mean(datos):10.1f} {np.std(datos, ddof=1):10.1f}")

print(f"\nF({3}, {35+30+32+28-4}) = {f_stat:.3f}")
print(f"p-valor = {p_valor:.4f}")
print(f"¿Diferencias significativas? {'SÍ (p<0.05)' if p_valor < 0.05 else 'NO (p≥0.05)'}")
```

### ANOVA con StatsModels (Tabla completa)

```python
# Preparar datos en formato largo
df_ventas = pd.DataFrame({
    'ventas': np.concatenate([norte, sur, centro, oeste]),
    'region': ['Norte']*35 + ['Sur']*30 + ['Centro']*32 + ['Oeste']*28
})

# Modelo ANOVA
modelo = ols('ventas ~ C(region)', data=df_ventas).fit()
tabla_anova = sm.stats.anova_lm(modelo, typ=2)

print("\n=== TABLA ANOVA COMPLETA ===")
print(tabla_anova)

# Eta-cuadrado (tamaño del efecto)
ss_entre = tabla_anova['sum_sq']['C(region)']
ss_total = tabla_anova['sum_sq'].sum()
eta_cuadrado = ss_entre / ss_total
print(f"\nη² (Eta-cuadrado) = {eta_cuadrado:.4f}")
print(f"El factor 'región' explica el {eta_cuadrado*100:.1f}% de la variabilidad en ventas")
```

---

## 33.3 Supuestos del ANOVA

```python
def verificar_supuestos_anova(df, var_dep, var_ind):
    """Verifica supuestos de normalidad y homogeneidad de varianzas"""
    grupos = [df[df[var_ind] == g][var_dep] for g in df[var_ind].unique()]
    
    # 1. Normalidad (Shapiro-Wilk por grupo)
    print("=== VERIFICACIÓN DE SUPUESTOS ===")
    print("\n1. Normalidad (Shapiro-Wilk):")
    for g in df[var_ind].unique():
        datos = df[df[var_ind] == g][var_dep]
        _, p = stats.shapiro(datos)
        estado = '✓' if p > 0.05 else '✗'
        print(f"  {g}: W={_:.4f}, p={p:.4f} {estado}")
    
    # 2. Homogeneidad de varianzas (Levene)
    stat, p = stats.levene(*grupos)
    print(f"\n2. Homogeneidad de varianzas:")
    print(f"  Levene: stat={stat:.3f}, p={p:.4f}")
    print(f"  {'✓ Varianzas homogéneas' if p > 0.05 else '✗ Varianzas NO homogéneas'}")
    
    # 3. Independencia (asumida por diseño)
    print(f"\n3. Independencia: Asumida por diseño muestral")
    
    return p > 0.05 and all(stats.shapiro(df[df[var_ind] == g][var_dep])[1] > 0.05 
                           for g in df[var_ind].unique())

verificar_supuestos_anova(df_ventas, 'ventas', 'region')
```

---

## 33.4 Comparaciones Post-Hoc (Tukey HSD)

```python
print("=== COMPARACIONES POST-HOC (TUKEY HSD) ===")
tukey = pairwise_tukeyhsd(df_ventas['ventas'], df_ventas['region'], alpha=0.05)
print(tukey)

# Resumen de diferencias significativas
print("\nDiferencias significativas:")
tukey_df = pd.DataFrame(data=tukey.summary().data[1:], 
                         columns=tukey.summary().data[0])
significantes = tukey_df[tukey_df['reject'] == True]
if len(significantes) > 0:
    for _, row in significantes.iterrows():
        print(f"  {row['group1']} - {row['group2']}: diff={row['meandiff']:.1f}, "
              f"p={row['p-adj']:.4f}")
else:
    print("  Ninguna diferencia significativa entre pares")
```

---

## 33.5 ANOVA de Dos Factores

```python
# ANOVA de dos factores: región × canal de venta
np.random.seed(42)

n_por_celda = 15
disenos = []

for region in ['Norte', 'Sur', 'Centro']:
    for canal in ['Online', 'Tienda']:
        if region == 'Norte' and canal == 'Online':
            ventas = stats.norm.rvs(loc=200, scale=25, size=n_por_celda)
        elif region == 'Norte' and canal == 'Tienda':
            ventas = stats.norm.rvs(loc=180, scale=25, size=n_por_celda)
        elif region == 'Sur':
            ventas = stats.norm.rvs(loc=160, scale=25, size=n_por_celda)
        else:
            ventas = stats.norm.rvs(loc=190, scale=25, size=n_por_celda)
        
        for v in ventas:
            disenos.append({'ventas': v, 'region': region, 'canal': canal})

df_2factores = pd.DataFrame(disenos)

# ANOVA de dos factores
modelo2 = ols('ventas ~ C(region) + C(canal) + C(region):C(canal)', 
              data=df_2factores).fit()
tabla_anova2 = sm.stats.anova_lm(modelo2, typ=2)

print("=== ANOVA DE DOS FACTORES ===")
print(tabla_anova2)

# Interpretación
for factor in ['C(region)', 'C(canal)', 'C(region):C(canal)']:
    p = tabla_anova2.loc[factor, 'PR(>F)']
    nombre = factor.replace('C(', '').replace(')', '')
    print(f"\n{nombre}: p={p:.4f} → {'Significativo' if p < 0.05 else 'No significativo'}")
```

---

## 33.6 ANOVA de Medidas Repetidas

```python
# Ventas de los mismos vendedores en 3 trimestres
np.random.seed(42)
n_vendedores = 20

trim1 = stats.norm.rvs(loc=100, scale=15, size=n_vendedores)
trim2 = trim1 + stats.norm.rvs(loc=8, scale=5, size=n_vendedores)
trim3 = trim1 + stats.norm.rvs(loc=12, scale=5, size=n_vendedores)

# ANOVA de medidas repetidas
from statsmodels.stats.anova import AnovaRM

df_medidas = pd.DataFrame({
    'vendedor': np.tile(range(n_vendedores), 3),
    'trimestre': ['Trim1']*n_vendedores + ['Trim2']*n_vendedores + ['Trim3']*n_vendedores,
    'ventas': np.concatenate([trim1, trim2, trim3])
})

modelo_rm = AnovaRM(df_medidas, 'ventas', 'vendedor', within=['trimestre']).fit()
print("=== ANOVA DE MEDIDAS REPETIDAS ===")
print(modelo_rm.anova_table)
```

---

## 33.7 Ejemplo en Ventas: Efecto del Descuento

```python
# ¿El nivel de descuento afecta las ventas?
np.random.seed(42)

descuentos = {
    '0%': stats.norm.rvs(loc=100, scale=20, size=30),
    '10%': stats.norm.rvs(loc=120, scale=22, size=28),
    '20%': stats.norm.rvs(loc=135, scale=25, size=25),
    '30%': stats.norm.rvs(loc=125, scale=30, size=22)
}

f, p = stats.f_oneway(*descuentos.values())

print("=== EFECTO DEL DESCUENTO EN VENTAS ===")
for nivel, datos in descuentos.items():
    print(f"  Descuento {nivel}: Media={np.mean(datos):.1f}, n={len(datos)}")

print(f"\nANOVA: F={f:.3f}, p={p:.4f}")
print(f"¿El descuento afecta ventas? {'SÍ' if p < 0.05 else 'NO'}")

# Post-hoc
df_desc = pd.DataFrame({
    'ventas': np.concatenate(list(descuentos.values())),
    'descuento': np.concatenate([[k]*len(v) for k, v in descuentos.items()])
})
tukey_desc = pairwise_tukeyhsd(df_desc['ventas'], df_desc['descuento'])
print(f"\nTukey HSD - Pares significativos:")
print(tukey_desc)
```

---

## 33.8 Ejemplo en Compras: Proveedores y Calidad

```python
# Comparación de calidad entre proveedores
np.random.seed(42)

proveedores_data = {
    'Prov_A': stats.norm.rvs(loc=4.5, scale=0.3, size=25),
    'Prov_B': stats.norm.rvs(loc=4.0, scale=0.5, size=22),
    'Prov_C': stats.norm.rvs(loc=4.2, scale=0.4, size=20),
    'Prov_D': stats.norm.rvs(loc=3.5, scale=0.6, size=18)
}

f_prov, p_prov = stats.f_oneway(*proveedores_data.values())

print("=== CALIDAD POR PROVEEDOR ===")
for prov, datos in proveedores_data.items():
    print(f"  {prov}: Media={np.mean(datos):.2f}/5, n={len(datos)}")

print(f"\nANOVA: F={f_prov:.3f}, p={p_prov:.4f}")
```

---

## 33.9 MANOVA (ANOVA Multivariante)

```python
from statsmodels.multivariate.manova import MANOVA

# Múltiples variables dependientes
np.random.seed(42)
n = 30

datos_manova = pd.DataFrame({
    'grupo': ['Control']*n + ['Tratamiento']*n,
    'ventas': np.concatenate([
        stats.norm.rvs(loc=100, scale=15, size=n),
        stats.norm.rvs(loc=120, scale=15, size=n)
    ]),
    'satisfaccion': np.concatenate([
        stats.norm.rvs(loc=3.5, scale=0.5, size=n),
        stats.norm.rvs(loc=4.2, scale=0.5, size=n)
    ]),
    'lealtad': np.concatenate([
        stats.norm.rvs(loc=60, scale=10, size=n),
        stats.norm.rvs(loc=75, scale=10, size=n)
    ])
})

maova = MANOVA.from_formula('ventas + satisfaccion + lealtad ~ grupo', data=datos_manova)
print("=== MANOVA ===")
print(maova.mv_test())
```

---

## 33.10 Resumen

| Tipo | Predictores | Uso |
|------|------------|-----|
| **Unifactorial** | 1 categórico | Comparar k grupos |
| **Bifactorial** | 2 categóricos | Efecto de 2 factores + interacción |
| **Medidas Repetidas** | 1 factor intra-sujeto | Mismos sujetos, diferentes condiciones |
| **MANOVA** | 1+ categóricos, 2+ continuos | Múltiples variables respuesta |

---

## Ejercicios Propuestos

1. **Ventas**: Prueba si hay diferencias en ventas entre 4 campañas publicitarias con ANOVA. ¿Cuáles son significativamente diferentes?
2. **Compras**: Con ANOVA de dos factores, evalúa si el precio depende del proveedor y la temporada
3. **Inventarios**: Usa ANOVA para comparar la rotación de inventario entre categorías A, B y C

---

[← Anterior](32-muestreo.md) | [Índice](index.md) | [Siguiente →](34-distribuciones-muestrales.md)
