# Capítulo 16: Prueba Chi-Cuadrado (χ²)

[← Anterior](15-pruebas-no-parametricas.md) | [Índice](index.md) | [Siguiente →](17-correlacion.md)

---

## 16.1 Introducción

La prueba **Chi-Cuadrado (χ²)** es una prueba no paramétrica para analizar **datos categóricos**. Tiene tres aplicaciones principales:

| Tipo | Propósito | Ejemplo |
|------|-----------|---------|
| **Bondad de ajuste** | ¿Los datos siguen una distribución esperada? | ¿Las ventas se distribuyen uniformemente entre días? |
| **Independencia** | ¿Dos variables categóricas están asociadas? | ¿La región está asociada con la categoría de producto? |
| **Homogeneidad** | ¿Diferentes grupos tienen la misma distribución? | ¿La proporción de defectos es igual entre proveedores? |

### Estadístico General

$$\chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}$$

Donde:
- $O_i$ = frecuencia observada en la categoría i
- $E_i$ = frecuencia esperada bajo H₀

---

## 16.2 Prueba de Bondad de Ajuste

¿La distribución observada se ajusta a una distribución teórica?

### Hipótesis
- **H₀**: Los datos siguen la distribución especificada
- **H₁**: Los datos NO siguen la distribución especificada

$$E_i = n \cdot p_i$$
$$\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i} \sim \chi^2_{k-1}$$

```python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)

def chi2_bondad_ajuste(observados, esperados_pct, nombres=None):
    """Prueba Chi-cuadrado de bondad de ajuste"""
    n = sum(observados)
    esperados = [n * p for p in esperados_pct]
    
    chi2_stat = sum((o - e)**2 / e for o, e in zip(observados, esperados))
    gl = len(observados) - 1
    p_valor = 1 - stats.chi2.cdf(chi2_stat, gl)
    
    # Usando SciPy
    chi2_scipy, p_scipy = stats.chisquare(observados, f_exp=esperados)
    
    if nombres is None:
        nombres = [f'Cat {i+1}' for i in range(len(observados))]
    
    print("=== χ² BONDAD DE AJUSTE ===")
    tabla = pd.DataFrame({
        'Categoría': nombres,
        'Observado': observados,
        'Esperado %': esperados_pct,
        'Esperado': [f'{e:.1f}' for e in esperados],
        '(O-E)²/E': [f'{(o-e)**2/e:.2f}' for o, e in zip(observados, esperados)]
    })
    print(tabla.to_string(index=False))
    print(f"\nχ² = {chi2_stat:.3f}")
    print(f"gl = {gl}")
    print(f"p-valor = {p_valor:.4f}")
    print(f"χ² crítico (α=0.05) = {stats.chi2.ppf(0.95, gl):.3f}")
    print(f"¿Se ajusta a la distribución esperada? {'SÍ (p≥0.05)' if p_valor >= 0.05 else 'NO (p<0.05)'}")

# Ejemplo: Distribución de ventas por día de la semana
# Esperamos igual distribución (1/7 ≈ 14.29% cada día)
dias = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
ventas_observadas = [95, 88, 92, 105, 150, 210, 60]  # Más fines de semana
p_esperado = [1/7] * 7

chi2_bondad_ajuste(ventas_observadas, p_esperado, dias)

print("\nConclusión: Las ventas NO se distribuyen uniformemente")
print("→ Hay un efecto día de la semana (más ventas en fin de semana)")
```

---

## 16.3 Prueba de Independencia (Tabla de Contingencia)

¿Dos variables categóricas son independientes?

### Hipótesis
- **H₀**: Las variables son independientes
- **H₁**: Las variables están asociadas

$$E_{ij} = \frac{(\text{Total fila }i) \times (\text{Total columna }j)}{n}$$

```python
def chi2_independencia(df, var_fila, var_col, titulo=""):
    """Prueba Chi-cuadrado de independencia"""
    # Tabla de contingencia
    tabla = pd.crosstab(df[var_fila], df[var_col])
    
    # Prueba
    chi2, p_valor, gl, frec_esperadas = stats.chi2_contingency(tabla)
    
    print(f"=== χ² INDEPENDENCIA: {var_fila} vs {var_col} {titulo}===")
    print("\nTabla de contingencia:")
    print(tabla)
    print(f"\nχ² = {chi2:.3f}")
    print(f"gl = {gl}")
    print(f"p-valor = {p_valor:.4f}")
    print(f"¿Variables independientes? {'SÍ (p≥0.05)' if p_valor >= 0.05 else 'NO (p<0.05)'}")
    
    # V de Cramer (tamaño del efecto)
    n = len(df)
    v_cramer = np.sqrt(chi2 / (n * min(tabla.shape) - 1))
    print(f"V de Cramer: {v_cramer:.4f} (0=indep, 1=dependencia perfecta)")
    
    # Residuos estandarizados
    if isinstance(frec_esperadas, np.ndarray):
        residuos = (tabla.values - frec_esperadas) / np.sqrt(frec_esperadas)
        print("\nResiduos estandarizados (>2 indica asociación fuerte):")
        residuos_df = pd.DataFrame(residuos, index=tabla.index, columns=tabla.columns)
        print(residuos_df.round(2))
    
    return chi2, p_valor, v_cramer

# Ejemplo: ¿El canal de venta depende de la región?
np.random.seed(42)
n = 500
datos = pd.DataFrame({
    'region': np.random.choice(['Norte', 'Sur', 'Centro', 'Oeste'], n),
    'canal': np.random.choice(['Online', 'Tienda', 'Catálogo'], n),
    'categoria': np.random.choice(['Electrónica', 'Ropa', 'Hogar', 'Deportes'], n)
})

# Hacemos que región y canal NO sean independientes
for i in range(n):
    if datos.loc[i, 'region'] == 'Norte':
        datos.loc[i, 'canal'] = np.random.choice(['Online', 'Tienda', 'Catálogo'], 
                                                   p=[0.6, 0.3, 0.1])
    elif datos.loc[i, 'region'] == 'Sur':
        datos.loc[i, 'canal'] = np.random.choice(['Online', 'Tienda', 'Catálogo'],
                                                   p=[0.3, 0.5, 0.2])

chi2_independencia(datos, 'region', 'canal')
```

---

## 16.4 Prueba de Homogeneidad

¿Diferentes poblaciones tienen la misma distribución de una variable categórica?

```python
def chi2_homogeneidad(tabla, nombres_grupos=None):
    """Prueba Chi-cuadrado de homogeneidad"""
    chi2, p_valor, gl, esperadas = stats.chi2_contingency(tabla)
    
    if nombres_grupos is None:
        nombres_grupos = [f'Grupo {i+1}' for i in range(len(tabla))]
    
    print("=== χ² HOMOGENEIDAD ===")
    print("Tabla de contingencias:")
    print(tabla)
    print(f"\nχ² = {chi2:.3f}")
    print(f"gl = {gl}")
    print(f"p-valor = {p_valor:.4f}")
    print(f"¿Grupos homogéneos? {'SÍ (p≥0.05)' if p_valor >= 0.05 else 'NO (p<0.05)'}")
    
    return chi2, p_valor

# Ejemplo: ¿La calidad del producto es homogénea entre proveedores?
tabla_calidad = pd.DataFrame({
    'Calidad A': [45, 30, 25],
    'Calidad B': [30, 35, 25],
    'Calidad C': [5, 15, 10]
}, index=['Proveedor 1', 'Proveedor 2', 'Proveedor 3'])

chi2_homogeneidad(tabla_calidad)
```

---

## 16.5 Supuestos y Limitaciones

### Regla de Cochran
- **Ninguna** celda esperada debe ser < 1
- **Máximo 20%** de las celdas deben tener frecuencia esperada < 5

```python
def verificar_supuestos_chi2(observados, esperados=None, titulo=""):
    """Verifica supuestos de la prueba Chi-cuadrado"""
    print(f"=== VERIFICACIÓN DE SUPUESTOS - {titulo} ===")
    
    if esperados is None:
        esperados = np.array(observados)  # Para bondad de ajuste
    
    celdas_pequenas = sum(1 for e in esperados.flatten() if e < 5)
    total_celdas = esperados.size
    pct_pequenas = celdas_pequenas / total_celdas * 100
    
    print(f"Celdas con E < 5: {celdas_pequenas}/{total_celdas} ({pct_pequenas:.1f}%)")
    print(f"Celdas con E < 1: {sum(1 for e in esperados.flatten() if e < 1)}")
    
    if celdas_pequenas == 0:
        print("✅ Todos los supuestos se cumplen")
    elif pct_pequenas <= 20:
        print("⚠ Supuestos marginalmente cumplidos")
        print("  → Considerar simulación exacta (Fisher)")
    else:
        print("❌ Supuestos NO cumplidos")
        print("  → Usar prueba exacta de Fisher o agrupar categorías")

# Ejemplo con muestra pequeña
tabla_pequena = np.array([[3, 0], [1, 5]])
verificar_supuestos_chi2(tabla_pequena, esperados=None, titulo="Tabla pequeña")
```

---

## 16.6 Prueba Exacta de Fisher

Alternativa al Chi-cuadrado cuando los supuestos no se cumplen (tablas 2×2).

```python
from scipy.stats import fisher_exact

# Tabla 2×2 pequeña: tratamiento vs resultado
tabla_2x2 = [[8, 2], [3, 7]]  # [[éxito, fracaso], [control, tratamiento]]

odds_ratio, p_valor = fisher_exact(tabla_2x2)

print("=== PRUEBA EXACTA DE FISHER ===")
print("Tabla 2×2:")
print(pd.DataFrame(tabla_2x2, 
                   index=['Tratamiento', 'Control'],
                   columns=['Éxito', 'Fracaso']))
print(f"Odds Ratio: {odds_ratio:.3f}")
print(f"p-valor (bilateral): {p_valor:.4f}")
print(f"¿Asociación significativa? {'SÍ' if p_valor < 0.05 else 'NO'}")
```

---

## 16.7 Medidas de Asociación para Tablas de Contingencia

### Phi (φ) - Tablas 2×2
$$\phi = \sqrt{\frac{\chi^2}{n}}$$

### V de Cramer - Tablas r×c
$$V = \sqrt{\frac{\chi^2}{n \cdot \min(r-1, c-1)}}$$

### Coeficiente de Contingencia
$$C = \sqrt{\frac{\chi^2}{\chi^2 + n}}$$

```python
def medidas_asociacion(tabla):
    """Calcula medidas de asociación para tablas de contingencia"""
    chi2, p, gl, _ = stats.chi2_contingency(tabla)
    n = tabla.sum().sum()
    r, c = tabla.shape
    
    if r == 2 and c == 2:
        phi = np.sqrt(chi2 / n)
        print(f"Phi (φ): {phi:.4f}")
    else:
        v_cramer = np.sqrt(chi2 / (n * min(r-1, c-1)))
        print(f"V de Cramer: {v_cramer:.4f}")
    
    c_cont = np.sqrt(chi2 / (chi2 + n))
    print(f"Coeficiente de contingencia: {c_cont:.4f}")
    print(f"Chi-cuadrado: {chi2:.3f}")
    print(f"p-valor: {p:.4f}")
    
    return chi2, p

# Ejemplo
tabla_test = pd.DataFrame({
    'Compra': [120, 80],
    'No Compra': [60, 140]
}, index=['Campaña A', 'Campaña B'])

print("=== MEDIDAS DE ASOCIACIÓN ===")
print(tabla_test)
medidas_asociacion(tabla_test)
```

---

## 16.8 Ejemplo en Ventas: Campañas por Región

```python
# Análisis de efectividad de campañas por región
np.random.seed(42)

datos_campana = pd.DataFrame({
    'region': np.random.choice(['Norte', 'Sur', 'Centro', 'Oeste'], 800),
    'campana': np.random.choice(['Email', 'Redes', 'TV', 'Radio'], 800),
    'compro': np.random.choice([True, False], 800, p=[0.3, 0.7])
})

# Creamos dependencia
for i in range(len(datos_campana)):
    if datos_campana.loc[i, 'region'] == 'Norte' and datos_campana.loc[i, 'campana'] == 'TV':
        datos_campana.loc[i, 'compro'] = np.random.choice([True, False], p=[0.6, 0.4])

print("=== VENTAS: CAMPAÑA × REGIÓN ===")
chi2, p, v = chi2_independencia(datos_campana, 'region', 'campana')

# Tabla de conversión
tabla_conv = pd.crosstab(
    [datos_campana['region'], datos_campana['campana']],
    datos_campana['compro'],
    normalize='index'
) * 100
print("\nTasa de conversión por región y campaña (%):")
print(tabla_conv.round(1))
```

---

## 16.9 Ejemplo en Compras: Defectos por Proveedor

```python
# Análisis de defectos por proveedor
np.random.seed(42)

datos_defectos = pd.DataFrame({
    'proveedor': np.random.choice(['Prov_A', 'Prov_B', 'Prov_C', 'Prov_D'], 500),
    'calidad': np.random.choice(['Aceptable', 'Defectuoso'], 500, p=[0.92, 0.08])
})

# Creamos un proveedor con más defectos
datos_defectos.loc[datos_defectos['proveedor'] == 'Prov_D', 'calidad'] = \
    np.random.choice(['Aceptable', 'Defectuoso'], 
                     size=datos_defectos['proveedor'].value_counts()['Prov_D'],
                     p=[0.80, 0.20])

print("=== COMPRAS: DEFECTOS POR PROVEEDOR ===")
tabla_def = pd.crosstab(datos_defectos['proveedor'], datos_defectos['calidad'])
print(tabla_def)

chi2, p, gl, esperadas = stats.chi2_contingency(tabla_def)
print(f"\nχ² = {chi2:.3f}")
print(f"p-valor = {p:.4f}")
print(f"¿Hay diferencias en calidad entre proveedores? {'SÍ' if p < 0.05 else 'NO'}")

# Tasa de defectos por proveedor
tasa_defectos = datos_defectos.groupby('proveedor')['calidad'].apply(
    lambda x: (x == 'Defectuoso').mean()
)
print(f"\nTasa de defectos por proveedor:")
print(tasa_defectos)
```

---

## 16.10 Ejemplo en Inventarios: Rotación por Ubicación

```python
# Relación entre ubicación en almacén y rotación
np.random.seed(42)

n = 300
datos_inv = pd.DataFrame({
    'ubicacion': np.random.choice(['Zona A (acceso rápido)', 'Zona B', 'Zona C (fondo)'], n),
    'rotacion': np.random.choice(['Alta', 'Media', 'Baja'], n)
})

# Creamos asociación
for i in range(n):
    if datos_inv.loc[i, 'ubicacion'] == 'Zona A (acceso rápido)':
        datos_inv.loc[i, 'rotacion'] = np.random.choice(['Alta', 'Media', 'Baja'], p=[0.6, 0.3, 0.1])
    elif datos_inv.loc[i, 'ubicacion'] == 'Zona C (fondo)':
        datos_inv.loc[i, 'rotacion'] = np.random.choice(['Alta', 'Media', 'Baja'], p=[0.1, 0.3, 0.6])

print("=== INVENTARIOS: UBICACIÓN × ROTACIÓN ===")
chi2_independencia(datos_inv, 'ubicacion', 'rotacion')

print("\nDistribución de rotación por ubicación:")
tabla_rot = pd.crosstab(datos_inv['ubicacion'], datos_inv['rotacion'], normalize='index') * 100
print(tabla_rot.round(1))
```

---

## 16.11 Análisis de Residuales Post-Pr

Los **residuales estandarizados** identifican qué celdas contribuyen más al χ²:

$$r_{ij} = \frac{O_{ij} - E_{ij}}{\sqrt{E_{ij}}}$$

Valores de $|r_{ij}| > 2$ indican asociación significativa en esa celda.

```python
def analisis_residuales(tabla):
    """Análisis detallado de residuales estandarizados"""
    chi2, p, gl, esperadas = stats.chi2_contingency(tabla)
    residuos = (tabla.values - esperadas) / np.sqrt(esperadas)
    
    print("=== ANÁLISIS DE RESIDUALES ===")
    print(f"χ² = {chi2:.3f}, p = {p:.4f}")
    print("\nResiduales estandarizados (|r| > 2 → contribución significativa):")
    
    residuos_df = pd.DataFrame(
        residuos,
        index=tabla.index,
        columns=tabla.columns
    )
    print(residuos_df.round(2))
    
    # Celdas con residual significativo
    print("\nCeldas con mayor contribución:")
    for i in residuos_df.index:
        for j in residuos_df.columns:
            r = residuos_df.loc[i, j]
            if abs(r) > 2:
                o_ij = tabla.loc[i, j]
                e_ij = esperadas[tabla.index.get_loc(i), tabla.columns.get_loc(j)]
                print(f"  {i} × {j}: r={r:.2f} (O={o_ij:.0f}, E={e_ij:.1f})")
    
    return residuos_df

# Aplicamos a ejemplo anterior
tabla_ejemplo = pd.crosstab(datos_defectos['proveedor'], datos_defectos['calidad'])
analisis_residuales(tabla_ejemplo)
```

---

## 16.12 Resumen

| Aplicación | Hipótesis Nula (H₀) | Grados de Libertad |
|-----------|---------------------|-------------------|
| **Bondad de Ajuste** | Los datos siguen la distribución esperada | k - 1 |
| **Independencia** | Las variables son independientes | (r-1)(c-1) |
| **Homogeneidad** | Los grupos tienen la misma distribución | (r-1)(c-1) |

### Checklist
1. ✅ Datos categóricos (frecuencias)
2. ✅ Muestra aleatoria independiente
3. ✅ Frecuencias esperadas ≥ 5 (regla de Cochran)
4. ✅ Si no se cumple: usar Fisher o agrupar categorías

---

## Ejercicios Propuestos

1. **Ventas**: Prueba si la preferencia de producto (A, B, C) es independiente de la región (3 regiones)
2. **Compras**: ¿La proporción de defectos es homogénea entre 4 proveedores? Datos: 500 unidades cada uno
3. **Inventarios**: Prueba bondad de ajuste para verificar si la demanda sigue distribución Poisson(λ=25)

---

[← Anterior](15-pruebas-no-parametricas.md) | [Índice](index.md) | [Siguiente →](17-correlacion.md)
