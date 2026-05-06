# Capítulo 19: Regresión Lineal Múltiple

[← Anterior](18-regresion-lineal.md) | [Índice](index.md) | [Siguiente →](20-regresion-logistica.md)

---

## 19.1 Introducción

La **regresión lineal múltiple** extiende la regresión simple a **múltiples variables predictoras**:

$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_k X_k + \varepsilon$$

### Cuándo Usarla

- Cuando múltiples factores afectan una variable de interés
- Para **controlar** por variables de confusión
- Para mejorar la **precisión de predicciones**
- Para entender la **importancia relativa** de cada predictor

---

## 19.2 Implementación con StatsModels

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Datos simulados de ventas con múltiples predictores
n = 200
datos = pd.DataFrame({
    'publicidad_tv': np.random.uniform(10, 100, n),
    'publicidad_redes': np.random.uniform(5, 80, n),
    'publicidad_radio': np.random.uniform(0, 30, n),
    'precio': np.random.uniform(30, 120, n),
    'descuento': np.random.choice([0, 5, 10, 15, 20], n, p=[0.3, 0.3, 0.2, 0.1, 0.1]),
    'satisfaccion': np.random.uniform(3, 5, n),
    'tienda_anos': np.random.randint(1, 20, n),
    'competidores': np.random.randint(0, 8, n)
})

# Generamos ventas con relación lineal
datos['ventas'] = (
    200 +
    3.5 * datos['publicidad_tv'] +
    2.0 * datos['publicidad_redes'] +
    1.0 * datos['publicidad_radio'] -
    1.2 * datos['precio'] +
    5.0 * datos['descuento'] +
    20 * datos['satisfaccion'] +
    5 * datos['tienda_anos'] -
    8 * datos['competidores'] +
    np.random.normal(0, 30, n)
)

print("=== DATOS DE VENTAS CON MÚLTIPLES PREDICTORES ===")
print(datos.head())
print(f"\nDimensiones: {datos.shape}")
```

### Modelo Completo

```python
# Variables predictoras
predictores = ['publicidad_tv', 'publicidad_redes', 'publicidad_radio', 
               'precio', 'descuento', 'satisfaccion', 'tienda_anos', 'competidores']
X = datos[predictores]
y = datos['ventas']

X_const = sm.add_constant(X)
modelo_completo = sm.OLS(y, X_const).fit()

print("=== REGRESIÓN LINEAL MÚLTIPLE - MODELO COMPLETO ===")
print(modelo_completo.summary())

# Tabla de coeficientes
print("\n=== COEFICIENTES DEL MODELO ===")
coef_df = pd.DataFrame({
    'Predictor': ['Constante'] + predictores,
    'Coeficiente': modelo_completo.params.values,
    'Std Error': modelo_completo.bse.values,
    't': modelo_completo.tvalues.values,
    'p-valor': modelo_completo.pvalues.values,
    'IC 95% Inf': modelo_completo.conf_int().values[:, 0],
    'IC 95% Sup': modelo_completo.conf_int().values[:, 1]
})
coef_df['Signif'] = coef_df['p-valor'].apply(
    lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
)
print(coef_df.to_string(index=False))
```

---

## 19.3 Interpretación de Coeficientes

```python
print("=== INTERPRETACIÓN DE COEFICIENTES ===")
for pred in predictores:
    coef = modelo_completo.params[pred]
    p_val = modelo_completo.pvalues[pred]
    sig = 'significativo' if p_val < 0.05 else 'NO significativo'
    print(f"\n{pred}:")
    print(f"  β = {coef:.4f} ({sig}, p={p_val:.4f})")
    if 'publicidad' in pred:
        print(f"  → Por cada $1K adicional en {pred}, las ventas cambian en ${coef:.2f}")
    elif pred == 'precio':
        print(f"  → Por cada $1 de aumento en precio, las ventas cambian en ${coef:.2f}")
    elif pred == 'satisfaccion':
        print(f"  → Por cada punto de satisfacción, las ventas cambian en ${coef:.2f}")

# Interpretación conjunta
print(f"\n=== METRICAS GLOBALES ===")
print(f"R² = {modelo_completo.rsquared:.4f} ({modelo_completo.rsquared*100:.1f}% de varianza explicada)")
print(f"R² ajustado = {modelo_completo.rsquared_adj:.4f}")
print(f"F({len(predictores)}, {n-len(predictores)-1}) = {modelo_completo.fvalue:.3f}, p = {modelo_completo.f_pvalue:.6f}")
print(f"AIC = {modelo_completo.aic:.1f}")
print(f"BIC = {modelo_completo.bic:.1f}")
```

---

## 19.4 Selección de Variables

### Métodos de Selección

```python
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.linear_model import LinearRegression

# 1. Correlación con la variable dependiente
correlaciones = datos[predictores + ['ventas']].corr()['ventas'].drop('ventas').sort_values(ascending=False)
print("=== CORRELACIÓN CON VENTAS ===")
print(correlaciones)

# 2. Feature Importance con sklearn
lr = LinearRegression()
lr.fit(datos[predictores], datos['ventas'])
importancias = pd.DataFrame({
    'Predictor': predictores,
    'Coeficiente': lr.coef_,
    '|Coef|': np.abs(lr.coef_)
}).sort_values('|Coef|', ascending=False)
print("\n=== IMPORTANCIA DE PREDICTORES ===")
print(importancias)

# 3. Selección hacia adelante (Forward Selection)
def forward_selection(X, y, threshold_in=0.05):
    """Forward selection basado en p-valor"""
    included = []
    while True:
        best_p = 1.0
        best_var = None
        for var in X.columns:
            if var not in included:
                cols = included + [var]
                model = sm.OLS(y, sm.add_constant(X[cols])).fit()
                p = model.pvalues[var]
                if p < best_p:
                    best_p = p
                    best_var = var
        if best_p < threshold_in:
            included.append(best_var)
        else:
            break
    return included

selected = forward_selection(datos[predictores], datos['ventas'])
print(f"\n=== FORWARD SELECTION ===")
print(f"Variables seleccionadas: {selected}")
```

---

## 19.5 Multicolinealidad

La **multicolinealidad** ocurre cuando dos o más predictores están altamente correlacionados, lo que infla la varianza de los coeficientes.

### Factor de Inflación de Varianza (VIF)

$$VIF_i = \frac{1}{1 - R_i^2}$$

Donde $R_i^2$ es el R² de la regresión del predictor i contra los demás.

```python
def calcular_vif(X):
    """Calcula VIF para cada predictor"""
    vif_data = pd.DataFrame()
    vif_data['Predictor'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_data['Tolerancia'] = 1 / vif_data['VIF']
    return vif_data

vif = calcular_vif(datos[predictores])
print("=== FACTOR DE INFLACIÓN DE VARIANZA (VIF) ===")
print(vif.round(4))

print("\nInterpretación:")
for _, row in vif.iterrows():
    if row['VIF'] > 10:
        print(f"⚠ {row['Predictor']}: VIF={row['VIF']:.1f} → Multicolinealidad SEVERA")
    elif row['VIF'] > 5:
        print(f"  {row['Predictor']}: VIF={row['VIF']:.1f} → Multicolinealidad MODERADA")
    else:
        print(f"✓ {row['Predictor']}: VIF={row['VIF']:.1f} → OK")

# Solución: eliminar variables con VIF alto o usar Ridge Regression
```

---

## 19.6 Interacciones entre Variables

A veces el efecto de un predictor depende del valor de otro:

$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 (X_1 \times X_2)$$

```python
# Modelo con interacción: publicidad_tv × descuento
datos['interaccion_tv_descuento'] = datos['publicidad_tv'] * datos['descuento']

X_inter = datos[['publicidad_tv', 'descuento', 'interaccion_tv_descuento']]
X_inter = sm.add_constant(X_inter)
modelo_inter = sm.OLS(datos['ventas'], X_inter).fit()

print("=== MODELO CON INTERACCIÓN ===")
print(modelo_inter.summary().tables[1])

print(f"\nInterpretación:")
print(f"  β_publicidad = {modelo_inter.params['publicidad_tv']:.4f}")
print(f"  β_descuento = {modelo_inter.params['descuento']:.4f}")
print(f"  β_interacción = {modelo_inter.params['interaccion_tv_descuento']:.4f}")
print(f"\n→ La efectividad de la publicidad en TV cambia según el descuento aplicado")
```

---

## 19.7 Variables Categóricas (Dummies)

```python
# Añadimos variables categóricas
datos['region'] = np.random.choice(['Norte', 'Sur', 'Centro', 'Oeste'], n)
datos['temporada'] = np.random.choice(['Alta', 'Baja', 'Media'], n)

# Convertir a dummies
dummies_region = pd.get_dummies(datos['region'], prefix='region', drop_first=True)
dummies_temp = pd.get_dummies(datos['temporada'], prefix='temp', drop_first=True)

X_dummies = pd.concat([datos[predictores], dummies_region, dummies_temp], axis=1)
X_dummies = sm.add_constant(X_dummies)
modelo_dummies = sm.OLS(datos['ventas'], X_dummies).fit()

print("=== MODELO CON VARIABLES CATEGÓRICAS ===")
print(modelo_dummies.summary().tables[1])

# Interpretación de dummies
print("\n=== EFECTO DE REGIÓN (vs Norte = base) ===")
for col in dummies_region.columns:
    coef = modelo_dummies.params[col]
    p = modelo_dummies.pvalues[col]
    print(f"  {col}: β = {coef:.2f} (p = {p:.4f}) → "
          f"Ventas {'mayores' if coef > 0 else 'menores'} en esta región")
```

---

## 19.8 Ejemplo en Ventas: Modelo Predictivo Completo

```python
# Construcción y evaluación de modelo predictivo
from sklearn.model_selection import cross_val_score, KFold

# División train/test
X_train, X_test, y_train, y_test = train_test_split(
    datos[predictores], datos['ventas'], test_size=0.2, random_state=42
)

# Modelo en train
X_train_c = sm.add_constant(X_train)
modelo_train = sm.OLS(y_train, X_train_c).fit()

# Predicciones en test
X_test_c = sm.add_constant(X_test)
y_pred = modelo_train.predict(X_test_c)

print("=== EVALUACIÓN DEL MODELO PREDICTIVO ===")
print(f"R² train: {modelo_train.rsquared:.4f}")
print(f"R² test: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE test: ${np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")
print(f"MAE test: ${mean_absolute_error(y_test, y_pred):,.2f}")

# Validación cruzada (5 folds)
lr_cv = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(lr_cv, datos[predictores], datos['ventas'], 
                         cv=kf, scoring='r2')
print(f"\nValidación cruzada (5 folds):")
print(f"  R² promedio: {scores.mean():.4f} ± {scores.std():.4f}")
```

---

## 19.9 Ejemplo en Compras: Modelo de Costo Total

```python
# Factores que afectan el costo total de compra
np.random.seed(42)
n_compras = 100

compras = pd.DataFrame({
    'volumen': np.random.randint(100, 10000, n_compras),
    'distancia_km': np.random.uniform(10, 1000, n_compras),
    'urgencia': np.random.choice([1, 2, 3], n_compras, p=[0.5, 0.3, 0.2]),
    'proveedor_confiable': np.random.choice([0, 1], n_compras, p=[0.3, 0.7]),
    'costo_combustible': np.random.uniform(20, 50, n_compras)
})

# Costo total
compras['costo_total'] = (
    1000 +  # costo fijo
    2 * compras['volumen'] +  # costo variable por unidad
    0.5 * compras['distancia_km'] +  # flete
    200 * compras['urgencia'] +  # recargo por urgencia
    -100 * compras['proveedor_confiable'] +  # descuento por confiabilidad
    np.random.normal(0, 200, n_compras)
)

X_comp = sm.add_constant(compras[['volumen', 'distancia_km', 'urgencia', 'proveedor_confiable', 'costo_combustible']])
modelo_comp = sm.OLS(compras['costo_total'], X_comp).fit()

print("=== MODELO DE COSTO TOTAL DE COMPRA ===")
print(modelo_comp.summary().tables[1])

# Interpretación de negocio
print(f"\nCosto fijo base: ${modelo_comp.params['const']:.2f}")
print(f"Costo variable por unidad: ${modelo_comp.params['volumen']:.4f}")
print(f"Costo por km: ${modelo_comp.params['distancia_km']:.4f}")
print(f"Recargo por nivel de urgencia: ${modelo_comp.params['urgencia']:.2f}")
print(f"Ahorro por proveedor confiable: ${-modelo_comp.params['proveedor_confiable']:.2f}")
```

---

## 19.10 Ejemplo en Inventarios: Modelo de Stock de Seguridad

```python
# Factores que determinan el stock de seguridad óptimo
np.random.seed(42)
n_prod = 80

inv = pd.DataFrame({
    'demanda_media': np.random.uniform(10, 500, n_prod),
    'demanda_std': np.random.uniform(2, 100, n_prod),
    'lead_time': np.random.choice([2, 3, 5, 7, 10], n_prod, p=[0.2, 0.3, 0.3, 0.1, 0.1]),
    'lead_time_std': np.random.uniform(0.5, 3, n_prod),
    'nivel_servicio_obj': np.random.choice([0.90, 0.95, 0.99], n_prod),
    'costo_unitario': np.random.uniform(10, 200, n_prod),
    'rotacion': np.random.uniform(1, 30, n_prod)
})

# Stock de seguridad (fórmula simplificada con variabilidad)
inv['stock_seguridad'] = (
    1.645 * np.sqrt(
        inv['lead_time'] * inv['demanda_std']**2 + 
        inv['demanda_media']**2 * inv['lead_time_std']**2
    ) + 
    np.random.normal(0, 10, n_prod)
)
inv['stock_seguridad'] = inv['stock_seguridad'].clip(0)

X_inv = sm.add_constant(inv[['demanda_media', 'demanda_std', 'lead_time', 
                              'lead_time_std', 'rotacion']])
modelo_inv = sm.OLS(inv['stock_seguridad'], X_inv).fit()

print("=== MODELO DE STOCK DE SEGURIDAD ===")
print(modelo_inv.summary().tables[1])

print(f"\nFactores que MÁS impactan el stock de seguridad:")
coefs_abs = modelo_inv.params.drop('const').abs().sort_values(ascending=False)
for var, coef in coefs_abs.items():
    print(f"  {var}: |β| = {coef:.4f}")
```

---

## 19.11 Regularización (Ridge, Lasso, ElasticNet)

Para evitar overfitting y manejar multicolinealidad:

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler

# Estandarizar predictores
scaler = StandardScaler()
X_scaled = scaler.fit_transform(datos[predictores])

# Ridge (L2)
ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, datos['ventas'])

# Lasso (L1) - selecciona variables
lasso = Lasso(alpha=1.0)
lasso.fit(X_scaled, datos['ventas'])

# ElasticNet (L1 + L2)
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)
elastic.fit(X_scaled, datos['ventas'])

print("=== COMPARACIÓN: MODELOS REGULARIZADOS ===")
print(f"{'Variable':20s} {'OLS':>10s} {'Ridge':>10s} {'Lasso':>10s} {'Elastic':>10s}")
print("-" * 60)
for i, pred in enumerate(predictores):
    print(f"{pred:20s} {lr.coef_[i]:>10.4f} {ridge.coef_[i]:>10.4f} "
          f"{lasso.coef_[i]:>10.4f} {elastic.coef_[i]:>10.4f}")

# Lasso elimina variables (coeficientes = 0)
eliminadas = [predictores[i] for i in range(len(predictores)) if lasso.coef_[i] == 0]
print(f"\nLasso eliminó {len(eliminadas)} variables: {eliminadas}")
```

---

## 19.12 Resumen

| Concepto | Descripción |
|----------|-------------|
| **R² ajustado** | Penaliza por número de predictores |
| **VIF** | Detecta multicolinealidad (>10 es severo) |
| **Interacciones** | Efecto conjunto de predictores |
| **Dummies** | Variables categóricas en regresión |
| **Regularización** | Ridge, Lasso, ElasticNet para overfitting |
| **Selección** | Forward, Backward, Stepwise |

---

## Ejercicios Propuestos

1. **Ventas**: Construye un modelo con 5 predictores de ventas. Calcula VIF y elimina multicolinealidad
2. **Compras**: Modela costo total en función de volumen, distancia y urgencia. Interpreta coeficientes
3. **Inventarios**: Crea un modelo de regresión múltiple para predecir stock de seguridad. Compara OLS vs Ridge

---

[← Anterior](18-regresion-lineal.md) | [Índice](index.md) | [Siguiente →](20-regresion-logistica.md)
