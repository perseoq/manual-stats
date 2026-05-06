# Capítulo 20: Regresión Logística

[← Anterior](19-regresion-multiple.md) | [Índice](index.md) | [Siguiente →](21-series-temporales-intro.md)

---

## 20.1 Introducción

La **regresión logística** modela la probabilidad de un evento binario (sí/no, éxito/fracaso, compra/no compra).

### ¿Por qué no usar regresión lineal?
- La variable dependiente es binaria (0/1)
- Los valores predichos deben estar entre 0 y 1
- La relación no es lineal sino sigmoide

### Función Logística (Sigmoide)

$$P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + ... + \beta_k X_k)}}$$

### Odds y Log-Odds

$$Odds = \frac{P(Y=1)}{P(Y=0)} = \frac{p}{1-p}$$

$$\ln\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 X_1 + ... + \beta_k X_k$$

---

## 20.2 Implementación con Scikit-learn

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (confusion_matrix, classification_report, 
                             roc_curve, roc_auc_score, accuracy_score,
                             precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler
import seaborn as sns

np.random.seed(42)

# Datos simulados: ¿un cliente compra?
n = 1000
datos_compra = pd.DataFrame({
    'edad': np.random.randint(18, 70, n),
    'ingreso': np.random.normal(50000, 20000, n),
    'visitas_sitio': np.random.poisson(3, n),
    'tiempo_sitio_min': np.random.exponential(10, n),
    'email_abierto': np.random.choice([0, 1], n, p=[0.4, 0.6]),
    'descuento_ofrecido': np.random.choice([0, 5, 10, 15, 20], n, p=[0.3, 0.25, 0.2, 0.15, 0.1])
})

# Probabilidad de compra (función logística)
log_odds = (
    -3 +
    0.02 * datos_compra['edad'] +
    0.00002 * datos_compra['ingreso'] +
    0.3 * datos_compra['visitas_sitio'] +
    0.05 * datos_compra['tiempo_sitio_min'] +
    0.8 * datos_compra['email_abierto'] +
    0.05 * datos_compra['descuento_ofrecido']
)
prob_compra = 1 / (1 + np.exp(-log_odds))
datos_compra['compro'] = np.random.binomial(1, prob_compra)

print(f"Tasa de compra: {datos_compra['compro'].mean():.2%}")
print(f"\nPrimeras filas:")
print(datos_compra.head())
```

### Entrenamiento del Modelo

```python
# Preparar datos
predictores = ['edad', 'ingreso', 'visitas_sitio', 'tiempo_sitio_min', 
               'email_abierto', 'descuento_ofrecido']
X = datos_compra[predictores]
y = datos_compra['compro']

# Estandarizar (importante para logística)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# Modelo
modelo_log = LogisticRegression(max_iter=1000)
modelo_log.fit(X_train, y_train)

print("=== REGRESIÓN LOGÍSTICA - COEFICIENTES ===")
coef_df = pd.DataFrame({
    'Predictor': predictores,
    'Coeficiente (β)': modelo_log.coef_[0],
    'Odds Ratio (e^β)': np.exp(modelo_log.coef_[0])
})
print(coef_df)
print(f"Intersección (β₀): {modelo_log.intercept_[0]:.4f}")

### Interpretación de Negocio

Los coeficientes de la regresión logística no se interpretan directamente como cambios en la probabilidad, sino como cambios en el **log-odds** (logaritmo de la razón de probabilidades). Al aplicar la exponencial, obtenemos el **Odds Ratio (OR)**, que sí tiene una interpretación intuitiva: un OR > 1 significa que el predictor aumenta las odds del evento (compra), mientras que OR < 1 las disminuye. Por ejemplo, si `visitas_sitio` tiene OR = 1.35, significa que por cada visita adicional al sitio web, las odds de compra se multiplican por 1.35 (aumentan un 35%), manteniendo todo lo demás constante. Esta interpretación "manteniendo todo lo demás constante" (ceteris paribus) es crucial: cada coeficiente refleja el efecto de esa variable controlando por las demás. El intercepto $\beta_0$ representa el log-odds cuando todos los predictores son cero, aunque esto puede no ser realista en la práctica.
```

---

## 20.3 Interpretación: Odds Ratio

$$Odds\ Ratio = e^{\beta_i}$$

- **OR = 1**: El predictor no afecta la probabilidad
- **OR > 1**: Aumenta la probabilidad (asociación positiva)
- **OR < 1**: Disminuye la probabilidad (asociación negativa)

```python
print("=== INTERPRETACIÓN DE ODDS RATIO ===")
for i, pred in enumerate(predictores):
    or_val = np.exp(modelo_log.coef_[0][i])
    efecto = "aumenta" if or_val > 1 else "disminuye"
    print(f"\n{pred}: OR = {or_val:.4f}")
    if or_val > 1:
        print(f"  → Por cada unidad de aumento en {pred},")
        print(f"    las odds de compra se multiplican por {or_val:.3f}")
        print(f"    (aumento del {(or_val-1)*100:.1f}%)")
    else:
        print(f"  → Por cada unidad de aumento en {pred},")
        print(f"    las odds de compra se multiplican por {or_val:.3f}")
        print(f"    (disminución del {(1-or_val)*100:.1f}%)")
```

---

## 20.4 Evaluación del Modelo

```python
# Predicciones
y_pred = modelo_log.predict(X_test)
y_prob = modelo_log.predict_proba(X_test)[:, 1]

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)

print("=== MATRIZ DE CONFUSIÓN ===")
print(pd.DataFrame(cm, index=['Real: No compra', 'Real: Compra'],
                   columns=['Pred: No compra', 'Pred: Compra']))

# Métricas
print(f"\n=== MÉTRICAS DE CLASIFICACIÓN ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")

# Reporte completo
print(f"\n=== CLASIFICATION REPORT ===")
print(classification_report(y_test, y_pred, target_names=['No compra', 'Compra']))
```

### Curva ROC y AUC

```python
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.4f})')
ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Aleatorio (AUC = 0.5)')
ax.fill_between(fpr, tpr, alpha=0.2, color='steelblue')
ax.set_xlabel('Tasa de Falsos Positivos (1 - Especificidad)', fontweight='bold')
ax.set_ylabel('Tasa de Verdaderos Positivos (Sensibilidad)', fontweight='bold')
ax.set_title('Curva ROC - Modelo de Predicción de Compra', fontweight='bold')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.savefig('roc_logistica.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n=== INTERPRETACIÓN DEL AUC ===")
if auc >= 0.9: print("Excelente discriminación")
elif auc >= 0.8: print("Buena discriminación")
elif auc >= 0.7: print("Aceptable")
else: print("Pobre discriminación")
```

---

## 20.5 Umbral de Decisión

El umbral por defecto es 0.5, pero podemos ajustarlo según el costo de errores:

```python
def optimizar_umbral(y_true, y_prob):
    """Encuentra el umbral óptimo según F1-score"""
    umbrales = np.linspace(0.1, 0.9, 50)
    f1_scores = []
    
    for umbral in umbrales:
        y_pred_umbral = (y_prob >= umbral).astype(int)
        f1_scores.append(f1_score(y_true, y_pred_umbral))
    
    idx_optimo = np.argmax(f1_scores)
    umbral_optimo = umbrales[idx_optimo]
    
    return umbral_optimo, max(f1_scores), umbrales, f1_scores

umbral_opt, f1_opt, umbrales, f1_scores = optimizar_umbral(y_test, y_prob)

print("=== OPTIMIZACIÓN DE UMBRAL ===")
print(f"Umbral óptimo: {umbral_opt:.2f}")
print(f"F1-score óptimo: {f1_opt:.4f}")

# Comparación
y_pred_default = (y_prob >= 0.5).astype(int)
y_pred_optimo = (y_prob >= umbral_opt).astype(int)

print(f"\nComparación de umbrales:")
print(f"{'Métrica':20s} {'Umbral 0.5':>12s} {'Umbral óptimo':>15s}")
print("-" * 47)
print(f"{'Accuracy':20s} {accuracy_score(y_test, y_pred_default):>12.4f} {accuracy_score(y_test, y_pred_optimo):>15.4f}")
print(f"{'Precision':20s} {precision_score(y_test, y_pred_default):>12.4f} {precision_score(y_test, y_pred_optimo):>15.4f}")
print(f"{'Recall':20s} {recall_score(y_test, y_pred_default):>12.4f} {recall_score(y_test, y_pred_optimo):>15.4f}")
print(f"{'F1':20s} {f1_score(y_test, y_pred_default):>12.4f} {f1_score(y_test, y_pred_optimo):>15.4f}")
```

---

## 20.6 Validación Cruzada

```python
# Validación cruzada para evaluar estabilidad
cv_scores = cross_val_score(modelo_log, X_scaled, y, cv=5, scoring='roc_auc')

print("=== VALIDACIÓN CRUZADA (5 FOLDS) ===")
print(f"AUC por fold: {cv_scores}")
print(f"AUC promedio: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

---

## 20.7 Ejemplo en Ventas: Predicción de Compra

```python
# Segmentación de clientes por probabilidad de compra
datos_compra['prob_compra'] = modelo_log.predict_proba(X_scaled)[:, 1]

# Crear segmentos
datos_compra['segmento'] = pd.cut(
    datos_compra['prob_compra'],
    bins=[0, 0.2, 0.4, 0.6, 0.8, 1],
    labels=['Muy Baja', 'Baja', 'Media', 'Alta', 'Muy Alta']
)

print("=== SEGMENTACIÓN POR PROBABILIDAD DE COMPRA ===")
segmentos = datos_compra.groupby('segmento', observed=True).agg({
    'compro': ['count', 'mean'],
    'ingreso': 'mean',
    'visitas_sitio': 'mean'
}).round(2)
print(segmentos)

# Perfil del cliente ideal
print(f"\n=== PERFIL DEL CLIENTE CON MAYOR PROBABILIDAD ===")
top_clientes = datos_compra.nlargest(10, 'prob_compra')
print(top_clientes[predictores + ['prob_compra']])
```

---

## 20.8 Ejemplo en Compras: Aprobación de Proveedor

```python
# ¿Aprobar o no a un proveedor?
np.random.seed(42)
n_prov = 500

proveedores = pd.DataFrame({
    'anos_experiencia': np.random.randint(1, 30, n_prov),
    'certificaciones': np.random.randint(0, 5, n_prov),
    'capacidad_mensual': np.random.uniform(100, 10000, n_prov),
    'precio_relativo': np.random.uniform(0.7, 1.3, n_prov),  # 1.0 = mercado
    'tiempo_entrega_prom': np.random.uniform(2, 20, n_prov),
    'reclamos_pasados': np.random.poisson(1, n_prov)
})

# Probabilidad de aprobación
log_odds_prov = (
    2 +
    0.05 * proveedores['anos_experiencia'] +
    0.3 * proveedores['certificaciones'] +
    0.0001 * proveedores['capacidad_mensual'] -
    2 * (proveedores['precio_relativo'] - 1) -
    0.1 * proveedores['tiempo_entrega_prom'] -
    0.5 * proveedores['reclamos_pasados']
)
prob_aprob = 1 / (1 + np.exp(-log_odds_prov))
proveedores['aprobado'] = np.random.binomial(1, prob_aprob)

# Modelo
X_prov = proveedores[['anos_experiencia', 'certificaciones', 'capacidad_mensual',
                       'precio_relativo', 'tiempo_entrega_prom', 'reclamos_pasados']]
y_prov = proveedores['aprobado']

modelo_prov = LogisticRegression(max_iter=1000)
modelo_prov.fit(X_prov, y_prov)

print("=== APRENDIZAJE: APROBACIÓN DE PROVEEDORES ===")
coef_prov = pd.DataFrame({
    'Factor': X_prov.columns,
    'β': modelo_prov.coef_[0],
    'OR': np.exp(modelo_prov.coef_[0])
}).sort_values('OR', ascending=False)
print(coef_prov)

# Regla de decisión
print(f"\nFactores que más influyen en aprobación:")
print(f"  + {coef_prov.iloc[0]['Factor']} (OR = {coef_prov.iloc[0]['OR']:.3f})")
print(f"  - {coef_prov.iloc[-1]['Factor']} (OR = {coef_prov.iloc[-1]['OR']:.3f})")
```

---

## 20.9 Ejemplo en Inventarios: Riesgo de Stockout

```python
# Predecir si un producto tendrá stockout en la próxima semana
np.random.seed(42)
n_prod = 300

productos_stock = pd.DataFrame({
    'stock_actual': np.random.randint(0, 200, n_prod),
    'demanda_semanal_prom': np.random.uniform(10, 100, n_prod),
    'demanda_semanal_std': np.random.uniform(2, 30, n_prod),
    'lead_time': np.random.choice([2, 3, 5, 7, 10], n_prod, p=[0.2, 0.3, 0.3, 0.1, 0.1]),
    'stock_seguridad': np.random.randint(5, 50, n_prod),
    'estacionalidad': np.random.choice([0, 1], n_prod, p=[0.6, 0.4])
})

# Probabilidad de stockout
z_score = (productos_stock['stock_actual'] - productos_stock['demanda_semanal_prom']) / \
          (productos_stock['demanda_semanal_std'].clip(lower=1))
prob_stockout = 1 / (1 + np.exp(-(-1.5 - 0.5 * z_score + 0.3 * productos_stock['lead_time'] -
                                    0.04 * productos_stock['stock_seguridad'] +
                                    0.8 * productos_stock['estacionalidad'])))
productos_stock['stockout'] = np.random.binomial(1, prob_stockout)

# Modelo
X_stock = productos_stock[['stock_actual', 'demanda_semanal_prom', 'demanda_semanal_std',
                            'lead_time', 'stock_seguridad', 'estacionalidad']]
y_stock = productos_stock['stockout']

modelo_stock = LogisticRegression(max_iter=1000)
modelo_stock.fit(X_stock, y_stock)

print("=== PREDICCIÓN DE STOCKOUT ===")
coef_stock = pd.DataFrame({
    'Factor': X_stock.columns,
    'β': modelo_stock.coef_[0],
    'OR': np.exp(modelo_stock.coef_[0])
})
print(coef_stock)

# Identificar productos en riesgo
productos_stock['prob_stockout_pred'] = modelo_stock.predict_proba(X_stock)[:, 1]
en_riesgo = productos_stock[productos_stock['prob_stockout_pred'] > 0.5].sort_values(
    'prob_stockout_pred', ascending=False
)
print(f"\nProductos en riesgo de stockout: {len(en_riesgo)} de {n_prod}")
print("\nTop 5 productos con mayor riesgo:")
print(en_riesgo[['stock_actual', 'demanda_semanal_prom', 'prob_stockout_pred']].head())
```

---

## 20.10 Regresión Logística Multinomial

Para más de 2 categorías:

```python
from sklearn.linear_model import LogisticRegression

# Ejemplo: Tipo de producto preferido (3 categorías)
datos_mult = pd.DataFrame({
    'edad': np.random.randint(18, 70, 500),
    'ingreso': np.random.normal(50000, 20000, 500),
    'genero': np.random.choice([0, 1], 500)
})

# 3 categorías: Electrónica, Ropa, Hogar
categorias = ['Electrónica', 'Ropa', 'Hogar']
log_odds_elec = -1 + 0.03 * datos_mult['edad'] + 0.00001 * datos_mult['ingreso']
log_odds_ropa = 0 - 0.02 * datos_mult['edad'] + 0.000005 * datos_mult['ingreso']
log_odds_hogar = 0  # categoría base

# Softmax
probs = np.column_stack([
    np.exp(log_odds_elec), 
    np.exp(log_odds_ropa), 
    np.exp(log_odds_hogar)
])
probs = probs / probs.sum(axis=1, keepdims=True)
datos_mult['categoria'] = [np.random.choice(3, p=probs[i]) for i in range(500)]
datos_mult['categoria_nombre'] = datos_mult['categoria'].map({0: 'Electrónica', 1: 'Ropa', 2: 'Hogar'})

# Modelo multinomial
modelo_multi = LogisticRegression(multi_class='multinomial', max_iter=1000)
modelo_multi.fit(datos_mult[['edad', 'ingreso', 'genero']], datos_mult['categoria'])

print("=== REGRESIÓN LOGÍSTICA MULTINOMIAL ===")
for i, cat in enumerate(categorias):
    print(f"\nCategoría: {cat}")
    for j, pred in enumerate(['edad', 'ingreso', 'genero']):
        print(f"  {pred}: β={modelo_multi.coef_[i][j]:.4f}")
```

---

## 20.11 Resumen

| Concepto | Descripción |
|----------|-------------|
| **Función Sigmoide** | Transforma combinación lineal a probabilidad [0,1] |
| **Odds Ratio** | $e^{\beta}$ - cambio en odds por unidad del predictor |
| **Log-Likelihood** | Medida de ajuste del modelo |
| **AUC-ROC** | Capacidad discriminante del modelo |
| **Precision/Recall** | Exactitud de predicciones positivas |
| **Umbral** | Punto de corte para clasificar (default 0.5) |

---

## Ejercicios Propuestos

1. **Ventas**: Crea un modelo logístico para predecir si un lead se convertirá en cliente. Evalúa con AUC-ROC
2. **Compras**: Modela la probabilidad de que un proveedor sea aprobado. Identifica los 3 factores más importantes
3. **Inventarios**: Predice qué productos tendrán stockout la próxima semana. Optimiza el umbral de decisión

---

[← Anterior](19-regresion-multiple.md) | [Índice](index.md) | [Siguiente →](21-series-temporales-intro.md)
