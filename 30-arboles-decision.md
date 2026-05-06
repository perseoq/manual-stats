# Capítulo 30: Árboles de Decisión

[← Anterior](29-clustering.md) | [Índice](index.md) | [Siguiente →](31-control-calidad.md)

---

## 30.1 Introducción

Los **árboles de decisión** son modelos de aprendizaje supervisado que particionan el espacio de predictores en regiones rectangulares, asignando una predicción a cada región. Son **interpretables** y manejan tanto clasificación como regresión.

### Ventajas
- ✅ Fáciles de interpretar y explicar
- ✅ Manejan variables numéricas y categóricas
- ✅ No requieren escalamiento de datos
- ✅ Capturan relaciones no lineales

### Desventajas
- ❌ Propensos a overfitting
- ❌ Sensibles a pequeñas variaciones en los datos
- ❌ Pueden ser inestables

---

## 30.2 Fundamentos

### Criterios de División

| Criterio | Fórmula | Tipo |
|----------|---------|------|
| **Gini** | $G = 1 - \sum_{i=1}^{k} p_i^2$ | Clasificación |
| **Entropía** | $H = -\sum_{i=1}^{k} p_i \log(p_i)$ | Clasificación |
| **MSE** | $\frac{1}{n}\sum (y_i - \bar{y})^2$ | Regresión |

### Ganancia de Información

$$Ganancia = Impureza_{padre} - \sum_{hijos} \frac{n_{hijo}}{n_{padre}} \times Impureza_{hijo}$$

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, mean_squared_error, r2_score)
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

np.random.seed(42)

# Datos: Aprobación de crédito para clientes
n = 1000
datos_credito = pd.DataFrame({
    'ingreso': np.random.normal(50000, 20000, n),
    'edad': np.random.randint(20, 70, n),
    'score_crediticio': np.random.randint(300, 850, n),
    'deuda_ingreso': np.random.uniform(0, 0.6, n),
    'empleado_anos': np.random.randint(0, 30, n),
    'historial_previo': np.random.choice([0, 1], n, p=[0.4, 0.6])
})

# Variable objetivo: ¿aprobado?
prob_aprob = (
    0.3 * (datos_credito['score_crediticio'] > 650) +
    0.2 * (datos_credito['deuda_ingreso'] < 0.3) +
    0.2 * (datos_credito['historial_previo'] == 1) +
    0.15 * (datos_credito['empleado_anos'] > 2) +
    0.15 * (datos_credito['ingreso'] > 40000)
)
datos_credito['aprobado'] = (prob_aprob + np.random.uniform(0, 0.3, n) > 0.5).astype(int)

print(f"Tasa de aprobación: {datos_credito['aprobado'].mean():.1%}")
```

---

## 30.3 Árbol de Clasificación

```python
# Preparar datos
predictores = ['ingreso', 'edad', 'score_crediticio', 'deuda_ingreso', 
               'empleado_anos', 'historial_previo']
X = datos_credito[predictores]
y = datos_credito['aprobado']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Árbol de decisión
arbol = DecisionTreeClassifier(max_depth=4, min_samples_split=20, random_state=42)
arbol.fit(X_train, y_train)

# Predicciones
y_pred = arbol.predict(X_test)
y_prob = arbol.predict_proba(X_test)[:, 1]

print("=== ÁRBOL DE CLASIFICACIÓN ===")
print(f"Profundidad: {arbol.get_depth()}")
print(f"Hojas: {arbol.get_n_leaves()}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"\nImportancia de predictores:")
for pred, imp in sorted(zip(predictores, arbol.feature_importances_), 
                         key=lambda x: x[1], reverse=True):
    print(f"  {pred}: {imp:.3f}")
```

### Visualización del Árbol

```python
fig, ax = plt.subplots(figsize=(20, 12))
plot_tree(arbol, feature_names=predictores, class_names=['Rechazar', 'Aprobar'],
          filled=True, rounded=True, fontsize=10, ax=ax)
ax.set_title('Árbol de Decisión - Aprobación de Crédito', fontweight='bold')
plt.savefig('arbol_decision.png', dpi=150, bbox_inches='tight')
plt.show()

# Reglas de decisión
print("\n=== REGLAS DE DECISIÓN ===")
from sklearn.tree import export_text
texto_reglas = export_text(arbol, feature_names=predictores)
print(texto_reglas)
```

---

## 30.4 Poda del Árbol (Pruning)

Controlar la complejidad para evitar overfitting:

```python
def evaluar_complejidad(X_train, y_train, X_test, y_test):
    """Evalúa accuracy vs profundidad del árbol"""
    profundidades = range(1, 21)
    train_scores = []
    test_scores = []
    
    for depth in profundidades:
        arbol = DecisionTreeClassifier(max_depth=depth, min_samples_split=10, random_state=42)
        arbol.fit(X_train, y_train)
        train_scores.append(accuracy_score(y_train, arbol.predict(X_train)))
        test_scores.append(accuracy_score(y_test, arbol.predict(X_test)))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(profundidades, train_scores, 'b-', linewidth=2, label='Train')
    ax.plot(profundidades, test_scores, 'r-', linewidth=2, label='Test')
    ax.axvline(np.argmax(test_scores) + 1, color='green', linestyle='--', 
               label=f'Mejor profundidad: {np.argmax(test_scores) + 1}')
    ax.set_xlabel('Profundidad máxima', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Overfitting: Accuracy vs Profundidad', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.savefig('complejidad_arbol.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    best_depth = np.argmax(test_scores) + 1
    print(f"Mejor profundidad: {best_depth}")
    print(f"Accuracy train (best): {train_scores[best_depth-1]:.4f}")
    print(f"Accuracy test (best): {test_scores[best_depth-1]:.4f}")
    
    return best_depth

best_depth = evaluar_complejidad(X_train, y_train, X_test, y_test)
```

---

## 30.5 Árbol de Regresión

```python
# Predicción de gasto del cliente
np.random.seed(42)

datos_gasto = pd.DataFrame({
    'ingreso': np.random.normal(50000, 20000, 500),
    'edad': np.random.randint(20, 70, 500),
    'hijos': np.random.poisson(1, 500),
    'score_credito': np.random.randint(400, 850, 500)
})

# Gasto anual (variable continua)
datos_gasto['gasto_anual'] = (
    2000 +
    0.05 * datos_gasto['ingreso'] +
    50 * datos_gasto['edad'] +
    -200 * datos_gasto['hijos'] +
    10 * datos_gasto['score_credito'] +
    np.random.normal(0, 2000, 500)
)

X_gasto = datos_gasto[['ingreso', 'edad', 'hijos', 'score_credito']]
y_gasto = datos_gasto['gasto_anual']

Xg_train, Xg_test, yg_train, yg_test = train_test_split(X_gasto, y_gasto, test_size=0.3, random_state=42)

# Árbol de regresión
arbol_reg = DecisionTreeRegressor(max_depth=5, min_samples_split=20, random_state=42)
arbol_reg.fit(Xg_train, yg_train)

yg_pred = arbol_reg.predict(Xg_test)

print("=== ÁRBOL DE REGRESIÓN ===")
print(f"R²: {r2_score(yg_test, yg_pred):.4f}")
print(f"RMSE: ${np.sqrt(mean_squared_error(yg_test, yg_pred)):,.0f}")
print(f"\nImportancia de predictores:")
for pred, imp in sorted(zip(X_gasto.columns, arbol_reg.feature_importances_),
                         key=lambda x: x[1], reverse=True):
    print(f"  {pred}: {imp:.3f}")
```

---

## 30.6 Random Forest

Ensemble de múltiples árboles para mejorar precisión y reducir overfitting:

```python
from sklearn.ensemble import RandomForestClassifier

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_split=20,
                             random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Comparación con árbol simple
arbol_simple = DecisionTreeClassifier(max_depth=6, min_samples_split=20, random_state=42)
arbol_simple.fit(X_train, y_train)
y_pred_simple = arbol_simple.predict(X_test)

print("=== COMPARACIÓN: ÁRBOL vs RANDOM FOREST ===")
print(f"{'Métrica':20s} {'Árbol Simple':>15s} {'Random Forest':>15s}")
print("-" * 50)
print(f"{'Accuracy':20s} {accuracy_score(y_test, y_pred_simple):>15.4f} {accuracy_score(y_test, y_pred_rf):>15.4f}")

# Importancia de variables (más estable que árbol simple)
print(f"\nImportancia de variables (Random Forest):")
for pred, imp in sorted(zip(predictores, rf.feature_importances_),
                         key=lambda x: x[1], reverse=True):
    print(f"  {pred}: {imp:.3f}")
```

---

## 30.7 Validación Cruzada

```python
# Validación cruzada para Random Forest
scores_rf = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
scores_tree = cross_val_score(arbol_simple, X, y, cv=5, scoring='accuracy')

print("=== VALIDACIÓN CRUZADA (5-FOLD) ===")
print(f"Árbol Simple: {scores_tree.mean():.4f} ± {scores_tree.std():.4f}")
print(f"Random Forest: {scores_rf.mean():.4f} ± {scores_rf.std():.4f}")
```

---

## 30.8 Ejemplo en Ventas: Predicción de Compra

```python
# Predecir si un cliente realizará una compra
np.random.seed(42)
n = 800

datos_compra = pd.DataFrame({
    'visitas_sitio': np.random.poisson(5, n),
    'tiempo_sitio_min': np.random.exponential(15, n),
    'email_abiertos': np.random.uniform(0, 1, n),
    'descuento_usado': np.random.choice([0, 1], n, p=[0.6, 0.4]),
    'visitas_pasadas': np.random.poisson(10, n),
    'dispositivo': np.random.choice([0, 1, 2], n)  # 0: mobile, 1: desktop, 2: tablet
})

# Regla de decisión subyacente
prob_compra = (
    0.25 * (datos_compra['visitas_sitio'] > 3) +
    0.20 * (datos_compra['email_abiertos'] > 0.5) +
    0.20 * (datos_compra['descuento_usado'] == 1) +
    0.20 * (datos_compra['visitas_pasadas'] > 5) +
    0.15 * (datos_compra['tiempo_sitio_min'] > 10)
) / 1.0 + np.random.uniform(-0.2, 0.2, n)

datos_compra['compro'] = (prob_compra > 0.5).astype(int)

X_compra = datos_compra.drop('compro', axis=1)
y_compra = datos_compra['compro']

# Árbol de decisión
arbol_compra = DecisionTreeClassifier(max_depth=4, min_samples_split=30, random_state=42)
arbol_compra.fit(X_compra, y_compra)

print("=== PREDICCIÓN DE COMPRA (ÁRBOL) ===")
for pred, imp in sorted(zip(X_compra.columns, arbol_compra.feature_importances_),
                         key=lambda x: x[1], reverse=True):
    print(f"  {pred}: {imp:.3f}")

# Reglas para segmentación
print(f"\nReglas del árbol:")
print(export_text(arbol_compra, feature_names=list(X_compra.columns)))
```

---

## 30.9 Ejemplo en Compras: Selección de Proveedores

```python
# Árbol para clasificar proveedores
np.random.seed(42)
n = 200

proveedores_tree = pd.DataFrame({
    'precio_relativo': np.random.uniform(0.7, 1.3, n),
    'calidad_promedio': np.random.uniform(3, 5, n),
    'lead_time': np.random.exponential(5, n) + 2,
    'confiabilidad': np.random.uniform(0.6, 1.0, n),
    'distancia': np.random.uniform(10, 500, n)
})

# Clasificación: 0=Rechazar, 1=Condicional, 2=Aprobar
score_prov = (
    0.3 * (proveedores_tree['calidad_promedio'] > 4) +
    0.25 * (proveedores_tree['confiabilidad'] > 0.85) +
    0.25 * (proveedores_tree['precio_relativo'] < 1.0) +
    0.20 * (proveedores_tree['lead_time'] < 7)
)
proveedores_tree['decision'] = pd.cut(score_prov + np.random.uniform(-0.2, 0.2, n),
                                       bins=[0, 0.4, 0.7, 1.0],
                                       labels=[0, 1, 2]).astype(int)

X_prov = proveedores_tree.drop('decision', axis=1)
y_prov = proveedores_tree['decision']

arbol_prov = DecisionTreeClassifier(max_depth=4, min_samples_split=15, random_state=42)
arbol_prov.fit(X_prov, y_prov)

print("=== CLASIFICACIÓN DE PROVEEDORES ===")
print(f"Accuracy: {arbol_prov.score(X_prov, y_prov):.3f}")
print(f"Reglas de decisión:")
print(export_text(arbol_prov, feature_names=list(X_prov.columns)))
```

---

## 30.10 Ejemplo en Inventarios: Clasificación ABC Automática

```python
# Árbol para clasificación ABC de inventario
np.random.seed(42)
n = 300

productos_abc = pd.DataFrame({
    'valor_anual': np.random.gamma(3, 50000, n),
    'rotacion': np.random.exponential(15, n) + 1,
    'volatilidad': np.random.exponential(0.3, n),
    'costo_stockout': np.random.uniform(100, 5000, n),
    'lead_time': np.random.choice([2, 3, 5, 7, 10], n)
})

# Clasificación ABC (80-15-5)
valor_ordenado = np.sort(productos_abc['valor_anual'])[::-1]
pct_acum = np.cumsum(valor_ordenado) / valor_ordenado.sum()
productos_abc['clase'] = np.select(
    [pct_acum <= 0.80, pct_acum <= 0.95],
    ['A', 'B'],
    default='C'
)

# Árbol para aprender la clasificación
le = LabelEncoder()
y_abc = le.fit_transform(productos_abc['clase'])

X_abc = productos_abc[['valor_anual', 'rotacion', 'volatilidad', 'costo_stockout', 'lead_time']]

arbol_abc = DecisionTreeClassifier(max_depth=3, random_state=42)
arbol_abc.fit(X_abc, y_abc)

print("=== CLASIFICACIÓN ABC AUTOMÁTICA ===")
print(f"Accuracy: {arbol_abc.score(X_abc, y_abc):.3f}")
print(f"Importancia de variables:")
for pred, imp in sorted(zip(X_abc.columns, arbol_abc.feature_importances_),
                         key=lambda x: x[1], reverse=True):
    print(f"  {pred}: {imp:.3f}")

print(f"\nReglas del árbol:")
print(export_text(arbol_abc, feature_names=list(X_abc.columns), 
                  class_names=list(le.classes_)))
```

---

## 30.11 Resumen

| Parámetro | Efecto | Valor típico |
|-----------|--------|-------------|
| **max_depth** | Controla profundidad (overfitting) | 3-10 |
| **min_samples_split** | Mínimo para dividir un nodo | 20-50 |
| **min_samples_leaf** | Mínimo en hoja | 10-30 |
| **max_features** | Variables consideradas en cada split | sqrt(n) |
| **n_estimators** (RF) | Número de árboles | 100-1000 |

### Random Forest vs Árbol Simple
| Aspecto | Árbol Simple | Random Forest |
|---------|-------------|---------------|
| Varianza | Alta | Baja |
| Sesgo | Bajo | Ligeramente mayor |
| Interpretabilidad | Alta | Media |
| Precisión | Menor | Mayor |

---

## Ejercicios Propuestos

1. **Ventas**: Crea un árbol de decisión para predecir si un visitante comprará. Identifica las 3 variables más importantes
2. **Compras**: Usa Random Forest para clasificar proveedores en 3 niveles. Compara con árbol simple
3. **Inventarios**: Implementa un árbol de regresión para predecir el stock de seguridad necesario basado en demanda, lead time y volatilidad

---

[← Anterior](29-clustering.md) | [Índice](index.md) | [Siguiente →](31-control-calidad.md)
