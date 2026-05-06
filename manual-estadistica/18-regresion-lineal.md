# Capítulo 18: Regresión Lineal Simple

[← Anterior](17-correlacion.md) | [Índice](index.md) | [Siguiente →](19-regresion-multiple.md)

---

## 18.1 Introducción

La **regresión lineal simple** modela la relación entre una variable dependiente (Y) y una variable independiente (X) mediante una línea recta.

$$Y = \beta_0 + \beta_1 X + \varepsilon$$

Donde:
- $\beta_0$: **Intersección** (valor de Y cuando X = 0)
- $\beta_1$: **Pendiente** (cambio en Y por unidad de cambio en X)
- $\varepsilon$: **Error aleatorio** (lo que no explica el modelo)

---

## 18.2 Método de Mínimos Cuadrados Ordinarios (MCO)

Minimiza la suma de cuadrados de los residuos:

$$\min_{\beta_0, \beta_1} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

### Solución
$$\hat{\beta}_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}$$
$$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$$

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

np.random.seed(42)

# --- Implementación Manual ---
class RegresionLinealManual:
    """Regresión lineal simple implementada desde cero"""
    
    def fit(self, X, y):
        n = len(X)
        self.x_mean = np.mean(X)
        self.y_mean = np.mean(y)
        
        # Pendiente
        numerador = np.sum((X - self.x_mean) * (y - self.y_mean))
        denominador = np.sum((X - self.x_mean)**2)
        self.beta_1 = numerador / denominador
        
        # Intersección
        self.beta_0 = self.y_mean - self.beta_1 * self.x_mean
        
        # Predicciones
        y_pred = self.predict(X)
        residuos = y - y_pred
        
        # Métricas
        n = len(X)
        self.sse = np.sum(residuos**2)
        self.sst = np.sum((y - self.y_mean)**2)
        self.ssr = self.sst - self.sse
        self.r2 = 1 - self.sse / self.sst
        self.rmse = np.sqrt(self.sse / n)
        self.mae = np.mean(np.abs(residuos))
        
        # Error estándar de los coeficientes
        sigma2 = self.sse / (n - 2)
        self.se_beta_1 = np.sqrt(sigma2 / denominador)
        self.se_beta_0 = np.sqrt(sigma2 * (1/n + self.x_mean**2 / denominador))
        
        # Estadísticos t
        self.t_beta_1 = self.beta_1 / self.se_beta_1
        self.t_beta_0 = self.beta_0 / self.se_beta_0
        self.p_beta_1 = 2 * (1 - stats.t.cdf(abs(self.t_beta_1), n-2))
        self.p_beta_0 = 2 * (1 - stats.t.cdf(abs(self.t_beta_0), n-2))
        
        return self
    
    def predict(self, X):
        return self.beta_0 + self.beta_1 * X
    
    def summary(self):
        print("=== REGRESIÓN LINEAL SIMPLE (MANUAL) ===")
        print(f"{'':20s} {'Coef':>10s} {'Std Err':>10s} {'t':>10s} {'P>|t|':>10s}")
        print("-" * 60)
        print(f"Constante (β₀): {self.beta_0:>10.4f} {self.se_beta_0:>10.4f} {self.t_beta_0:>10.3f} {self.p_beta_0:>10.6f}")
        print(f"Pendiente (β₁): {self.beta_1:>10.4f} {self.se_beta_1:>10.4f} {self.t_beta_1:>10.3f} {self.p_beta_1:>10.6f}")
        print(f"\nR² = {self.r2:.4f}")
        print(f"RMSE = {self.rmse:.4f}")
        print(f"MAE = {self.mae:.4f}")
        print(f"SSE = {self.sse:.4f}")
        print(f"Ecuación: Y = {self.beta_0:.4f} + {self.beta_1:.4f} × X")

# Datos: Publicidad vs Ventas
publicidad = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65])
ventas = np.array([120, 150, 180, 200, 240, 260, 290, 310, 350, 370, 400, 430])

modelo_manual = RegresionLinealManual()
modelo_manual.fit(publicidad, ventas)
modelo_manual.summary()

### Interpretación de Negocio

Los coeficientes resultantes tienen una lectura directa para la toma de decisiones: la pendiente $\beta_1 = 5.96$ indica que por cada \$1,000 adicionales invertidos en publicidad, las ventas aumentan en promedio \$5,960. La intersección $\beta_0 = 69.59$ representa las ventas base cuando no hay inversión publicitaria (aunque en la práctica esto puede no tener sentido si la inversión nunca es cero). El $R^2 = 0.993$ señala que el modelo explica el 99.3% de la variabilidad en ventas, un ajuste excelente. El p-valor de la pendiente ($p < 0.001$) confirma que la relación es estadísticamente significativa. Desde una perspectiva gerencial, este modelo permite simular escenarios: "si aumentamos la inversión publicitaria en \$5,000, esperamos un incremento en ventas de aproximadamente \$29,800".
```

---

## 18.3 Regresión con StatsModels

```python
# Usando StatsModels (producción)
X = sm.add_constant(publicidad)
modelo_sm = sm.OLS(ventas, X).fit()

print("=== REGRESIÓN LINEAL - STATSMODELS ===")
print(modelo_sm.summary())

# Extraer métricas clave
print(f"\n=== MÉTRICAS CLAVE ===")
print(f"R²: {modelo_sm.rsquared:.4f}")
print(f"R² ajustado: {modelo_sm.rsquared_adj:.4f}")
print(f"F-statistic: {modelo_sm.fvalue:.3f}")
print(f"p-valor (F): {modelo_sm.f_pvalue:.6f}")
print(f"AIC: {modelo_sm.aic:.1f}")
print(f"BIC: {modelo_sm.bic:.1f}")
```

---

## 18.4 Interpretación de Resultados

```python
# Interpretación completa del modelo
print("=== INTERPRETACIÓN DEL MODELO ===")
print(f"Ecuación: Ventas = {modelo_manual.beta_0:.2f} + {modelo_manual.beta_1:.2f} × Publicidad")
print(f"\n→ Por cada $1,000 adicionales en publicidad,")
print(f"  las ventas aumentan en ${modelo_manual.beta_1:.2f}")
print(f"\n→ Cuando no hay publicidad (X=0),")
print(f"  las ventas base son ${modelo_manual.beta_0:.2f}")
print(f"\n→ El modelo explica el {modelo_manual.r2*100:.1f}% de la variabilidad en ventas")
print(f"  (R² = {modelo_manual.r2:.4f})")
print(f"\n→ La publicidad es un predictor {'SIGNIFICATIVO' if modelo_manual.p_beta_1 < 0.05 else 'NO significativo'}")
print(f"  (p = {modelo_manual.p_beta_1:.6f})")

# Predicción
nueva_inversion = 75
venta_estimada = modelo_manual.predict(np.array([nueva_inversion]))
print(f"\n→ Con inversión de ${nueva_inversion}K:")
print(f"  Ventas estimadas: ${venta_estimada[0]:.2f}")
```

---

## 18.5 Diagnóstico de Residuos

### Supuestos del Modelo

1. **Linealidad**: Relación lineal entre X e Y
2. **Independencia**: Residuos independientes
3. **Homocedasticidad**: Varianza constante de residuos
4. **Normalidad**: Residuos distribuyen Normal(0, σ²)

```python
def diagnosticar_modelo(modelo, X, y, titulo="Modelo"):
    """Diagnóstico completo de regresión lineal"""
    y_pred = modelo.predict(X) if hasattr(modelo, 'predict') else modelo.fittedvalues
    residuos = y - y_pred
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Residuos vs Ajustados (Homocedasticidad)
    axes[0, 0].scatter(y_pred, residuos, alpha=0.6, edgecolors='white')
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_xlabel('Valores Ajustados')
    axes[0, 0].set_ylabel('Residuos')
    axes[0, 0].set_title('Residuos vs Ajustados', fontweight='bold')
    
    # 2. Q-Q plot (Normalidad)
    stats.probplot(residuos, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot (Normalidad)', fontweight='bold')
    
    # 3. Histograma de residuos
    axes[0, 2].hist(residuos, bins=20, density=True, alpha=0.7, 
                    color='steelblue', edgecolor='white')
    x_norm = np.linspace(residuos.min(), residuos.max(), 100)
    axes[0, 2].plot(x_norm, stats.norm.pdf(x_norm, np.mean(residuos), np.std(residuos)),
                    'r-', linewidth=2)
    axes[0, 2].set_title('Histograma de Residuos', fontweight='bold')
    
    # 4. Residuos vs Orden (Independencia)
    axes[1, 0].plot(residuos, 'o-', alpha=0.6, markersize=4)
    axes[1, 0].axhline(y=0, color='red', linestyle='--')
    axes[1, 0].set_xlabel('Orden de observación')
    axes[1, 0].set_ylabel('Residuos')
    axes[1, 0].set_title('Residuos vs Orden', fontweight='bold')
    
    # 5. Residuos vs X (Linealidad)
    axes[1, 1].scatter(X if len(X.shape)==1 else X[:,1], residuos, alpha=0.6, edgecolors='white')
    axes[1, 1].axhline(y=0, color='red', linestyle='--')
    axes[1, 1].set_xlabel('Variable X')
    axes[1, 1].set_ylabel('Residuos')
    axes[1, 1].set_title('Residuos vs X', fontweight='bold')
    
    # 6. Texto con métricas
    axes[1, 2].text(0.1, 0.9, f'R² = {r2_score(y, y_pred):.4f}', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.8, f'RMSE = {np.sqrt(mean_squared_error(y, y_pred)):.4f}', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.7, f'MAE = {mean_absolute_error(y, y_pred):.4f}', fontsize=12, transform=axes[1, 2].transAxes)
    
    # Pruebas estadísticas
    _, p_norm = stats.shapiro(residuos)
    _, p_het = sm.stats.diagnostic.het_breuschpagan(residuos, sm.add_constant(np.arange(len(residuos))))
    axes[1, 2].text(0.1, 0.5, f'Shapiro (Normal): p={p_norm:.4f}', fontsize=10, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.4, f'Breusch-Pagan (Homoc): p={p_het[1]:.4f}', fontsize=10, transform=axes[1, 2].transAxes)
    axes[1, 2].axis('off')
    
    fig.suptitle(f'Diagnóstico de Residuos - {titulo}', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'diagnostico_{titulo.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n=== PRUEBAS DE DIAGNÓSTICO ===")
    print(f"Shapiro-Wilk (Normalidad): p = {p_norm:.4f}")
    print(f"Breusch-Pagan (Homocedasticidad): p = {p_het[1]:.4f}")
    print(f"Durbin-Watson (Independencia): {sm.stats.durbin_watson(residuos):.4f}")
    if p_norm > 0.05: print("✓ Residuos normales")
    else: print("✗ Residuos NO normales")
    if p_het[1] > 0.05: print("✓ Homocedasticidad")
    else: print("✗ Heterocedasticidad presente")

diagnosticar_modelo(modelo_sm, publicidad, ventas, "Publicidad → Ventas")
```

---

## 18.6 Ejemplo en Ventas: Pronóstico con Regresión

```python
# Pronóstico de ventas basado en inversión publicitaria
np.random.seed(42)

datos_ventas = pd.DataFrame({
    'mes': pd.date_range('2024-01-01', periods=24, freq='ME'),
    'publicidad': np.random.uniform(20, 100, 24),
})
datos_ventas['ventas'] = (500 + 15 * datos_ventas['publicidad'] + 
                           np.random.normal(0, 50, 24))

# Modelo
X = sm.add_constant(datos_ventas['publicidad'])
modelo_ventas = sm.OLS(datos_ventas['ventas'], X).fit()

# Predicción para el próximo mes
nueva_pub = 80
X_nuevo = sm.add_constant([nueva_pub])
pred = modelo_ventas.get_prediction(X_nuevo)
pred_summary = pred.summary_frame(alpha=0.05)

print("=== PRONÓSTICO DE VENTAS ===")
print(f"Inversión planeada: ${nueva_pub}K")
print(f"Ventas estimadas: ${pred_summary['mean'].values[0]:.0f}")
print(f"IC 95%: [${pred_summary['mean_ci_lower'].values[0]:.0f}, "
      f"${pred_summary['mean_ci_upper'].values[0]:.0f}]")
print(f"IC predicción 95%: [${pred_summary['obs_ci_lower'].values[0]:.0f}, "
      f"${pred_summary['obs_ci_upper'].values[0]:.0f}]")
```

---

## 18.7 Ejemplo en Compras: Costo vs Volumen

```python
# Modelo de costo en función del volumen de compra
np.random.seed(42)
volumen = np.random.randint(100, 5000, 30)
# Costo unitario disminuye con el volumen (economías de escala)
costo_unitario = 100 - 0.008 * volumen + np.random.normal(0, 5, 30)
costo_unitario = np.clip(costo_unitario, 50, 110)

# Costo total
costo_total = volumen * costo_unitario

print("=== REGRESIÓN: COSTO TOTAL vs VOLUMEN ===")
X_vol = sm.add_constant(volumen)
modelo_costo = sm.OLS(costo_total, X_vol).fit()
print(modelo_costo.summary().tables[1])

print(f"\nEcuación: Costo Total = {modelo_costo.params[0]:.2f} + "
      f"{modelo_costo.params[1]:.4f} × Volumen")
print(f"R² = {modelo_costo.rsquared:.4f}")

# Interpretación
print(f"\n→ Costo fijo estimado: ${modelo_costo.params[0]:.2f}")
print(f"→ Costo variable por unidad: ${modelo_costo.params[1]:.4f}")
print(f"→ Economías de escala: {'Presentes' if modelo_costo.params[1] < 0 else 'No detectadas'}")
```

---

## 18.8 Ejemplo en Inventarios: Demanda vs Precio

```python
# Elasticidad precio de la demanda
np.random.seed(42)
precio = np.random.uniform(20, 100, 50)
# A mayor precio, menor demanda (elasticidad negativa)
demanda = 500 - 3 * precio + np.random.normal(0, 30, 50)
demanda = np.clip(demanda, 50, 600)

X_precio = sm.add_constant(precio)
modelo_demanda = sm.OLS(demanda, X_precio).fit()

print("=== REGRESIÓN: DEMANDA vs PRECIO ===")
print(f"Ecuación: Demanda = {modelo_demanda.params[0]:.2f} + "
      f"{modelo_demanda.params[1]:.2f} × Precio")
print(f"R² = {modelo_demanda.rsquared:.4f}")

# Elasticidad precio
elasticidad = modelo_demanda.params[1] * np.mean(precio) / np.mean(demanda)
print(f"\nElasticidad precio de la demanda: {elasticidad:.4f}")
print(f"Interpretación: Por cada 1% de aumento en precio,")
print(f"la demanda se reduce un {abs(elasticidad):.2f}%")
print(f"Demanda: {'Elástica' if abs(elasticidad) > 1 else 'Inelástica'}")

# Precio óptimo (simplificado)
print(f"\n→ Si el costo unitario es $20:")
costo_var = 20
# Ingreso marginal vs costo marginal (simplificado)
print(f"  Precio que maximiza ingreso: depende de la elasticidad")
```

---

## 18.9 Transformaciones (Log-Log, Log-Lin)

Para relaciones no lineales, podemos transformar variables:

### Modelo Log-Log (Elasticidad constante)
$$\ln(Y) = \beta_0 + \beta_1 \ln(X)$$

### Modelo Log-Lin (Crecimiento porcentual constante)
$$\ln(Y) = \beta_0 + \beta_1 X$$

```python
# Relación no lineal: ventas crecen con publicidad pero con rendimientos decrecientes
publicidad2 = np.linspace(1, 100, 50)
ventas2 = 200 + 50 * np.log(publicidad2) + np.random.normal(0, 10, 50)

# Modelo lineal simple (subóptimo)
X_lin = sm.add_constant(publicidad2)
modelo_lin = sm.OLS(ventas2, X_lin).fit()

# Modelo log-log
log_pub = np.log(publicidad2)
X_log = sm.add_constant(log_pub)
modelo_log = sm.OLS(ventas2, X_log).fit()

print("=== COMPARACIÓN: LINEAL vs LOG-LOG ===")
print(f"Modelo lineal: R² = {modelo_lin.rsquared:.4f}, AIC = {modelo_lin.aic:.1f}")
print(f"Modelo log-log: R² = {modelo_log.rsquared:.4f}, AIC = {modelo_log.aic:.1f}")
print(f"\n→ El modelo {'log-log' if modelo_log.rsquared > modelo_lin.rsquared else 'lineal'}")
print(f"  explica mejor la relación (mayor R², menor AIC)")
```

---

## 18.10 Resumen

| Componente | Interpretación |
|------------|---------------|
| **β₀** | Valor de Y cuando X = 0 (base) |
| **β₁** | Cambio en Y por unidad de cambio en X |
| **R²** | Proporción de varianza explicada |
| **p-valor** | Significancia del coeficiente |
| **RMSE** | Error típico de predicción |
| **Residuos** | Deben ser N(0, σ²) independientes y homocedásticos |

---

## Ejercicios Propuestos

1. **Ventas**: Crea un modelo de regresión para pronosticar ventas basado en el número de visitas a la tienda
2. **Compras**: Modela la relación entre volumen de compra y costo unitario. ¿Hay economías de escala?
3. **Inventarios**: Estima la demanda en función del precio. Calcula la elasticidad precio de la demanda

---

[← Anterior](17-correlacion.md) | [Índice](index.md) | [Siguiente →](19-regresion-multiple.md)
