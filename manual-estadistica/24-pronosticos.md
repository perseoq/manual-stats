# Capítulo 24: Pronósticos de Ventas

[← Anterior](23-suavizacion-exponencial.md) | [Índice](index.md) | [Siguiente →](25-estadistica-bayesiana.md)

---

## 24.1 Introducción

El **pronóstico de ventas** es una de las aplicaciones más valiosas de la estadística en negocios. Combina múltiples técnicas para predecir la demanda futura y optimizar decisiones de inventario, producción y fuerza de ventas.

### Horizonte de Pronóstico

| Horizonte | Período | Uso |
|-----------|---------|-----|
| **Corto plazo** | Días - Semanas | Inventarios, reposición |
| **Medio plazo** | Meses - Trimestres | Presupuestos, producción |
| **Largo plazo** | Años | Estrategia, capacidad |

---

## 24.2 Métodos de Pronóstico

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt, SimpleExpSmoothing
from sklearn.metrics import (mean_squared_error, mean_absolute_error, 
                             mean_absolute_percentage_error, r2_score)
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Datos: Ventas mensuales con tendencia + estacionalidad + ruido
n = 60  # 5 años
meses = pd.date_range('2020-01-01', periods=n, freq='ME')
t = np.arange(n)

ventas = (5000 + 
          20 * t +  # tendencia
          200 * np.sin(2 * np.pi * t / 12) +  # estacionalidad anual
          50 * np.random.normal(0, 1, n))  # ruido

df = pd.DataFrame({'fecha': meses, 'ventas': ventas})
df = df.set_index('fecha')

print("=== DATOS DE VENTAS MENSUALES ===")
print(df.head())
print(f"\nRango: {df.index[0].strftime('%Y-%m')} a {df.index[-1].strftime('%Y-%m')}")
print(f"Ventas: media={df['ventas'].mean():.0f}, min={df['ventas'].min():.0f}, max={df['ventas'].max():.0f}")
```

## 24.3 Método 1: Promedio Móvil

```python
def pronostico_promedio_movil(serie, ventana, pasos=12):
    """Pronóstico por promedio móvil"""
    historial = serie.copy()
    predicciones = []
    
    for _ in range(pasos):
        pred = historial.iloc[-ventana:].mean()
        predicciones.append(pred)
        # Agregamos ruido pequeño para simular incertidumbre
        historial = pd.concat([historial, pd.Series([pred + np.random.normal(0, 50)])])
    
    return np.array(predicciones)

```

### ¿Por qué usar este método?

El **promedio móvil** es el método más simple de pronóstico. Consiste en calcular el promedio de los últimos k períodos y usar ese valor como predicción para el siguiente. Su principal ventaja es que filtra el ruido aleatorio y suaviza fluctuaciones a corto plazo. Sin embargo, tiene limitaciones importantes: no captura tendencias (siempre va rezagado respecto a la serie), ignora la estacionalidad y asigna el mismo peso a todas las observaciones de la ventana. Es útil como referencia o baseline contra el cual comparar métodos más sofisticados. En la práctica, nadie usa el promedio móvil como método único de pronóstico, pero es valioso para detectar rápidamente la dirección general de la serie.

```python
# Promedio móvil de 3 y 12 meses
pm_3 = df['ventas'].rolling(3).mean()
pm_12 = df['ventas'].rolling(12).mean()

print("=== PRONÓSTICO POR PROMEDIO MÓVIL ===")
print(f"Último PM(3): {pm_3.iloc[-1]:.0f}")
print(f"Último PM(12): {pm_12.iloc[-1]:.0f}")
```

---

## 24.4 Método 2: Regresión Lineal con Tendencia

### ¿Por qué usar este método?

La **regresión lineal con tendencia temporal** modela las ventas como una función lineal del tiempo: $Ventas_t = \beta_0 + \beta_1 \times t + \varepsilon_t$. La pendiente $\beta_1$ representa el cambio esperado en ventas por cada unidad de tiempo (día, mes, año). Este método captura la tendencia de largo plazo y proporciona una ecuación interpretable que la gerencia puede entender fácilmente. Sin embargo, ignora por completo la estacionalidad y los ciclos, por lo que suele combinarse con otros métodos o usarse solo para datos desestacionalizados. Es especialmente útil para pronósticos de medio a largo plazo donde la tendencia domina sobre las fluctuaciones estacionales.

```python
# Modelo de regresión con tendencia temporal
X_trend = np.arange(len(df)).reshape(-1, 1)
y = df['ventas'].values

modelo_trend = LinearRegression()
modelo_trend.fit(X_trend, y)

# Proyectar 12 meses
X_future = np.arange(len(df), len(df) + 12).reshape(-1, 1)
pred_trend = modelo_trend.predict(X_future)

print("=== PRONÓSTICO POR REGRESIÓN LINEAL ===")
print(f"Tendencia: +${modelo_trend.coef_[0]:.2f}/mes")
print(f"Ventas base: ${modelo_trend.intercept_:.0f}")
```

---

## 24.5 Método 3: Holt-Winters

```python
modelo_hw = ExponentialSmoothing(
    df['ventas'],
    trend='add',
    seasonal='add',
    seasonal_periods=12
).fit()

# Pronóstico 12 meses
pred_hw = modelo_hw.forecast(12)

print("=== PRONÓSTICO HOLT-WINTERS (12 MESES) ===")
print(f"α (nivel): {modelo_hw.params['smoothing_level']:.4f}")
print(f"β (tendencia): {modelo_hw.params['smoothing_trend']:.4f}")
print(f"γ (estacionalidad): {modelo_hw.params['smoothing_seasonal']:.4f}")

fig, ax = plt.subplots(figsize=(14, 6))
df['ventas'].plot(ax=ax, color='steelblue', linewidth=1.5, label='Histórico')
modelo_hw.fittedvalues.plot(ax=ax, color='green', linewidth=1.5, label='Ajustado')
pred_hw.plot(ax=ax, color='red', linewidth=2, label='Pronóstico HW', marker='o')
ax.set_title('Pronóstico Holt-Winters - Ventas Mensuales', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.savefig('pronostico_hw_ventas.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 24.6 Método 4: ARIMA

```python
from statsmodels.tsa.arima.model import ARIMA

# Encontrar mejor ARIMA
best_aic = np.inf
best_order = None

for p in range(3):
    for d in range(2):
        for q in range(3):
            try:
                modelo = ARIMA(df['ventas'], order=(p, d, q))
                resultado = modelo.fit()
                if resultado.aic < best_aic:
                    best_aic = resultado.aic
                    best_order = (p, d, q)
            except:
                continue

print(f"=== ARIMA ÓPTIMO: {best_order} (AIC={best_aic:.2f}) ===")

modelo_arima = ARIMA(df['ventas'], order=best_order).fit()
pred_arima = modelo_arima.forecast(12)
```

---

## 24.7 Comparación de Métodos

```python
def evaluar_pronostico(serie, test_size=12):
    """Evalúa diferentes métodos de pronóstico"""
    train = serie.iloc[:-test_size]
    test = serie.iloc[-test_size:]
    
    resultados = {}
    
    # 1. Promedio histórico
    pred_hist = np.full(test_size, train.mean())
    resultados['Histórico'] = pred_hist
    
    # 2. Holt-Winters
    try:
        hw = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12).fit()
        resultados['Holt-Winters'] = hw.forecast(test_size).values
    except:
        resultados['Holt-Winters'] = np.full(test_size, train.mean())
    
    # 3. ARIMA
    try:
        arima = ARIMA(train, order=(1, 1, 1)).fit()
        resultados['ARIMA'] = arima.forecast(test_size).values
    except:
        resultados['ARIMA'] = np.full(test_size, train.mean())
    
    # 4. Último valor (naive)
    resultados['Naive'] = np.full(test_size, train.iloc[-1])
    
    # Calcular errores
    metricas = []
    for nombre, pred in resultados.items():
        rmse = np.sqrt(mean_squared_error(test, pred))
        mae = mean_absolute_error(test, pred)
        mape = mean_absolute_percentage_error(test, pred) * 100
        metricas.append({'Método': nombre, 'RMSE': rmse, 'MAE': mae, 'MAPE(%)': mape})
    
    return pd.DataFrame(metricas).sort_values('RMSE')

comparacion = evaluar_pronostico(df['ventas'], test_size=12)
print("=== COMPARACIÓN DE MÉTODOS DE PRONÓSTICO ===")
print(comparacion.to_string(index=False))
```

---

## 24.8 Pronóstico con Intervalos de Confianza

```python
def pronostico_con_ic(modelo, pasos=12, nivel=0.95):
    """Genera pronóstico con intervalos de confianza"""
    # Para modelo Holt-Winters
    if hasattr(modelo, 'forecast'):
        pred = modelo.forecast(pasos)
        residuos = modelo.resid
        
        # IC basado en residuos históricos
        z = stats.norm.ppf(1 - (1 - nivel) / 2)
        sigma = residuos.std()
        
        ic_inf = pred - z * sigma * np.sqrt(np.arange(1, pasos + 1))
        ic_sup = pred + z * sigma * np.sqrt(np.arange(1, pasos + 1))
        
        return pred, ic_inf, ic_sup

pred_hw, ic_inf, ic_sup = pronostico_con_ic(modelo_hw, 12)

print("=== PRONÓSTICO CON INTERVALOS DE CONFIANZA ===")
print(f"{'Mes':>6s} {'Pronóstico':>10s} {'IC Inf':>10s} {'IC Sup':>10s}")
print("-" * 38)
for i in range(12):
    print(f"{i+1:>6d} ${pred_hw.iloc[i]:>7,.0f} ${ic_inf[i]:>7,.0f} ${ic_sup[i]:>7,.0f}")
```

---

## 24.9 Ejemplo: Presupuesto Anual de Ventas

```python
# Construcción de presupuesto anual basado en pronóstico
np.random.seed(42)

# Ventas de año actual
ventas_actual = df['ventas'].values
ventas_totales_actual = ventas_actual.sum()

# Pronóstico para próximo año
modelo_presupuesto = ExponentialSmoothing(
    df['ventas'],
    trend='add',
    seasonal='add',
    seasonal_periods=12
).fit()

pronostico_anual = modelo_presupuesto.forecast(12)

print("=== PRESUPUESTO ANUAL DE VENTAS ===")
print(f"{'Mes':>10s} {'Pronóstico':>12s} {'% Var':>8s}")
print("-" * 32)
for i in range(12):
    mes_nombre = (pd.Timestamp('2025-01-01') + pd.DateOffset(months=i)).strftime('%b')
    var = (pronostico_anual.iloc[i] / ventas_actual[-12:].mean() - 1) * 100
    print(f"{mes_nombre:>10s} ${pronostico_anual.iloc[i]:>8,.0f} {var:>7.1f}%")

total_pronosticado = pronostico_anual.sum()
print(f"\nTotal año actual: ${ventas_totales_actual:,.0f}")
print(f"Total año próximo: ${total_pronosticado:,.0f}")
print(f"Crecimiento: {(total_pronosticado/ventas_totales_actual - 1)*100:.1f}%")
```

---

## 24.10 Ejemplo en Compras: Plan de Abastecimiento

```python
# Plan de compras basado en pronóstico de demanda
np.random.seed(42)

demanda_mensual = pd.Series(
    800 + 5 * np.arange(24) + 100 * np.sin(2 * np.pi * np.arange(24) / 12) + np.random.normal(0, 30, 24),
    index=pd.date_range('2023-01-01', periods=24, freq='ME'),
    name='demanda'
)

# Pronóstico de demanda
modelo_dem = ExponentialSmoothing(demanda_mensual, trend='add', seasonal='add', seasonal_periods=12).fit()
pred_dem = modelo_dem.forecast(6)

# Plan de abastecimiento
lead_time = 2  # meses
stock_seguridad_meses = 1

print("=== PLAN DE ABASTECIMIENTO (PRÓXIMOS 6 MESES) ===")
print(f"{'Mes':>10s} {'Demanda':>10s} {'Compras':>10s}")
print("-" * 32)
for i in range(6):
    mes = (pd.Timestamp('2025-01-01') + pd.DateOffset(months=i)).strftime('%b')
    demanda = pred_dem.iloc[i]
    compras = demanda * (1 + lead_time * 0.1 + stock_seguridad_meses * 0.05)
    print(f"{mes:>10s} {demanda:>8,.0f} {compras:>8,.0f}")

print(f"\nDemanda total 6 meses: {pred_dem.sum():.0f}")
```

---

## 24.11 Ejemplo en Inventarios: Política de Reposición

```python
# Política de inventario basada en pronóstico
np.random.seed(42)

# Demanda semanal
demanda_semanal = pd.Series(
    200 + 2 * np.arange(52) + 30 * np.sin(2 * np.pi * np.arange(52) / 52) + np.random.normal(0, 20, 52),
    index=pd.date_range('2024-01-01', periods=52, freq='W'),
    name='demanda_semanal'
)

# Modelo de pronóstico
modelo_inv = ExponentialSmoothing(demanda_semanal, trend='add', seasonal='add', seasonal_periods=52).fit()
pred_inv = modelo_inv.forecast(8)

# Política de inventario
lead_time = 2  # semanas
z = 1.645  # 95% nivel de servicio
costo_pedido = 500
costo_mantener = 2  # $/unidad/semana

for semana in range(8):
    dem_esperada = pred_inv.iloc[semana]
    ss = z * demanda_semanal.std() * np.sqrt(lead_time)
    punto_reorden = dem_esperada * lead_time + ss
    costo_total = (demanda_semanal.mean() * costo_pedido / dem_esperada + 
                   punto_reorden/2 * costo_mantener)
    
    print(f"Semana {semana+1}: Dem={dem_esperada:.0f}, "
          f"SS={ss:.0f}, ROP={punto_reorden:.0f}")
```

---

## 24.12 Validación de Pronósticos (Backtesting)

```python
def backtesting(serie, modelo_func, test_size=12, step=1):
    """Validación walk-forward de pronóstico"""
    n = len(serie)
    predictions = []
    actuals = []
    
    for start in range(0, test_size, step):
        train_end = n - test_size + start
        if train_end < 10:
            continue
        train = serie.iloc[:train_end]
        test = serie.iloc[train_end:train_end + step]
        
        try:
            modelo = modelo_func(train)
            pred = modelo.forecast(len(test))
            predictions.extend(pred.values)
            actuals.extend(test.values)
        except:
            continue
    
    if predictions:
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mape = mean_absolute_percentage_error(actuals, predictions) * 100
        print("=== BACKTESTING WALK-FORWARD ===")
        print(f"Períodos evaluados: {len(predictions)}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2f}%")
    else:
        print("No se pudieron generar predicciones")

# Backtesting de Holt-Winters
backtesting(df['ventas'], 
            lambda s: ExponentialSmoothing(s, trend='add', seasonal='add', seasonal_periods=12).fit(),
            test_size=24, step=3)
```

---

## 24.13 Resumen

| Método | Ventajas | Limitaciones |
|--------|----------|-------------|
| **Promedio Móvil** | Simple, filtra ruido | Ignora tendencia y estacionalidad |
| **Regresión** | Fácil interpretación | Solo captura tendencia lineal |
| **Holt-Winters** | Captura todos los patrones | Requiere estacionalidad constante |
| **ARIMA** | Flexible, base teórica sólida | Complejo de configurar |
| **Naive** | Referencia simple (último valor) | Muy impreciso en datos con patrón |

---

## Ejercicios Propuestos

1. **Ventas**: Genera un presupuesto anual de ventas usando Holt-Winters con datos de 3 años
2. **Compras**: Diseña un plan de abastecimiento basado en pronóstico de demanda para 6 meses
3. **Inventarios**: Implementa una política de reposición usando pronóstico + stock de seguridad

---

[← Anterior](23-suavizacion-exponencial.md) | [Índice](index.md) | [Siguiente →](25-estadistica-bayesiana.md)
