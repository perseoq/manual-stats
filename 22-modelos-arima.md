# Capítulo 22: Modelos ARIMA

[← Anterior](21-series-temporales-intro.md) | [Índice](index.md) | [Siguiente →](23-suavizacion-exponencial.md)

---

## 22.1 Introducción a ARIMA

**ARIMA** (AutoRegressive Integrated Moving Average) es uno de los modelos más poderosos para pronóstico de series temporales.

### Componentes

| Sigla | Significado | Parámetro |
|-------|-------------|-----------|
| **AR** | AutoRegresivo: depende de valores pasados | p |
| **I** | Integrado: diferenciación para estacionariedad | d |
| **MA** | Media Móvil: depende de errores pasados | q |

### Notación

$$ARIMA(p, d, q)$$

Donde:
- **p**: Orden del componente autorregresivo
- **d**: Orden de diferenciación
- **q**: Orden de media móvil

### Ecuación General

$$Y_t' = c + \phi_1 Y_{t-1}' + ... + \phi_p Y_{t-p}' + \theta_1 \varepsilon_{t-1} + ... + \theta_q \varepsilon_{t-q} + \varepsilon_t$$

Donde $Y_t'$ es la serie diferenciada d veces.

---

## 22.2 Identificación del Modelo (ACF/PACF)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Crear serie ARIMA(1,1,1) simulada
n = 300
# Parámetros
phi = 0.7  # AR(1)
theta = 0.3  # MA(1)

# Generar proceso
errores = np.random.normal(0, 2, n)
y = np.zeros(n)

# AR(1): y_t = phi * y_{t-1} + error_t
for t in range(1, n):
    y[t] = phi * y[t-1] + errores[t] + theta * errores[t-1]

# Añadir tendencia para hacerlo I(1)
tendencia = np.linspace(0, 5, n)
serie_original = y + tendencia

serie = pd.Series(serie_original, name='serie')

print("=== SERIE SIMULADA ARIMA(1,1,1) ===")
print(f"Parámetros: φ={phi}, θ={theta}")
print(f"Media: {serie.mean():.2f}")
print(f"Std: {serie.std():.2f}")
```

### Identificación con ACF y PACF

```python
# Diferenciar
serie_diff = serie.diff().dropna()

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

plot_acf(serie, lags=30, ax=axes[0, 0])
axes[0, 0].set_title('ACF - Serie Original (no estacionaria)', fontweight='bold')

plot_pacf(serie, lags=30, ax=axes[0, 1])
axes[0, 1].set_title('PACF - Serie Original', fontweight='bold')

plot_acf(serie_diff, lags=30, ax=axes[1, 0])
axes[1, 0].set_title('ACF - Serie Diferenciada (estacionaria)', fontweight='bold')

plot_pacf(serie_diff, lags=30, ax=axes[1, 1])
axes[1, 1].set_title('PACF - Serie Diferenciada', fontweight='bold')

plt.tight_layout()
plt.savefig('identificacion_arima.png', dpi=150, bbox_inches='tight')
plt.show()

# Reglas de identificación
print("=== IDENTIFICACIÓN ARIMA ===")
print("\nACF de la diferenciada:")
acf_vals = acf(serie_diff, nlags=20)
print(f"  Primer rezago significativo: {np.where(np.abs(acf_vals[1:]) > 1.96/np.sqrt(len(serie_diff)))[0]}")
print("  Si ACF decae lentamente → AR significativo")
print("  Si ACF tiene corte brusco → MA significativo")

print("\nPACF de la diferenciada:")
pacf_vals = pacf(serie_diff, nlags=20)
print(f"  Primer rezago significativo: {np.where(np.abs(pacf_vals[1:]) > 1.96/np.sqrt(len(serie_diff)))[0]}")
print("  Si PACF tiene corte brusco → AR(p)")
print("  Si PACF decae lentamente → MA(q)")
```

---

## 22.3 Estimación del Modelo ARIMA

```python
# Ajustar modelo ARIMA(1,1,1)
modelo_arima = ARIMA(serie, order=(1, 1, 1))
resultado = modelo_arima.fit()

print("=== MODELO ARIMA(1,1,1) - RESULTADOS ===")
print(resultado.summary())

# Coeficientes estimados vs reales
print(f"\n=== COEFICIENTES: ESTIMADOS vs REALES ===")
print(f"φ estimado: {resultado.arparams[0]:.4f} (real: {phi})")
print(f"θ estimado: {resultado.maparams[0]:.4f} (real: {theta})")
print(f"AIC: {resultado.aic:.2f}")
print(f"BIC: {resultado.bic:.2f}")
```

---

## 22.4 Diagnóstico de Residuos

```python
residuos = resultado.resid

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Residuos
axes[0, 0].plot(residuos)
axes[0, 0].axhline(y=0, color='red', linestyle='--')
axes[0, 0].set_title('Residuos del Modelo ARIMA', fontweight='bold')

# Histograma
axes[0, 1].hist(residuos, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='white')
x = np.linspace(residuos.min(), residuos.max(), 100)
axes[0, 1].plot(x, stats.norm.pdf(x, residuos.mean(), residuos.std()), 'r-', linewidth=2)
axes[0, 1].set_title('Histograma de Residuos', fontweight='bold')

# Q-Q plot
stats.probplot(residuos, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot', fontweight='bold')

# ACF de residuos
plot_acf(residuos, lags=30, ax=axes[1, 1])
axes[1, 1].set_title('ACF de Residuos (deben ser ruido blanco)', fontweight='bold')

plt.tight_layout()
plt.savefig('diagnostico_arima.png', dpi=150, bbox_inches='tight')
plt.show()

# Prueba Ljung-Box para autocorrelación en residuos
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test = acorr_ljungbox(residuos, lags=[10, 20, 30])
print("=== PRUEBA LJUNG-BOX (H₀: residuos son ruido blanco) ===")
print(lb_test)
```

---

## 22.5 Pronóstico con ARIMA

```python
# Pronóstico a 30 períodos
n_forecast = 30
forecast = resultado.get_forecast(steps=n_forecast)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

print("=== PRONÓSTICO ARIMA(1,1,1) - PRÓXIMOS 30 PERÍODOS ===")
for i in range(n_forecast):
    print(f"  t+{i+1:2d}: {forecast_mean.iloc[i]:.2f} "
          f"[{forecast_ci.iloc[i, 0]:.2f}, {forecast_ci.iloc[i, 1]:.2f}]")

# Visualización
fig, ax = plt.subplots(figsize=(14, 6))

# Últimos 100 períodos + pronóstico
serie.iloc[-100:].plot(ax=ax, color='steelblue', linewidth=1.5, label='Histórico')
forecast_mean.plot(ax=ax, color='red', linewidth=2, label='Pronóstico')
ax.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1],
                color='red', alpha=0.1, label='IC 95%')
ax.axvline(x=serie.index[-1], color='gray', linestyle='--', alpha=0.5)
ax.set_title('Pronóstico ARIMA(1,1,1)', fontweight='bold')
ax.set_xlabel('Tiempo')
ax.set_ylabel('Valor')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('pronostico_arima.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 22.6 Selección Automática de ARIMA (p,d,q)

```python
from itertools import product

def seleccionar_arima(serie, max_p=5, max_d=2, max_q=5, criterio='aic'):
    """Búsqueda de mejor ARIMA(p,d,q) por AIC/BIC"""
    best_aic = np.inf
    best_order = None
    results = []
    
    for d in range(max_d + 1):
        # Verificar si diferenciación es necesaria
        if d > 0:
            serie_test = serie.diff(d).dropna()
        else:
            serie_test = serie
        
        p_adf = adfuller(serie_test.dropna())[1]
        
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    modelo = ARIMA(serie, order=(p, d, q))
                    resultado = modelo.fit()
                    aic = resultado.aic
                    bic = resultado.bic
                    
                    val = aic if criterio == 'aic' else bic
                    
                    if val < best_aic:
                        best_aic = val
                        best_order = (p, d, q)
                    
                    results.append({'order': (p, d, q), 'aic': aic, 'bic': bic})
                    
                except:
                    continue
    
    # Mostrar top 5
    df_results = pd.DataFrame(results).sort_values(criterio).head(10)
    
    print(f"=== SELECCIÓN AUTOMÁTICA ARIMA (por {criterio.upper()}) ===")
    print(df_results.to_string(index=False))
    print(f"\nMejor modelo: ARIMA{best_order} (AIC={best_aic:.2f})")
    
    return best_order

# Buscar mejor ARIMA
mejor_orden = seleccionar_arima(serie, max_p=5, max_d=2, max_q=5)
```

---

## 22.7 ARIMA Estacional (SARIMA)

Para series con estacionalidad:

$$SARIMA(p,d,q)(P,D,Q)_m$$

Donde $m$ es el período estacional.

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Crear serie con estacionalidad semanal
np.random.seed(42)
n = 200
t = np.arange(n)
serie_sazonal = 50 + 0.1 * t + 10 * np.sin(2 * np.pi * t / 7) + np.random.normal(0, 3, n)
serie_saz = pd.Series(serie_sazonal, name='ventas_semanales')

# SARIMA(1,1,1)(1,1,1,7)
modelo_sarima = SARIMAX(serie_saz, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
resultado_sarima = modelo_sarima.fit(disp=False)

print("=== SARIMA(1,1,1)(1,1,1,7) - RESUMEN ===")
print(resultado_sarima.summary().tables[0])
print(resultado_sarima.summary().tables[1])

# Pronóstico
forecast_sarima = resultado_sarima.get_forecast(steps=14)
fc_mean = forecast_sarima.predicted_mean
fc_ci = forecast_sarima.conf_int()

print(f"\nPronóstico SARIMA (14 días):")
for i in range(14):
    print(f"  Día {i+1:2d}: {fc_mean.iloc[i]:.1f} [{fc_ci.iloc[i,0]:.1f}, {fc_ci.iloc[i,1]:.1f}]")
```

---

## 22.8 Ejemplo en Ventas: Pronóstico Semanal

```python
# Pronóstico de ventas semanales con ARIMA
np.random.seed(42)
n_semanas = 104
ventas_sem = 500 + np.cumsum(np.random.normal(0, 10, n_semanas)) + \
             20 * np.sin(2 * np.pi * np.arange(n_semanas) / 52)
ventas_sem_serie = pd.Series(ventas_sem, name='ventas_semanales')

# Ajustar ARIMA
modelo_vtas = ARIMA(ventas_sem_serie, order=(1, 1, 1))
resultado_vtas = modelo_vtas.fit()

# Pronóstico 12 semanas
pred_vtas = resultado_vtas.forecast(steps=12)

print("=== PRONÓSTICO DE VENTAS SEMANALES (ARIMA) ===")
print(f"{'Semana':>8s} {'Pronóstico':>12s}")
print("-" * 22)
for i, v in enumerate(pred_vtas, 1):
    print(f"{i:>8d} ${v:>8,.0f}")

# Evaluación
print(f"\nAIC: {resultado_vtas.aic:.1f}")
print(f"BIC: {resultado_vtas.bic:.1f}")
```

---

## 22.9 Ejemplo en Compras: Precios Futuros

```python
# Pronóstico de precios de insumos
np.random.seed(42)
n = 50
precios = 100 + np.cumsum(np.random.normal(0.5, 1, n)) + 5 * np.sin(2 * np.pi * np.arange(n) / 12)
precios_serie = pd.Series(precios, name='precio')

modelo_precio = ARIMA(precios_serie, order=(1, 1, 0))
resultado_precio = modelo_precio.fit()

pred_precio = resultado_precio.forecast(steps=3)

print("=== PRONÓSTICO DE PRECIOS (PRÓXIMOS 3 MESES) ===")
for i, p in enumerate(pred_precio, 1):
    print(f"  Mes {i}: ${p:.2f}")

# Decisión de compra
print(f"\nPrecio actual: ${precios_serie.iloc[-1]:.2f}")
print(f"Precio pronosticado mes 3: ${pred_precio.iloc[2]:.2f}")
recomendacion = "Comprar ahora" if pred_precio.iloc[2] > precios_serie.iloc[-1] else "Esperar"
print(f"Recomendación: {recomendacion}")
```

---

## 22.10 Ejemplo en Inventarios: Demanda Futura

```python
# Pronóstico de demanda para planificación de inventarios
np.random.seed(42)
n_dias = 90
demanda_hist = 30 + 0.05 * np.arange(n_dias) + \
               5 * np.sin(2 * np.pi * np.arange(n_dias) / 7) + \
               np.random.poisson(2, n_dias)

demanda_serie = pd.Series(demanda_hist, name='demanda')

modelo_dem = ARIMA(demanda_serie, order=(2, 1, 1))
resultado_dem = modelo_dem.fit()

# Pronóstico 14 días
pred_dem = resultado_dem.forecast(steps=14)
ic_dem = resultado_dem.get_forecast(steps=14).conf_int()

print("=== PRONÓSTICO DE DEMANDA (14 DÍAS) ===")
demanda_total = pred_dem.sum()
ss_total = 1.645 * pred_dem.std() * np.sqrt(7)
print(f"Demanda pronosticada total (14d): {demanda_total:.0f} unidades")
print(f"Stock de seguridad recomendado: {ss_total:.0f} unidades")
print(f"Inventario necesario: {demanda_total + ss_total:.0f} unidades")

print(f"\nDesglose diario:")
for i in range(14):
    print(f"  Día {i+1:2d}: {pred_dem.iloc[i]:.0f} [{ic_dem.iloc[i,0]:.0f}, {ic_dem.iloc[i,1]:.0f}]")
```

---

## 22.11 Validación de Pronósticos

```python
from sklearn.metrics import mean_absolute_percentage_error

def validar_pronostico(serie, train_size=0.8, order=(1,1,1)):
    """Valida modelo ARIMA con división train/test"""
    n = len(serie)
    split = int(n * train_size)
    
    train = serie.iloc[:split]
    test = serie.iloc[split:]
    
    # Modelo
    modelo = ARIMA(train, order=order)
    resultado = modelo.fit()
    
    # Pronóstico
    pred = resultado.forecast(steps=len(test))
    
    # Métricas
    rmse = np.sqrt(mean_squared_error(test, pred))
    mae = mean_absolute_error(test, pred)
    mape = mean_absolute_percentage_error(test, pred) * 100
    
    print("=== VALIDACIÓN DEL PRONÓSTICO ===")
    print(f"Tamaño train: {len(train)}, test: {len(test)}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    return rmse, mae, mape

validar_pronostico(serie, train_size=0.8, order=(1, 1, 1))
```

---

## 22.12 Resumen

| Componente | Descripción | Identificación |
|-----------|-------------|---------------|
| **AR(p)** | Depende de valores pasados | PACF con corte brusco en lag p |
| **I(d)** | Diferenciación para estacionariedad | ADF test |
| **MA(q)** | Depende de errores pasados | ACF con corte brusco en lag q |
| **SARIMA** | Versión estacional de ARIMA | Estacionalidad visible |

---

## Ejercicios Propuestos

1. **Ventas**: Ajusta un modelo ARIMA a ventas semanales. Pronostica las próximas 4 semanas
2. **Compras**: Usa SARIMA para modelar precios de insumos con estacionalidad trimestral
3. **Inventarios**: Encuentra el mejor ARIMA(p,d,q) para demanda diaria mediante búsqueda automática

---

[← Anterior](21-series-temporales-intro.md) | [Índice](index.md) | [Siguiente →](23-suavizacion-exponencial.md)
