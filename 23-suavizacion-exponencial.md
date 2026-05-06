# Capítulo 23: Suavización Exponencial

[← Anterior](22-modelos-arima.md) | [Índice](index.md) | [Siguiente →](24-pronosticos.md)

---

## 23.1 Introducción

Los métodos de **suavización exponencial** asignan pesos decrecientes a las observaciones pasadas, dando más importancia a las más recientes. Son métodos:

- **Simples** de implementar y entender
- **Robustos** para pronósticos a corto plazo
- **Adaptativos** a cambios en el patrón

### Tipos de Suavización

| Método | Componentes | Patrón |
|--------|-------------|--------|
| **Simple (SES)** | Nivel | Sin tendencia ni estacionalidad |
| **Holt** | Nivel + Tendencia | Tendencia lineal |
| **Holt-Winters** | Nivel + Tendencia + Estacionalidad | Tendencia + Estacionalidad |

---

## 23.2 Suavización Exponencial Simple (SES)

Para series sin tendencia ni estacionalidad:

$$\hat{y}_{t+1} = \alpha y_t + (1-\alpha) \hat{y}_t$$

Donde:
- $\alpha$: constante de suavización (0 < α < 1)
- Mayor α = más peso a observaciones recientes
- Menor α = más suavizado

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.holtwinters import (SimpleExpSmoothing, 
                                          ExponentialSmoothing,
                                          Holt)
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Serie sin tendencia (demanda estable)
n = 100
demanda_estable = 50 + np.random.normal(0, 5, n)
serie_ses = pd.Series(demanda_estable, name='demanda')

# Ajustar SES con diferentes α
alphas = [0.1, 0.3, 0.7, 0.9]

fig, ax = plt.subplots(figsize=(14, 6))
serie_ses.plot(ax=ax, color='steelblue', linewidth=1, alpha=0.5, label='Original')

for alpha in alphas:
    modelo = SimpleExpSmoothing(serie_ses).fit(smoothing_level=alpha, optimized=False)
    ajustados = modelo.fittedvalues
    ax.plot(ajustados, label=f'α = {alpha}', linewidth=1.5)

ax.set_title('Suavización Exponencial Simple - Diferentes α', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.savefig('ses_comparacion.png', dpi=150, bbox_inches='tight')
plt.show()

# SES con α óptimo
modelo_ses = SimpleExpSmoothing(serie_ses).fit()
print("=== SUAVIZACIÓN EXPONENCIAL SIMPLE ===")
print(f"α óptimo: {modelo_ses.params['smoothing_level']:.4f}")
print(f"AIC: {modelo_ses.aic:.2f}")
print(f"BIC: {modelo_ses.bic:.2f}")

# Pronóstico
pred_ses = modelo_ses.forecast(10)
print(f"\nPronóstico próximos 10 períodos:")
for i, v in enumerate(pred_ses, 1):
    print(f"  t+{i:2d}: {v:.2f}")
```

---

## 23.3 Método de Holt (Tendencia Lineal)

Extiende SES para capturar **tendencia**:

- **Nivel**: $L_t = \alpha y_t + (1-\alpha)(L_{t-1} + T_{t-1})$
- **Tendencia**: $T_t = \beta(L_t - L_{t-1}) + (1-\beta)T_{t-1}$
- **Pronóstico**: $\hat{y}_{t+h} = L_t + h \times T_t$

```python
# Serie con tendencia
np.random.seed(42)
t = np.arange(100)
tendencia_lineal = 50 + 0.5 * t + np.random.normal(0, 5, 100)
serie_tendencia = pd.Series(tendencia_lineal, name='ventas')

# Holt
modelo_holt = Holt(serie_tendencia).fit()
print("=== MÉTODO DE HOLT (TENDENCIA LINEAL) ===")
print(f"α (nivel): {modelo_holt.params['smoothing_level']:.4f}")
print(f"β (tendencia): {modelo_holt.params['smoothing_trend']:.4f}")

# Pronóstico
pred_holt = modelo_holt.forecast(20)

fig, ax = plt.subplots(figsize=(14, 6))
serie_tendencia.plot(ax=ax, color='steelblue', linewidth=1.5, label='Original')
modelo_holt.fittedvalues.plot(ax=ax, color='green', linewidth=1.5, label='Ajustado')
pred_holt.plot(ax=ax, color='red', linewidth=2, label='Pronóstico', marker='o')
ax.set_title('Método de Holt - Tendencia Lineal', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.savefig('holt_tendencia.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nPronóstico tendencia (próximos 5):")
for i in range(5):
    print(f"  t+{i+1}: {pred_holt.iloc[i]:.2f}")
```

---

## 23.4 Holt-Winters (Estacionalidad)

Maneja **nivel + tendencia + estacionalidad**:

- **Nivel**: $L_t = \alpha(y_t - S_{t-m}) + (1-\alpha)(L_{t-1} + T_{t-1})$
- **Tendencia**: $T_t = \beta(L_t - L_{t-1}) + (1-\beta)T_{t-1}$
- **Estacionalidad**: $S_t = \gamma(y_t - L_t) + (1-\gamma)S_{t-m}$
- **Pronóstico**: $\hat{y}_{t+h} = L_t + hT_t + S_{t+h-m(k+1)}$

### Modos

| Modo | Estacionalidad | Componentes |
|------|---------------|-------------|
| **Aditivo** | $Y = T + S + R$ | Amplitud constante |
| **Multiplicativo** | $Y = T \times S \times R$ | Amplitud creciente |

```python
# Serie con tendencia y estacionalidad semanal
np.random.seed(42)
n = 200
t = np.arange(n)
# Tendencia + estacionalidad semanal (período 7)
serie_estacional = 100 + 0.3 * t + 20 * np.sin(2 * np.pi * t / 7) + np.random.normal(0, 5, n)
serie_hw = pd.Series(serie_estacional, name='ventas')

# Holt-Winters Aditivo
modelo_hw_add = ExponentialSmoothing(
    serie_hw, 
    trend='add', 
    seasonal='add', 
    seasonal_periods=7
).fit()

# Holt-Winters Multiplicativo
modelo_hw_mul = ExponentialSmoothing(
    serie_hw,
    trend='add',
    seasonal='mul',
    seasonal_periods=7
).fit()

print("=== HOLT-WINTERS ===")
print(f"Aditivo - AIC: {modelo_hw_add.aic:.2f}, BIC: {modelo_hw_add.bic:.2f}")
print(f"Multiplicativo - AIC: {modelo_hw_mul.aic:.2f}, BIC: {modelo_hw_mul.bic:.2f}")

print(f"\nParámetros (Aditivo):")
print(f"  α (nivel): {modelo_hw_add.params['smoothing_level']:.4f}")
print(f"  β (tendencia): {modelo_hw_add.params['smoothing_trend']:.4f}")
print(f"  γ (estacionalidad): {modelo_hw_add.params['smoothing_seasonal']:.4f}")

# Pronóstico
pred_hw = modelo_hw_add.forecast(21)  # 3 semanas

fig, ax = plt.subplots(figsize=(14, 6))
serie_hw.iloc[-60:].plot(ax=ax, color='steelblue', linewidth=1.5, label='Original (últimos 60)')
modelo_hw_add.fittedvalues.iloc[-60:].plot(ax=ax, color='green', linewidth=1.5, label='Ajustado')
pred_hw.plot(ax=ax, color='red', linewidth=2, label='Pronóstico 21 días', marker='o')
ax.set_title('Holt-Winters - Pronóstico con Estacionalidad', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.savefig('holt_winters_pronostico.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 23.5 Comparación de Métodos

```python
from statsmodels.tsa.holtwinters import Holt

def comparar_metodos(serie, test_size=20):
    """Compara SES, Holt y Holt-Winters"""
    train = serie.iloc[:-test_size]
    test = serie.iloc[-test_size:]
    
    modelos = {
        'SES': SimpleExpSmoothing(train).fit(),
        'Holt': Holt(train).fit(),
        'HW_Add': ExponentialSmoothing(train, trend='add', seasonal='add', 
                                        seasonal_periods=7).fit() if len(train) > 14 else None,
    }
    
    resultados = []
    for nombre, modelo in modelos.items():
        if modelo is None:
            continue
        pred = modelo.forecast(test_size)
        rmse = np.sqrt(mean_squared_error(test, pred))
        mae = mean_absolute_error(test, pred)
        resultados.append({'Método': nombre, 'RMSE': rmse, 'MAE': mae, 'AIC': modelo.aic})
    
    return pd.DataFrame(resultados).sort_values('RMSE')

# Serie con estacionalidad
resultados = comparar_metodos(serie_hw, test_size=21)
print("=== COMPARACIÓN DE MÉTODOS DE SUAVIZACIÓN ===")
print(resultados.to_string(index=False))
```

---

## 23.6 ETS (Error, Trend, Seasonal)

StatsModels implementa ETS (Error-Trend-Seasonal) que automatiza la selección:

```python
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

# Modelo ETS automático
modelo_ets = ETSModel(
    serie_hw,
    error='add',
    trend='add',
    seasonal='add',
    seasonal_periods=7
)
resultado_ets = modelo_ets.fit()

print("=== MODELO ETS ===")
print(resultado_ets.summary())

# Pronóstico
pred_ets = resultado_ets.forecast(14)
print(f"\nPronóstico ETS (14 días):")
for i in range(14):
    print(f"  Día {i+1:2d}: {pred_ets.iloc[i]:.1f}")
```

---

## 23.7 Ejemplo en Ventas: Pronóstico con Holt-Winters

```python
# Ventas diarias con patrón semanal
np.random.seed(42)
n = 182  # 26 semanas
ventas_diarias = 200 + 0.2 * np.arange(n) + \
                 30 * np.sin(2 * np.pi * np.arange(n) / 7) + \
                 40 * np.sin(2 * np.pi * np.arange(n) / 365) + \
                 np.random.normal(0, 15, n)

serie_ventas = pd.Series(ventas_diarias, name='ventas')

modelo_vtas = ExponentialSmoothing(
    serie_ventas, 
    trend='add', 
    seasonal='add', 
    seasonal_periods=7
).fit()

pred_vtas = modelo_vtas.forecast(14)

print("=== PRONÓSTICO DE VENTAS (14 DÍAS) ===")
print(f"{'Día':>5s} {'Pronóstico':>12s}")
print("-" * 18)
total = 0
for i, v in enumerate(pred_vtas, 1):
    print(f"{i:>5d} ${v:>8,.0f}")
    total += v

print(f"\nVentas totales estimadas (14d): ${total:,.0f}")
print(f"Ventas promedio diarias: ${total/14:,.0f}")
```

---

## 23.8 Ejemplo en Compras: Precios con Tendencia

```python
# Precios de insumo con tendencia alcista
np.random.seed(42)
n = 52  # semanas
precios_sem = 80 + 0.5 * np.arange(n) + 3 * np.sin(2 * np.pi * np.arange(n) / 13) + \
              np.random.normal(0, 2, n)

serie_precios = pd.Series(precios_sem, name='precio')

modelo_precios = Holt(serie_precios).fit()
pred_precios = modelo_precios.forecast(4)

print("=== PRONÓSTICO DE PRECIOS (PRÓXIMAS 4 SEMANAS) ===")
for i, p in enumerate(pred_precios, 1):
    print(f"  Semana {i}: ${p:.2f}")

# Momento óptimo de compra
print(f"\nPrecio actual: ${serie_precios.iloc[-1]:.2f}")
print(f"Precio estimado semana 4: ${pred_precios.iloc[3]:.2f}")
print(f"Tendencia: {'↑ Alcista' if pred_precios.iloc[3] > serie_precios.iloc[-1] else '↓ Bajista'}")
```

---

## 23.9 Ejemplo en Inventarios: Demanda con Estacionalidad

```python
# Demanda semanal con estacionalidad mensual
np.random.seed(42)
n = 104  # 2 años
demanda_sem = 500 + 2 * np.arange(n) + \
              50 * np.sin(2 * np.pi * np.arange(n) / 4.33) + \
              np.random.normal(0, 30, n)

serie_dem_sem = pd.Series(demanda_sem, name='demanda_semanal')

modelo_dem = ExponentialSmoothing(
    serie_dem_sem,
    trend='add',
    seasonal='add',
    seasonal_periods=4  # estacionalidad mensual (~4 semanas)
).fit()

pred_dem = modelo_dem.forecast(8)  # 2 meses

print("=== PRONÓSTICO DE DEMANDA (8 SEMANAS) ===")
demanda_total = pred_dem.sum()
print(f"Demanda total estimada: {demanda_total:.0f} unidades")
print(f"Demanda semanal promedio: {demanda_total/8:.0f} unidades")

# Stock de seguridad
ss = 1.645 * pred_dem.std() * np.sqrt(2)  # lead time = 2 semanas
print(f"Stock de seguridad recomendado: {ss:.0f} unidades")
print(f"Inventario necesario: {demanda_total + ss:.0f} unidades")
```

---

## 23.10 Parámetros y Sensibilidad

```python
def analizar_sensibilidad_alpha(serie, test_size=20):
    """Analiza cómo afecta α al error de pronóstico"""
    train = serie.iloc[:-test_size]
    test = serie.iloc[-test_size:]
    
    alphas = np.linspace(0.05, 0.95, 19)
    errores = []
    
    for alpha in alphas:
        modelo = SimpleExpSmoothing(train).fit(smoothing_level=alpha, optimized=False)
        pred = modelo.forecast(test_size)
        rmse = np.sqrt(mean_squared_error(test, pred))
        errores.append(rmse)
    
    best_alpha = alphas[np.argmin(errores)]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(alphas, errores, 'bo-', linewidth=2)
    ax.axvline(best_alpha, color='red', linestyle='--', label=f'Óptimo α={best_alpha:.2f}')
    ax.set_xlabel('α (constante de suavización)', fontweight='bold')
    ax.set_ylabel('RMSE', fontweight='bold')
    ax.set_title('Sensibilidad de α en Suavización Exponencial Simple', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.savefig('sensibilidad_alpha.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"=== SENSIBILIDAD DE α ===")
    print(f"α óptimo: {best_alpha:.2f} (RMSE mínimo: {min(errores):.2f})")
    print(f"α=0.1: RMSE={errores[1]:.2f}")
    print(f"α=0.5: RMSE={errores[9]:.2f}")
    print(f"α=0.9: RMSE={errores[17]:.2f}")

analizar_sensibilidad_alpha(serie_ses, test_size=15)
```

---

## 23.11 Resumen

| Método | Componentes | Cuándo Usar |
|--------|-------------|-------------|
| **SES** | Nivel | Serie sin tendencia ni estacionalidad |
| **Holt** | Nivel + Tendencia | Serie con tendencia lineal |
| **Holt-Winters** | Nivel + Tendencia + Estacionalidad | Serie con tendencia y estacionalidad |
| **ETS** | Automático | Cuando no sabes qué componentes elegir |

### Ventajas sobre ARIMA
- Más simples e intuitivos
- Se adaptan a cambios de patrón
- Requieren menos datos históricos
- Fáciles de interpretar para negocios

---

## Ejercicios Propuestos

1. **Ventas**: Aplica Holt-Winters a ventas diarias con estacionalidad semanal. Pronostica 14 días
2. **Compras**: Usa Holt para pronosticar precios de insumos con tendencia lineal
3. **Inventarios**: Compara SES, Holt y Holt-Winters para demanda de producto. ¿Cuál da menor RMSE?

---

[← Anterior](22-modelos-arima.md) | [Índice](index.md) | [Siguiente →](24-pronosticos.md)
