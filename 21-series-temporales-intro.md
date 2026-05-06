# Capítulo 21: Introducción a Series Temporales

[← Anterior](20-regresion-logistica.md) | [Índice](index.md) | [Siguiente →](22-modelos-arima.md)

---

## 21.1 ¿Qué es una Serie Temporal?

Una **serie temporal** es una secuencia de observaciones ordenadas cronológicamente. A diferencia de los datos transversales, las observaciones en una serie temporal **no son independientes**: el valor de hoy está relacionado con el de ayer.

### Ejemplos en Negocios

| Área | Serie Temporal |
|------|---------------|
| **Ventas** | Ventas diarias, semanales, mensuales |
| **Compras** | Precios de insumos, volúmenes de pedido |
| **Inventarios** | Nivel de stock, demanda diaria |
| **Finanzas** | Precios de acciones, tipo de cambio |

---

## 21.2 Componentes de una Serie Temporal

$$Y_t = T_t + S_t + C_t + \varepsilon_t \quad (\text{Aditivo})$$
$$Y_t = T_t \times S_t \times C_t \times \varepsilon_t \quad (\text{Multiplicativo})$$

| Componente | Descripción | Ejemplo |
|-----------|-------------|---------|
| **Tendencia (T)** | Movimiento a largo plazo | Crecimiento anual de ventas |
| **Estacionalidad (S)** | Patrón periódico fijo | Más ventas en diciembre |
| **Ciclo (C)** | Fluctuaciones no periódicas | Ciclo económico 5-10 años |
| **Residuo (ε)** | Variación aleatoria | Ruido impredecible |

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns

np.random.seed(42)

# Crear serie temporal sintética con todos los componentes
n_dias = 365
fechas = pd.date_range('2024-01-01', periods=n_dias, freq='D')

# Tendencia lineal: crecimiento de 0.1 por día
tendencia = np.linspace(100, 140, n_dias)

# Estacionalidad semanal: más ventas en fin de semana
estacionalidad_semanal = 10 * np.sin(2 * np.pi * np.arange(n_dias) / 7)

# Estacionalidad anual: pico en diciembre
estacionalidad_anual = 15 * np.sin(2 * np.pi * np.arange(n_dias) / 365 - np.pi/2)

# Ciclo trimestral
ciclo = 5 * np.sin(2 * np.pi * np.arange(n_dias) / 90)

# Ruido
ruido = np.random.normal(0, 8, n_dias)

# Serie completa
ventas = tendencia + estacionalidad_semanal + estacionalidad_anual + ciclo + ruido

serie = pd.Series(ventas, index=fechas, name='ventas')

print("=== SERIE TEMPORAL: VENTAS DIARIAS ===")
print(serie.head())
print(f"\nRango: {serie.min():.1f} - {serie.max():.1f}")
print(f"Media: {serie.mean():.1f}, Std: {serie.std():.1f}")
```

---

## 21.3 Visualización de Series Temporales

```python
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Serie completa
serie.plot(ax=axes[0], color='steelblue', linewidth=0.8)
axes[0].set_title('Ventas Diarias - Serie Completa', fontweight='bold')
axes[0].set_ylabel('Ventas ($)')
axes[0].grid(alpha=0.3)

# Primeros 60 días
serie[:60].plot(ax=axes[1], color='darkgreen', linewidth=1.5, marker='o', markersize=3)
axes[1].set_title('Primeros 60 Días (detalle)', fontweight='bold')
axes[1].set_ylabel('Ventas ($)')
axes[1].grid(alpha=0.3)

# Media móvil de 7 y 30 días
serie.plot(ax=axes[2], color='steelblue', alpha=0.5, linewidth=0.5, label='Original')
serie.rolling(7).mean().plot(ax=axes[2], color='red', linewidth=2, label='Media Móvil 7d')
serie.rolling(30).mean().plot(ax=axes[2], color='darkgreen', linewidth=2, label='Media Móvil 30d')
axes[2].set_title('Media Móvil de 7 y 30 Días', fontweight='bold')
axes[2].set_ylabel('Ventas ($)')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('serie_temporal_ventas.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 21.4 Descomposición de la Serie

```python
# Descomposición aditiva (por defecto)
descomposicion = seasonal_decompose(serie, model='additive', period=7)

fig, axes = plt.subplots(4, 1, figsize=(14, 10))

descomposicion.observed.plot(ax=axes[0], color='steelblue')
axes[0].set_title('Serie Original', fontweight='bold')
axes[0].set_ylabel('Ventas')

descomposicion.trend.plot(ax=axes[1], color='red')
axes[1].set_title('Tendencia', fontweight='bold')
axes[1].set_ylabel('Ventas')

descomposicion.seasonal.plot(ax=axes[2], color='darkgreen')
axes[2].set_title('Estacionalidad (semanal)', fontweight='bold')
axes[2].set_ylabel('Estacionalidad')

descomposicion.resid.plot(ax=axes[3], color='gray')
axes[3].set_title('Residuos', fontweight='bold')
axes[3].set_ylabel('Residuos')

plt.tight_layout()
plt.savefig('descomposicion_serie.png', dpi=150, bbox_inches='tight')
plt.show()

# Interpretación
print("=== DESCOMPOSICIÓN DE LA SERIE ===")
print(f"Tendencia: {'Creciente' if descomposicion.trend.iloc[-1] > descomposicion.trend.iloc[0] else 'Decreciente'}")
print(f"Estacionalidad: Detectada (período = 7 días)")
print(f"Residuos: Media = {descomposicion.resid.mean():.2f}, Std = {descomposicion.resid.std():.2f}")
```

---

## 21.5 Estacionariedad

Una serie es **estacionaria** si sus propiedades estadísticas (media, varianza, autocorrelación) no cambian con el tiempo.

### Prueba de Dickey-Fuller Aumentada (ADF)

- **H₀**: La serie NO es estacionaria (tiene raíz unitaria)
- **H₁**: La serie es estacionaria

```python
def prueba_estacionariedad(serie, nombre="Serie"):
    """Prueba de Dickey-Fuller Aumentada"""
    resultado = adfuller(serie.dropna(), autolag='AIC')
    
    print(f"=== PRUEBA DE ESTACIONARIEDAD (ADF) - {nombre} ===")
    print(f"Estadístico ADF: {resultado[0]:.6f}")
    print(f"p-valor: {resultado[1]:.6f}")
    print(f"Valores críticos:")
    for key, value in resultado[4].items():
        print(f"  {key}: {value:.4f}")
    
    if resultado[1] < 0.05:
        print(f"✓ La serie {nombre} ES estacionaria (p < 0.05)")
    else:
        print(f"✗ La serie {nombre} NO es estacionaria (p ≥ 0.05)")
        print("  → Se requiere diferenciación")
    
    return resultado[1]

# Prueba con la serie original
p_original = prueba_estacionariedad(serie, "Ventas originales")
```

### Diferenciación

```python
# Primera diferencia
serie_diff1 = serie.diff().dropna()

# Segunda diferencia (rara vez necesaria)
serie_diff2 = serie_diff1.diff().dropna()

print("=== DIFERENCIACIÓN ===")
p_diff1 = prueba_estacionariedad(serie_diff1, "Ventas diferenciadas (orden 1)")

# Visualización
fig, axes = plt.subplots(3, 1, figsize=(14, 8))
serie.plot(ax=axes[0], title='Original (No estacionaria)', color='steelblue')
serie_diff1.plot(ax=axes[1], title='Primera Diferencia (Estacionaria)', color='darkgreen')
serie_diff2.plot(ax=axes[2], title='Segunda Diferencia', color='coral')
plt.tight_layout()
plt.savefig('diferenciacion.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 21.6 Autocorrelación (ACF) y Autocorrelación Parcial (PACF)

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# ACF de la serie original
plot_acf(serie.dropna(), lags=40, ax=axes[0, 0])
axes[0, 0].set_title('ACF - Serie Original (no estacionaria)', fontweight='bold')

# PACF de la serie original
plot_pacf(serie.dropna(), lags=40, ax=axes[0, 1])
axes[0, 1].set_title('PACF - Serie Original (no estacionaria)', fontweight='bold')

# ACF de la serie diferenciada
plot_acf(serie_diff1, lags=40, ax=axes[1, 0])
axes[1, 0].set_title('ACF - Primera Diferencia (estacionaria)', fontweight='bold')

# PACF de la serie diferenciada
plot_pacf(serie_diff1, lags=40, ax=axes[1, 1])
axes[1, 1].set_title('PACF - Primera Diferencia (estacionaria)', fontweight='bold')

plt.tight_layout()
plt.savefig('acf_pacf.png', dpi=150, bbox_inches='tight')
plt.show()

# Interpretación de ACF/PACF
print("=== INTERPRETACIÓN DE ACF/PACF ===")
print("\nACF decae lentamente → Serie no estacionaria (confirmado)")
print("ACF de diferencia tiene 1 pico significativo → MA(1)")
print("PACF de diferencia tiene 1 pico significativo → AR(1)")
print("\nPosible modelo: ARIMA(1,1,1)")
```

---

## 21.7 Lag Features (Variables Rezagadas)

Crear predictores basados en valores pasados:

```python
def crear_lags(serie, n_lags=7):
    """Crea variables rezagadas"""
    df = pd.DataFrame({'y': serie})
    for lag in range(1, n_lags + 1):
        df[f'y_lag_{lag}'] = serie.shift(lag)
    return df.dropna()

lags_df = crear_lags(serie, 7)
print("=== VARIABLES REZAGADAS ===")
print(lags_df.head(10))
```

---

## 21.8 Ejemplo en Ventas: Análisis de Estacionalidad

```python
# Patrón de ventas por día de la semana
ventas_por_dia = pd.DataFrame({
    'ventas': serie.values,
    'dia_semana': serie.index.dayofweek,
    'mes': serie.index.month
})

patron_semanal = ventas_por_dia.groupby('dia_semana')['ventas'].agg(['mean', 'std', 'count'])
patron_semanal.index = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']

print("=== PATRÓN DE VENTAS POR DÍA ===")
print(patron_semanal)

# Efecto día de la semana
fig, ax = plt.subplots(figsize=(10, 5))
patron_semanal['mean'].plot(kind='bar', ax=ax, color='steelblue', edgecolor='white', yerr=patron_semanal['std'])
ax.set_title('Ventas Promedio por Día de la Semana', fontweight='bold')
ax.set_xlabel('Día')
ax.set_ylabel('Ventas Promedio')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('patron_semanal.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 21.9 Ejemplo en Compras: Precios de Insumos

```python
# Serie de precios de un insumo
np.random.seed(42)
n_semanas = 104
fechas_sem = pd.date_range('2024-01-01', periods=n_semanas, freq='W')

# Precio con tendencia + estacionalidad + ruido
tendencia_precio = np.linspace(100, 130, n_semanas)
estac_precio = 5 * np.sin(2 * np.pi * np.arange(n_semanas) / 26)  # ciclo 26 semanas
ruido_precio = np.random.normal(0, 3, n_semanas)
precio_insumo = tendencia_precio + estac_precio + ruido_precio

serie_precio = pd.Series(precio_insumo, index=fechas_sem, name='precio')

print("=== SERIE DE PRECIOS DE INSUMO ===")
print(f"Precio actual: ${serie_precio.iloc[-1]:.2f}")
print(f"Precio hace un año: ${serie_precio.iloc[-52]:.2f}")
print(f"Variación anual: {(serie_precio.iloc[-1]/serie_precio.iloc[-52]-1)*100:.1f}%")

# Descomposición
descomp_precio = seasonal_decompose(serie_precio, model='additive', period=26)
print(f"\nTendencia: ${descomp_precio.trend.iloc[-1]:.2f}")
print(f"Componente estacional actual: ${descomp_precio.seasonal.iloc[-1]:.2f}")
```

---

## 21.10 Ejemplo en Inventarios: Demanda Diaria

```python
# Serie de demanda con patrón semanal y tendencia
np.random.seed(42)
n_dias = 180
fechas_inv = pd.date_range('2024-01-01', periods=n_dias, freq='D')

# Demanda base + tendencia + estacionalidad semanal
demanda_base = 30
tendencia_dem = np.linspace(0, 10, n_dias)
estac_dem = 8 * np.sin(2 * np.pi * np.arange(n_dias) / 7)
ruido_dem = np.random.poisson(3, n_dias) - 3  # ruido Poisson

demanda = demanda_base + tendencia_dem + estac_dem + ruido_dem
demanda = np.clip(demanda, 5, 80).astype(int)

serie_demanda = pd.Series(demanda, index=fechas_inv, name='demanda')

print("=== DEMANDA DIARIA - ESTADÍSTICAS ===")
print(f"Demanda media: {serie_demanda.mean():.1f}")
print(f"Demanda máxima: {serie_demanda.max()}")
print(f"Demanda mínima: {serie_demanda.min()}")
print(f"Desviación: {serie_demanda.std():.1f}")

# Días de mayor demanda
print(f"\nTop 5 días de mayor demanda:")
top_dias = serie_demanda.nlargest(5)
for fecha, valor in top_dias.items():
    print(f"  {fecha.date()}: {valor} unidades ({fecha.day_name()})")

# Stock de seguridad basado en demanda
z = 1.645
ss = z * serie_demanda.std()
print(f"\nStock de seguridad recomendado (95%): {ss:.0f} unidades")
```

---

## 21.11 Resumen

| Concepto | Descripción |
|----------|-------------|
| **Componentes** | Tendencia, Estacionalidad, Ciclo, Residuo |
| **Estacionariedad** | Media y varianza constantes en el tiempo |
| **Diferenciación** | Transformación para hacer estacionaria |
| **ACF** | Autocorrelación (correlación con rezagos) |
| **PACF** | Autocorrelación parcial (efecto directo) |
| **Media Móvil** | Suavizado para ver tendencia |
| **Descomposición** | Separa componentes de la serie |

---

## Ejercicios Propuestos

1. **Ventas**: Descompón una serie de ventas con estacionalidad semanal y mensual. Identifica los componentes
2. **Compras**: Analiza la estacionariedad de una serie de precios. ¿Requiere diferenciación?
3. **Inventarios**: Calcula la ACF y PACF de la demanda diaria. ¿Qué rezagos son significativos?

---

[← Anterior](20-regresion-logistica.md) | [Índice](index.md) | [Siguiente →](22-modelos-arima.md)
