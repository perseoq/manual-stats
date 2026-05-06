# Capítulo 26: Simulación Montecarlo

[← Anterior](25-estadistica-bayesiana.md) | [Índice](index.md) | [Siguiente →](27-bootstrap.md)

---

## 26.1 Introducción

La **simulación Montecarlo** es una técnica computacional que utiliza muestreo aleatorio repetido para obtener resultados numéricos. Es especialmente útil cuando los problemas son demasiado complejos para resolverse analíticamente.

### Principio Básico

1. Definir el modelo con sus variables aleatorias
2. Generar miles de escenarios posibles
3. Analizar la distribución de resultados

### Aplicaciones en Negocios

| Aplicación | Variables Aleatorias | Resultado |
|-----------|---------------------|-----------|
| **Riesgo de inventario** | Demanda, lead time | Probabilidad de stockout |
| **Evaluación de proyectos** | Ingresos, costos | VAN, TIR, riesgo |
| **Fijación de precios** | Demanda, elasticidad | Ingreso esperado |
| **Gestión de compras** | Precios, tipos de cambio | Costo esperado |

---

## 26.2 Simulación de una Variable Aleatoria

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

np.random.seed(42)

# Demanda diaria: Normal(100, 20) con límite inferior en 0
def simular_demanda(n_dias=1000, media=100, std=20):
    """Simula demanda diaria"""
    demanda = np.random.normal(media, std, n_dias)
    return np.maximum(demanda, 0)  # No demanda negativa

demanda_sim = simular_demanda(10000)

print("=== SIMULACIÓN DE DEMANDA ===")
print(f"Media: {demanda_sim.mean():.1f} (teórica: 100)")
print(f"Std: {demanda_sim.std():.1f} (teórica: 20)")
print(f"P90: {np.percentile(demanda_sim, 90):.1f}")
print(f"P95: {np.percentile(demanda_sim, 95):.1f}")
print(f"P99: {np.percentile(demanda_sim, 99):.1f}")
```

---

## 26.3 Simulación de Inventario (Stockout)

```python
def simular_inventario(demanda_media=100, demanda_std=20, 
                       stock_inicial=500, lead_time=5, n_dias=365,
                       punto_reorden=300, cantidad_pedido=400):
    """Simula operación de inventario con Montecarlo"""
    stock = stock_inicial
    pedido_pendiente = 0
    dias_pedido = 0
    
    historial = []
    
    for dia in range(n_dias):
        # Demanda del día
        demanda = max(0, np.random.normal(demanda_media, demanda_std))
        
        # ¿Llega pedido?
        if pedido_pendiente > 0:
            dias_pedido -= 1
            if dias_pedido <= 0:
                stock += pedido_pendiente
                pedido_pendiente = 0
        
        # Venta
        venta = min(demanda, stock)
        stock -= venta
        stockout = demanda > stock + venta
        
        # ¿Reordenar?
        if stock <= punto_reorden and pedido_pendiente == 0:
            pedido_pendiente = cantidad_pedido
            dias_pedido = lead_time
        
        historial.append({
            'dia': dia, 'demanda': demanda, 'venta': venta,
            'stock': stock, 'stockout': stockout
        })
    
    df = pd.DataFrame(historial)
    return df

# Ejecutar simulación
resultado = simular_inventario()

print("=== SIMULACIÓN DE INVENTARIO (365 DÍAS) ===")
print(f"Días con stockout: {resultado['stockout'].sum()}")
print(f"Nivel de servicio: {(1 - resultado['stockout'].mean())*100:.1f}%")
print(f"Stock promedio: {resultado['stock'].mean():.0f}")
print(f"Stock mínimo: {resultado['stock'].min():.0f}")
print(f"Stock final: {resultado['stock'].iloc[-1]:.0f}")
print(f"Ventas totales: {resultado['venta'].sum():.0f}")
print(f"Demanda total: {resultado['demanda'].sum():.0f}")
print(f"Ventas perdidas: {resultado['demanda'].sum() - resultado['venta'].sum():.0f}")
```

---

## 26.4 Análisis de Múltiples Escenarios

```python
def simulacion_multiple(n_simulaciones=1000, **kwargs):
    """Ejecuta múltiples simulaciones y analiza resultados"""
    resultados = []
    
    for _ in range(n_simulaciones):
        df = simular_inventario(**kwargs)
        resultados.append({
            'nivel_servicio': 1 - df['stockout'].mean(),
            'stock_promedio': df['stock'].mean(),
            'stock_min': df['stock'].min(),
            'ventas_totales': df['venta'].sum()
        })
    
    return pd.DataFrame(resultados)

# Ejecutar 1000 simulaciones
multi_sim = simulacion_multiple(1000)

print("=== ANÁLISIS DE 1000 SIMULACIONES ===")
print(f"{'Métrica':20s} {'Media':>10s} {'P5':>10s} {'P95':>10s}")
print("-" * 50)
for col in multi_sim.columns:
    print(f"{col:20s} {multi_sim[col].mean():>10.3f} "
          f"{multi_sim[col].quantile(0.05):>10.3f} "
          f"{multi_sim[col].quantile(0.95):>10.3f}")

# Riesgo de nivel de servicio bajo
riesgo = (multi_sim['nivel_servicio'] < 0.95).mean()
print(f"\nRiesgo de servicio < 95%: {riesgo:.1%}")
```

---

## 26.5 Simulación de Demanda con Incertidumbre

```python
def simular_demanda_incierta(n_sim=10000, horizonte=30):
    """Simula demanda futura con incertidumbre creciente"""
    np.random.seed(42)
    
    demandas = np.zeros((n_sim, horizonte))
    
    for sim in range(n_sim):
        demanda_actual = 100
        for t in range(horizonte):
            # La incertidumbre crece con el horizonte
            incertidumbre = 10 + t * 1.5
            demanda_actual = max(0, demanda_actual + 
                                 np.random.normal(0.5, incertidumbre))
            demandas[sim, t] = demanda_actual
    
    return demandas

horizonte = 30
demandas_futuras = simular_demanda_incierta(5000, horizonte)

# Estadísticas por período
print("=== DEMANDA FUTURA CON INCERTIDUMBRE ===")
print(f"{'Día':>5s} {'Media':>8s} {'P10':>8s} {'P50':>8s} {'P90':>8s}")
print("-" * 37)
for t in range(0, horizonte, 5):
    medias = demandas_futuras[:, t]
    print(f"{t+1:>5d} {medias.mean():>8.1f} "
          f"{np.percentile(medias, 10):>8.1f} "
          f"{np.percentile(medias, 50):>8.1f} "
          f"{np.percentile(medias, 90):>8.1f}")

# Stock de seguridad dinámico
print(f"\nStock de seguridad (día 1): {np.percentile(demandas_futuras[:, 0], 95):.0f}")
print(f"Stock de seguridad (día 30): {np.percentile(demandas_futuras[:, -1], 95):.0f}")
```

---

## 26.6 Optimización de Parámetros con Montecarlo

```python
def optimizar_punto_reorden(n_sim=500):
    """Encuentra el punto de reorden óptimo"""
    puntos_reorden = np.arange(100, 500, 50)
    resultados = []
    
    for pr in puntos_reorden:
        nivel_servicio = []
        stock_prom = []
        
        for _ in range(n_sim):
            df = simular_inventario(punto_reorden=pr, cantidad_pedido=400)
            nivel_servicio.append(1 - df['stockout'].mean())
            stock_prom.append(df['stock'].mean())
        
        resultados.append({
            'punto_reorden': pr,
            'nivel_servicio': np.mean(nivel_servicio),
            'stock_promedio': np.mean(stock_prom)
        })
    
    return pd.DataFrame(resultados)

opt_results = optimizar_punto_reorden(200)

print("=== OPTIMIZACIÓN DE PUNTO DE REORDEN ===")
print(opt_results.to_string(index=False))

# Encontrar el punto de reorden mínimo para 95% servicio
optimo = opt_results[opt_results['nivel_servicio'] >= 0.95].iloc[0]
print(f"\nPunto de reorden óptimo (servicio ≥ 95%): {optimo['punto_reorden']:.0f}")
print(f"Nivel de servicio: {optimo['nivel_servicio']:.1%}")
print(f"Stock promedio: {optimo['stock_promedio']:.0f}")
```

---

## 26.7 Ejemplo en Ventas: Pronóstico con Incertidumbre

```python
"""Evaluación probabilística del presupuesto de ventas"""
np.random.seed(42)

n_sim = 10000

# Variables inciertas
crecimiento_mercado = np.random.normal(0.05, 0.02, n_sim)  # 5% ± 2%
participacion = np.random.uniform(0.15, 0.25, n_sim)  # 20% ± 5%
precio_promedio = np.random.normal(100, 10, n_sim)  # $100 ± $10
clientes_totales = np.random.poisson(50000, n_sim)  # 50,000 clientes

# Cálculo de ventas
ventas_estimadas = clientes_totales * participacion * precio_promedio * (1 + crecimiento_mercado)

print("=== PRESUPUESTO DE VENTAS (MONTE CARLO) ===")
print(f"Ventas esperadas: ${ventas_estimadas.mean():,.0f}")
print(f"Desviación: ${ventas_estimadas.std():,.0f}")
print(f"P10: ${np.percentile(ventas_estimadas, 10):,.0f}")
print(f"P50: ${np.percentile(ventas_estimadas, 50):,.0f}")
print(f"P90: ${np.percentile(ventas_estimadas, 90):,.0f}")
print(f"\nProbabilidad de superar $1.5M: {(ventas_estimadas > 1500000).mean():.1%}")
```

---

## 26.8 Ejemplo en Compras: Riesgo de Tipo de Cambio

```python
"""Evaluación del riesgo cambiario en compras internacionales"""
np.random.seed(42)

n_sim = 10000

# Tipo de cambio actual: 18.5 MXN/USD
tc_actual = 18.5
volatilidad_tc = 0.15  # 15% anual
tiempo = 3/12  # 3 meses

# Simulación de tipo de cambio futuro
tc_futuro = tc_actual * np.exp(
    np.random.normal(0, volatilidad_tc * np.sqrt(tiempo), n_sim)
)

# Compra planeada
monto_usd = 50000  # USD
costo_mxn_actual = monto_usd * tc_actual
costo_mxn_futuro = monto_usd * tc_futuro

print("=== RIESGO CAMBIARIO - COMPRAS INTERNACIONALES ===")
print(f"Costo actual: ${costo_mxn_actual:,.0f} MXN")
print(f"Costo esperado (3 meses): ${costo_mxn_futuro.mean():,.0f} MXN")
print(f"Peor caso (P95): ${np.percentile(costo_mxn_futuro, 95):,.0f} MXN")
print(f"Mejor caso (P5): ${np.percentile(costo_mxn_futuro, 5):,.0f} MXN")
print(f"\nRiesgo de sobrecosto > 10%: {(costo_mxn_futuro > costo_mxn_actual * 1.10).mean():.1%}")

# Recomendación de cobertura
if (costo_mxn_futuro > costo_mxn_actual * 1.10).mean() > 0.20:
    print("→ Recomendación: Contratar cobertura cambiaria (forward)")
else:
    print("→ Bajo riesgo de tipo de cambio significativo")
```

---

## 26.9 Ejemplo en Inventarios: Política Óptima

```python
"""Simulación completa para determinar política óptima de inventario"""
np.random.seed(42)

def evaluar_politica(punto_reorden, cantidad_pedido, n_sim=500):
    """Evalúa una política de inventario"""
    resultados = []
    
    for _ in range(n_sim):
        df = simular_inventario(
            punto_reorden=punto_reorden,
            cantidad_pedido=cantidad_pedido
        )
        
        costo_mantenimiento = df['stock'].mean() * 0.5  # $0.50/unidad/día
        costo_stockout = df['stockout'].sum() * 100  # $100 por stockout
        costo_total = costo_mantenimiento + costo_stockout
        
        resultados.append({
            'nivel_servicio': 1 - df['stockout'].mean(),
            'costo_total': costo_total,
            'costo_mantenimiento': costo_mantenimiento,
            'costo_stockout': costo_stockout
        })
    
    return pd.DataFrame(resultados)

# Evaluar diferentes políticas
politicas = [(200, 300), (250, 350), (300, 400), (350, 450), (400, 500)]
print("=== EVALUACIÓN DE POLÍTICAS DE INVENTARIO ===")
print(f"{'ROP':>5s} {'EOQ':>5s} {'Servicio':>10s} {'Costo Total':>12s} {'Costo Mant':>12s} {'Costo SO':>12s}")
print("-" * 56)

for rop, eoq in politicas:
    df_pol = evaluar_politica(rop, eoq, 200)
    print(f"{rop:>5d} {eoq:>5d} {df_pol['nivel_servicio'].mean():>10.1%} "
          f"${df_pol['costo_total'].mean():>9,.0f} "
          f"${df_pol['costo_mantenimiento'].mean():>9,.0f} "
          f"${df_pol['costo_stockout'].mean():>9,.0f}")
```

---

## 26.10 Valor en Riesgo (VaR) para Inventarios

```python
"""Cálculo del Valor en Riesgo para nivel de inventario"""
np.random.seed(42)

n_sim = 10000

# Parámetros de demanda
demanda_diaria_media = 100
demanda_diaria_std = 25
lead_time = 5
stock_actual = 600

# Simular demanda durante lead time
demanda_lt = np.random.normal(
    demanda_diaria_media * lead_time,
    demanda_diaria_std * np.sqrt(lead_time),
    n_sim
)
demanda_lt = np.maximum(demanda_lt, 0)

# Stock final después de lead time
stock_final = stock_actual - demanda_lt

# VaR
var_95 = np.percentile(stock_final, 5)
var_99 = np.percentile(stock_final, 1)

print("=== VALOR EN RIESGO (VaR) DE INVENTARIO ===")
print(f"Stock actual: {stock_actual}")
print(f"VaR 95%: {var_95:.0f} unidades")
print(f"(95% de probabilidad de tener al menos {var_95:.0f} unidades después del lead time)")
print(f"VaR 99%: {var_99:.0f} unidades")
print(f"\nProbabilidad de stockout: {(stock_final < 0).mean():.1%}")
```

---

## 26.11 Resumen

| Concepto | Aplicación |
|----------|-----------|
| **Muestreo aleatorio** | Generar escenarios posibles |
| **Ley de Grandes Números** | Más simulaciones = mayor precisión |
| **Distribución de resultados** | Cuantiles, percentiles, riesgo |
| **Optimización** | Encontrar mejores parámetros |
| **VaR** | Valor en Riesgo (peor escenario) |

---

## Ejercicios Propuestos

1. **Ventas**: Simula 10,000 escenarios de ingresos anuales con 3 variables inciertas (precio, volumen, participación)
2. **Compras**: Evalúa el riesgo cambiario de una importación a 6 meses con TC actual 20.5 y volatilidad 12%
3. **Inventarios**: Encuentra el punto de reorden óptimo para 95% nivel de servicio usando simulación Montecarlo

---

[← Anterior](25-estadistica-bayesiana.md) | [Índice](index.md) | [Siguiente →](27-bootstrap.md)
