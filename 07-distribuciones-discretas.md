# Capítulo 7: Distribuciones Discretas de Probabilidad

[← Anterior](06-visualizacion-datos.md) | [Índice](index.md) | [Siguiente →](08-distribuciones-continuas.md)

---

## 7.1 Introducción

Una **distribución de probabilidad discreta** describe la probabilidad de cada valor posible de una variable aleatoria discreta. En el contexto empresarial, las distribuciones discretas modelan fenómenos como conteos de ventas, número de defectos, llegada de clientes, etc.

### Características
- La variable toma valores contables (0, 1, 2, ...)
- $0 \leq P(X = x) \leq 1$
- $\sum_x P(X = x) = 1$

---

## 7.2 Distribución Bernoulli

Modela un **experimento con dos resultados posibles**: éxito (1) o fracaso (0).

### Parámetros
- $p$: probabilidad de éxito

### Funciones
$$P(X = 1) = p, \quad P(X = 0) = 1 - p$$
$$E[X] = p, \quad Var(X) = p(1-p)$$

### Ejemplo en Ventas: Conversión de un Lead

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Un lead tiene 30% de probabilidad de convertirse en venta
p_conversion = 0.30

# Bernoulli: resultado de un solo lead
lead = stats.bernoulli(p_conversion)
print(f"P(venta) = {lead.pmf(1):.2f}")
print(f"P(no venta) = {lead.pmf(0):.2f}")
print(f"Valor esperado = {lead.mean():.2f}")
print(f"Varianza = {lead.var():.4f}")

# Simulación de 1000 leads
n_leads = 1000
resultados = lead.rvs(n_leads)
tasa_conversion = resultados.mean()
print(f"\nSimulación de {n_leads} leads:")
print(f"Conversiones: {resultados.sum()}")
print(f"Tasa de conversión real: {tasa_conversion:.3f} (teórica: {p_conversion})")
```

---

## 7.3 Distribución Binomial

Modela el **número de éxitos en n ensayos independientes** con la misma probabilidad de éxito.

### Parámetros
- $n$: número de ensayos
- $p$: probabilidad de éxito en cada ensayo

### Funciones
$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$
$$E[X] = np, \quad Var(X) = np(1-p)$$

### Ejemplo en Ventas: Conversión de Leads

```python
# 20 leads, cada uno con 30% de probabilidad de conversión
n_leads = 20
p_conv = 0.30

binomial = stats.binom(n=n_leads, p=p_conv)

# Probabilidad de exactamente k conversiones
for k in [0, 3, 5, 10, 15, 20]:
    prob = binomial.pmf(k)
    print(f"P(X = {k:2d} conversiones) = {prob:.4f} ({prob*100:.2f}%)")

# Probabilidades acumuladas
print(f"\nP(X ≤ 5) = {binomial.cdf(5):.4f} (≤ 5 conversiones)")
print(f"P(X > 8) = {1 - binomial.cdf(8):.4f} (más de 8 conversiones)")
print(f"P(5 ≤ X ≤ 10) = {binomial.cdf(10) - binomial.cdf(4):.4f}")

# Valor esperado
print(f"\nConversiones esperadas: {binomial.mean():.1f} de {n_leads}")
print(f"Desviación estándar: {binomial.std():.2f}")

# Visualización
k_values = np.arange(0, n_leads + 1)
probs = binomial.pmf(k_values)

plt.figure(figsize=(12, 5))
plt.bar(k_values, probs, color='steelblue', edgecolor='white', alpha=0.8)
plt.axvline(binomial.mean(), color='red', linestyle='--', linewidth=2, label=f'Media = {binomial.mean():.1f}')
plt.xlabel('Número de Conversiones')
plt.ylabel('Probabilidad')
plt.title(f'Distribución Binomial (n={n_leads}, p={p_conv})', fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('binomial_ventas.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Ejemplo en Compras: Defectos en Lote

```python
# Un lote de 100 productos, 2% de tasa de defectos
n_lote = 100
p_defecto = 0.02

binom_defectos = stats.binom(n_lote, p_defecto)

print("=== CONTROL DE CALIDAD - LOTES DE 100 UNIDADES ===")
print(f"Defectos esperados: {binom_defectos.mean():.1f}")
print(f"P(0 defectos) = {binom_defectos.pmf(0):.4f}")
print(f"P(≥ 1 defecto) = {1 - binom_defectos.pmf(0):.4f}")
print(f"P(≤ 3 defectos) = {binom_defectos.cdf(3):.4f}")
print(f"P(> 5 defectos) = {1 - binom_defectos.cdf(5):.4f}")

# Plan de muestreo: aceptar lote si ≤ 2 defectos
def prob_aceptar_lote(n_muestra, p_def, c=2):
    """Probabilidad de aceptar un lote con plan de muestreo (n, c)"""
    return stats.binom.cdf(c, n_muestra, p_def)

for p in [0.01, 0.02, 0.05, 0.10]:
    pa = prob_aceptar_lote(50, p, c=2)
    print(f"P(aceptar lote con {p*100:.0f}% defectos) = {pa:.4f}")
```

### Ejemplo en Inventarios: Probabilidad de Stockout

```python
# Demanda diaria sigue Binomial(20, 0.5) - 20 clientes, 50% compra
n_clientes = 20
p_compra = 0.5
stock_diario = 12  # unidades disponibles

binom_demanda = stats.binom(n_clientes, p_compra)

# Probabilidad de stockout (demanda > stock)
p_stockout = 1 - binom_demanda.cdf(stock_diario)
print(f"=== PROBABILIDAD DE STOCKOUT ===")
print(f"Stock disponible: {stock_diario} unidades")
print(f"P(stockout) = {p_stockout:.4f} ({p_stockout*100:.2f}%)")

# Nivel de servicio
nivel_servicio = binom_demanda.cdf(stock_diario)
print(f"Nivel de servicio: {nivel_servicio:.2%}")

# Stock necesario para 95% de nivel de servicio
for nivel in [0.90, 0.95, 0.99]:
    stock_needed = binom_demanda.ppf(nivel)
    print(f"Stock para {nivel:.0%} servicio: {stock_needed:.0f} unidades")
```

---

## 7.4 Distribución Poisson

Modela el **número de eventos que ocurren en un intervalo fijo** de tiempo o espacio, cuando los eventos ocurren de forma independiente a una tasa constante.

### Parámetros
- $\lambda$: tasa media de ocurrencia (eventos por intervalo)

### Funciones
$$P(X = k) = \frac{e^{-\lambda} \lambda^k}{k!}$$
$$E[X] = \lambda, \quad Var(X) = \lambda$$

### Ejemplo en Ventas: Llegada de Clientes

```python
# Una tienda recibe en promedio 15 clientes por hora
lambda_clientes = 15

poisson_clientes = stats.poisson(mu=lambda_clientes)

print("=== LLEGADA DE CLIENTES (Poisson(15)) ===")

for k in [10, 12, 15, 18, 20, 25]:
    print(f"P(X = {k:2d} clientes/hora) = {poisson_clientes.pmf(k):.4f}")

print(f"\nP(≤ 10 clientes) = {poisson_clientes.cdf(10):.4f}")
print(f"P(≥ 20 clientes) = {1 - poisson_clientes.cdf(19):.4f}")
print(f"P(entre 12 y 18) = {poisson_clientes.cdf(18) - poisson_clientes.cdf(11):.4f}")

# Simulación de un día (8 horas)
horas = 8
clientes_por_hora = poisson_clientes.rvs(horas)
print(f"\nClientes por hora (simulación): {clientes_por_hora}")
print(f"Total del día: {clientes_por_hora.sum()} clientes")
```

### Ejemplo en Compras: Órdenes de Compra

```python
# En promedio se reciben 8 órdenes de compra por día
lambda_ordenes = 8
poisson_ordenes = stats.poisson(mu=lambda_ordenes)

print("=== ÓRDENES DE COMPRA DIARIAS ===")
print(f"Órdenes esperadas: {poisson_ordenes.mean():.1f}")
print(f"P(0 órdenes) = {poisson_ordenes.pmf(0):.4f}")
print(f"P(> 12 órdenes) = {1 - poisson_ordenes.cdf(12):.4f}")

# Capacidad del equipo: 10 órdenes/día
capacidad = 10
p_saturacion = 1 - poisson_ordenes.cdf(capacidad)
print(f"P(saturación del equipo) = {p_saturacion:.4f}")
print(f"Días saturados/año ≈ {p_saturacion * 250:.0f} días")

# ¿Cuánta capacidad se necesita para 99% de cobertura?
for nivel in [0.90, 0.95, 0.99]:
    cap_needed = poisson_ordenes.ppf(nivel)
    print(f"Capacidad para {nivel:.0%}: {cap_needed:.0f} órdenes/día")
```

### Ejemplo en Inventarios: Demanda Diaria

```python
# Demanda diaria de un producto: Poisson(20)
lambda_demanda = 20
poisson_demanda = stats.poisson(mu=lambda_demanda)

print("=== ANÁLISIS DE INVENTARIO - POISSON ===")
print(f"Demanda media diaria: {lambda_demanda} unidades")
print(f"Desviación: {np.sqrt(lambda_demanda):.2f} unidades")

# Stock óptimo para diferentes niveles de servicio
stock_actual = 25
for nivel in [0.85, 0.90, 0.95, 0.99]:
    stock_optimo = poisson_demanda.ppf(nivel)
    print(f"\nNivel de servicio {nivel:.0%}:")
    print(f"  Stock necesario: {stock_optimo:.0f} unidades")
    print(f"  Stock de seguridad: {stock_optimo - lambda_demanda:.0f} unidades")
    print(f"  Costo de mantener: ${(stock_optimo - lambda_demanda) * 5:.0f}/día")

# Riesgo con stock actual
print(f"\nCon stock = {stock_actual}: Nivel servicio = {poisson_demanda.cdf(stock_actual):.2%}")
```

---

## 7.5 Distribución Geométrica

Modela el **número de ensayos hasta el primer éxito** (incluyendo el éxito).

### Parámetros
- $p$: probabilidad de éxito

### Funciones
$$P(X = k) = (1-p)^{k-1}p$$
$$E[X] = \frac{1}{p}, \quad Var(X) = \frac{1-p}{p^2}$$

### Ejemplo: Llamadas hasta primera venta

```python
# Probabilidad de venta por llamada: 20%
p_venta = 0.20
geometrica = stats.geom(p=p_venta)

print("=== LLAMADAS HASTA PRIMERA VENTA ===")
print(f"Llamadas esperadas: {geometrica.mean():.1f}")
print(f"P(venta en 1ra llamada) = {geometrica.pmf(1):.4f}")
print(f"P(venta en 3ra llamada) = {geometrica.pmf(3):.4f}")
print(f"P(≤ 5 llamadas) = {geometrica.cdf(5):.4f}")
print(f"P(> 10 llamadas) = {1 - geometrica.cdf(10):.4f}")

# Número de llamadas para tener 90% de probabilidad de éxito
for prob in [0.50, 0.75, 0.90, 0.95]:
    n_llamadas = geometrica.ppf(prob)
    print(f"Con {n_llamadas:.0f} llamadas: {prob:.0%} de haber hecho una venta")
```

---

## 7.6 Distribución Binomial Negativa

Modela el **número de fracasos antes de alcanzar r éxitos**.

### Parámetros
- $r$: número de éxitos deseados
- $p$: probabilidad de éxito

### Funciones
$$P(X = k) = \binom{k + r - 1}{k} p^r (1-p)^k$$
$$E[X] = \frac{r(1-p)}{p}, \quad Var(X) = \frac{r(1-p)}{p^2}$$

### Ejemplo: Ventas hasta alcanzar cuota

```python
# Un vendedor necesita 5 ventas. Probabilidad de venta por contacto: 25%
r_ventas = 5
p_exito = 0.25

bin_neg = stats.nbinom(n=r_ventas, p=p_exito)

print("=== VENTAS HASTA ALCANZAR CUOTA ===")
print(f"Meta: {r_ventas} ventas")
print(f"Contactos esperados (fracasos): {bin_neg.mean():.1f}")
print(f"Contactos totales esperados: {bin_neg.mean() + r_ventas:.1f}")

for k in [0, 5, 10, 15, 20]:
    prob = bin_neg.pmf(k)
    print(f"P(exactamente {k:2d} fracasos) = {prob:.4f}")

print(f"P(≤ 10 fracasos) = {bin_neg.cdf(10):.4f}")
print(f"P(≤ 15 contactos totales) = {bin_neg.cdf(10):.4f}")

# Probabilidad de alcanzar meta con 20 contactos (15 fracasos + 5 éxitos)
p_con_20_contactos = bin_neg.cdf(15)
print(f"\nProbabilidad de alcanzar meta en ≤ 20 contactos: {p_con_20_contactos:.2%}")
```

---

## 7.7 Distribución Hipergeométrica

Modela el número de éxitos en **n extracciones sin reemplazo** de una población finita.

### Parámetros
- $N$: tamaño de la población
- $K$: número de éxitos en la población
- $n$: tamaño de la muestra

### Funciones
$$P(X = k) = \frac{\binom{K}{k} \binom{N-K}{n-k}}{\binom{N}{n}}$$
$$E[X] = n\frac{K}{N}$$

### Ejemplo en Control de Calidad

```python
# Lote de 50 productos, 5 defectuosos. Muestreamos 10 sin reemplazo.
N = 50  # tamaño del lote
K = 5   # defectuosos
n = 10  # muestra

hipergeom = stats.hypergeom(M=N, n=K, N=n)

print("=== MUESTREO SIN REEMPLAZO - HIPERGEOMÉTRICA ===")
print(f"Lote: {N} unidades, {K} defectuosas")
print(f"Muestra: {n} unidades")

for k in range(0, 6):
    print(f"P({k} defectuosos en muestra) = {hipergeom.pmf(k):.4f}")

print(f"\nP(0 defectuosos) = {hipergeom.pmf(0):.4f}")
print(f"P(≥ 1 defectuoso) = {1 - hipergeom.pmf(0):.4f}")

# Comparación con binomial (con reemplazo)
binom = stats.binom(n, K/N)
print(f"\nComparación con Binomial (aprox):")
print(f"P(0) Hipergeométrica: {hipergeom.pmf(0):.4f}")
print(f"P(0) Binomial: {binom.pmf(0):.4f}")
print(f"La hipergeométrica da menor P(0) porque sin reemplazo")
print(f"aumenta la probabilidad al reducirse la población")
```

---

## 7.8 Comparación entre Distribuciones Discretas

| Distribución | Parámetros | Ejemplo en Negocios |
|-------------|------------|---------------------|
| **Bernoulli** | $p$ | ¿Un lead se convierte? |
| **Binomial** | $n, p$ | Conversiones de $n$ leads |
| **Poisson** | $\lambda$ | Clientes por hora |
| **Geométrica** | $p$ | Intentos hasta primera venta |
| **Bin. Negativa** | $r, p$ | Intentos hasta $r$ ventas |
| **Hipergeométrica** | $N, K, n$ | Defectos en muestra sin reemplazo |

### Selección de Distribución

```python
def seleccionar_distribucion(tipo_dato, descripcion):
    """Guía para seleccionar distribución discreta"""
    print(f"=== GUÍA DE SELECCIÓN ===")
    print(f"Tipo de dato: {tipo_dato}")
    print(f"Descripción: {descripcion}")
    print(f"Distribución recomendada: ", end="")
    
    if tipo_dato == "conteo sin límite superior":
        print("Poisson")
    elif tipo_dato == "conteo con límite superior":
        if descripcion == "sin reemplazo":
            print("Hipergeométrica")
        else:
            print("Binomial")
    elif tipo_dato == "número de intentos hasta éxito":
        if descripcion == "primer éxito":
            print("Geométrica")
        else:
            print("Binomial Negativa")
    elif tipo_dato == "ensayo único":
        print("Bernoulli")
    else:
        print("Consultar tabla de distribuciones")

seleccionar_distribucion("conteo sin límite superior", "clientes que llegan a tienda")
print()
seleccionar_distribucion("conteo con límite superior", "con reemplazo")
print()
seleccionar_distribucion("número de intentos hasta éxito", "primer éxito")
```

---

## 7.9 Ejemplo Integrador: Decisión de Inventario

```python
# Una tienda quiere determinar el stock óptimo para un producto
np.random.seed(42)

# Parámetros del problema
demanda_media_diaria = 25  # Poisson
lead_time = 3  # días
costo_unitario = 50  # $
precio_venta = 120  # $
costo_almacenamiento = 2  # $/unidad/día

# Distribución de demanda durante lead time (3 días)
lambda_lt = demanda_media_diaria * lead_time  # 75
demanda_lt = stats.poisson(mu=lambda_lt)

# Simular 1000 días de operación
n_dias = 1000
stock_inicial = 85  # Política actual

resultados = []
stock_actual = stock_inicial

for dia in range(n_dias):
    # Demanda del día
    demanda = stats.poisson.rvs(mu=demanda_media_diaria)
    
    # Ventas (limitadas por stock)
    ventas = min(demanda, stock_actual)
    stock_actual -= ventas
    
    # Stockout?
    stockout = demanda > stock_actual if stock_actual >= 0 else 0
    
    # Reordenar si es necesario
    if stock_actual <= 30:
        stock_actual += np.random.poisson(75)  # Llega pedido
    
    resultados.append({'ventas': ventas, 'stock': stock_actual, 'stockout': stockout})

df_sim = pd.DataFrame(resultados)
print(f"=== SIMULACIÓN DE INVENTARIO ({n_dias} DÍAS) ===")
print(f"Stock inicial: {stock_inicial}")
print(f"Ventas promedio diarias: {df_sim['ventas'].mean():.1f}")
print(f"Stock promedio: {df_sim['stock'].mean():.1f}")
print(f"Días con stockout: {df_sim['stockout'].sum()} ({df_sim['stockout'].mean():.1%})")
print(f"Nivel de servicio: {(1 - df_sim['stockout'].mean()):.1%}")
```

---

## 7.10 Resumen

- **Bernoulli**: Un solo ensayo éxito/fracaso
- **Binomial**: Conteo de éxitos en n ensayos independientes
- **Poisson**: Conteo de eventos en intervalo continuo
- **Geométrica**: Intentos hasta primer éxito
- **Binomial Negativa**: Intentos hasta r-ésimo éxito
- **Hipergeométrica**: Muestreo sin reemplazo
- La elección correcta depende de la naturaleza del proceso

---

## Ejercicios Propuestos

1. **Ventas**: Un call center tiene 30% de conversión. ¿Cuál es la probabilidad de hacer exactamente 8 ventas en 20 llamadas?
2. **Compras**: Un proveedor envía lotes de 200 con 3% defectuoso. Si se muestrean 20, ¿cuál es la probabilidad de encontrar 0 defectos?
3. **Inventarios**: La demanda diaria sigue Poisson(30). ¿Qué stock da 95% de nivel de servicio?

---

[← Anterior](06-visualizacion-datos.md) | [Índice](index.md) | [Siguiente →](08-distribuciones-continuas.md)
