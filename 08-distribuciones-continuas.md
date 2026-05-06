# Capítulo 8: Distribuciones Continuas de Probabilidad

[← Anterior](07-distribuciones-discretas.md) | [Índice](index.md) | [Siguiente →](09-teorema-limite-central.md)

---

## 8.1 Introducción

Las **distribuciones continuas** modelan variables que pueden tomar cualquier valor en un intervalo real. A diferencia de las discretas, la probabilidad de un valor exacto es cero; solo tiene sentido hablar de probabilidad en intervalos.

### Conceptos Clave

- **Función de Densidad (PDF)**: $f(x)$, área bajo la curva = 1
- **Función Acumulada (CDF)**: $F(x) = P(X \leq x) = \int_{-\infty}^{x} f(t) dt$
- **Cuantil**: $F^{-1}(p)$ valor x tal que $P(X \leq x) = p$

---

## 8.2 Distribución Normal (Gaussiana)

La distribución más importante en estadística. Describe fenómenos naturales y, por el Teorema del Límite Central, aproxima sumas de variables aleatorias.

### Parámetros
- $\mu$: media (centro)
- $\sigma$: desviación estándar (dispersión)

### Función de Densidad
$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}$$

### Normal Estándar
$$Z = \frac{X - \mu}{\sigma} \sim N(0,1)$$

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

# --- Distribución Normal ---
mu, sigma = 1000, 200
normal = norm(loc=mu, scale=sigma)

print("=== DISTRIBUCIÓN NORMAL ===")
print(f"Media (μ): {mu}")
print(f"Desviación estándar (σ): {sigma}")

# Probabilidades en intervalos clave
print(f"\nP(X < 800) = {normal.cdf(800):.4f}")
print(f"P(800 < X < 1200) = {normal.cdf(1200) - normal.cdf(800):.4f}")
print(f"P(X > 1400) = {1 - normal.cdf(1400):.4f}")

# Cuantiles
print(f"\nPercentil 10: {normal.ppf(0.10):.0f}")
print(f"Percentil 25 (Q1): {normal.ppf(0.25):.0f}")
print(f"Percentil 50 (Mediana): {normal.ppf(0.50):.0f}")
print(f"Percentil 75 (Q3): {normal.ppf(0.75):.0f}")
print(f"Percentil 90: {normal.ppf(0.90):.0f}")
print(f"Percentil 95: {normal.ppf(0.95):.0f}")
print(f"Percentil 99: {normal.ppf(0.99):.0f}")

# Regla empírica
for k in [1, 2, 3]:
    prob = normal.cdf(mu + k*sigma) - normal.cdf(mu - k*sigma)
    print(f"P(μ ± {k}σ) = {prob:.4f}")
```

### Ejemplo en Ventas: Pronóstico de Demanda

```python
print("=== PRONÓSTICO DE VENTAS SEMANALES ===")
ventas_semanales = norm(loc=50000, scale=8000)

# Presupuesto base: $50,000
print(f"P(ventas < 40000) = {ventas_semanales.cdf(40000):.2%}")
print(f"P(ventas > 60000) = {1 - ventas_semanales.cdf(60000):.2%}")

# Presupuesto optimista: exceder $65,000
p_exito = 1 - ventas_semanales.cdf(65000)
print(f"P(ventas > $65,000) = {p_exito:.2%}")

# Metas para diferentes probabilidades
for prob in [0.50, 0.75, 0.90, 0.95]:
    meta = ventas_semanales.ppf(prob)
    print(f"Meta con {prob:.0%} confianza: ${meta:,.0f}")

# Calcular probabilidad de cumplir presupuesto
presupuesto = 45000
p_cumplir = 1 - ventas_semanales.cdf(presupuesto)
print(f"\nP(cumplir presupuesto ${presupuesto:,}) = {p_cumplir:.2%}")
```

### Ejemplo en Compras: Negociación de Precios

```python
# Precios históricos de un insumo: Normal(120, 15)
precios_insumo = norm(loc=120, scale=15)

print("=== NEGOCIACIÓN DE PRECIOS ===")
precio_objetivo = 100
p_obtener_precio = precios_insumo.cdf(precio_objetivo)
print(f"P(precio ≤ ${precio_objetivo}) = {p_obtener_precio:.2%}")

# Precio que solo el 10% de las veces se supera
precio_max = precios_insumo.ppf(0.90)
print(f"Precio máximo (90% confianza): ${precio_max:.2f}")

# Probabilidad de ahorro
ahorro_objetivo = 15
precio_meta = 120 - ahorro_objetivo
p_ahorro = precios_insumo.cdf(precio_meta)
print(f"P(ahorro ≥ ${ahorro_objetivo}) = {p_ahorro:.2%}")

# Simulación de negociación (1000 proveedores)
precios_simulados = precios_insumo.rvs(1000)
aceptados = precios_simulados[precios_simulados <= 105]
print(f"\nDe 1000 cotizaciones, {len(aceptados)} estarían ≤ $105")
print(f"Tasa de aceptación: {len(aceptados)/1000:.1%}")
```

### Ejemplo en Inventarios: Demanda en Lead Time

```python
# Demanda durante lead time sigue Normal(300, 50)
demanda_lt = norm(loc=300, scale=50)

print("=== INVENTARIOS - DEMANDA EN LEAD TIME ===")

# Stock de seguridad para 95% nivel de servicio
zs = {0.90: 1.28, 0.95: 1.645, 0.99: 2.326}
demanda_prom = 300
sigma_lt = 50

for nivel, z in zs.items():
    ss = z * sigma_lt
    punto_reorden = demanda_prom + ss
    print(f"Nivel servicio {nivel:.0%}:")
    print(f"  Z = {z:.3f}")
    print(f"  Stock seguridad: {ss:.0f} unidades")
    print(f"  Punto reorden: {punto_reorden:.0f} unidades")
    print()

# Costo del stock de seguridad
costo_mantenimiento = 3  # $/unidad/año
for nivel in [0.90, 0.95, 0.99]:
    ss = stats.norm.ppf(nivel) * sigma_lt
    costo_ss = ss * costo_mantenimiento
    print(f"SS para {nivel:.0%}: {ss:.0f} uds, Costo: ${costo_ss:.0f}/año")
```

---

## 8.3 Distribución Uniforme Continua

Todos los valores en un intervalo $[a, b]$ tienen igual probabilidad.

### Parámetros
- $a$: límite inferior
- $b$: límite superior

### Funciones
$$f(x) = \frac{1}{b-a}, \quad a \leq x \leq b$$
$$E[X] = \frac{a+b}{2}, \quad Var(X) = \frac{(b-a)^2}{12}$$

```python
# Tiempo de entrega uniforme entre 3 y 7 días
a, b = 3, 7
uniforme = stats.uniform(loc=a, scale=b-a)

print("=== DISTRIBUCIÓN UNIFORME ===")
print(f"Lead time: Uniforme({a}, {b})")
print(f"Media: {uniforme.mean():.1f} días")
print(f"P(entrega en ≤ 4 días) = {uniforme.cdf(4):.2%}")
print(f"P(entrega en ≥ 6 días) = {1 - uniforme.cdf(6):.2%}")

# Percentiles
for p in [0.25, 0.50, 0.75]:
    print(f"Percentil {p:.0%}: {uniforme.ppf(p):.1f} días")
```

---

## 8.4 Distribución Exponencial

Modela el **tiempo entre eventos** en un proceso Poisson. Describe tiempos de espera, vida de componentes, etc.

### Parámetros
- $\lambda$: tasa de ocurrencia (eventos por unidad de tiempo)
- A veces parametrizada con $\beta = 1/\lambda$ (tiempo medio entre eventos)

### Funciones
$$f(x) = \lambda e^{-\lambda x}, \quad x \geq 0$$
$$F(x) = 1 - e^{-\lambda x}$$
$$E[X] = \frac{1}{\lambda}, \quad Var(X) = \frac{1}{\lambda^2}$$

**Propiedad clave (falta de memoria):**
$$P(X > s + t | X > s) = P(X > t)$$

```python
# Tiempo entre llegadas de clientes: 15 clientes/hora → λ = 15, β = 1/15 horas
lambda_clientes = 15  # clientes por hora
beta = 1 / lambda_clientes  # horas entre clientes
exponencial = stats.expon(scale=beta)

print("=== DISTRIBUCIÓN EXPONENCIAL ===")
print(f"Tasa (λ): {lambda_clientes} clientes/hora")
print(f"Tiempo medio entre llegadas: {beta*60:.1f} minutos")

# Probabilidades
print(f"P(tiempo ≤ 2 min) = {exponencial.cdf(2/60):.4f}")
print(f"P(tiempo ≥ 5 min) = {1 - exponencial.cdf(5/60):.4f}")
print(f"P(2 ≤ tiempo ≤ 5 min) = {exponencial.cdf(5/60) - exponencial.cdf(2/60):.4f}")

# Falta de memoria
t_espera = 3/60  # 3 minutos
p_mas_5min = 1 - exponencial.cdf(5/60)
p_mas_8min_dado_mas_3min = (1 - exponencial.cdf(8/60)) / (1 - exponencial.cdf(3/60))
print(f"\nPropiedad de falta de memoria:")
print(f"P(más 5 min desde inicio) = {p_mas_5min:.4f}")
print(f"P(más 5 min extra | ya esperé 3 min) = {p_mas_8min_dado_mas_3min:.4f}")
```

### Ejemplo en Ventas: Tiempo entre Compras

```python
# En promedio, un cliente compra cada 30 minutos (2 por hora)
tasa_compras = 2  # por hora
tiempo_entre_compras = stats.expon(scale=1/tasa_compras)

print("=== TIEMPO ENTRE COMPRAS ===")
print(f"Tiempo medio entre compras: {60/tasa_compras:.0f} minutos")
print(f"P(próxima compra en ≤15 min) = {tiempo_entre_compras.cdf(0.25):.4f}")
print(f"P(próxima compra en >1 hora) = {1 - tiempo_entre_compras.cdf(1):.4f}")

# Configuración de turnos
for p in [0.50, 0.75, 0.90, 0.95]:
    tiempo = tiempo_entre_compras.ppf(p) * 60  # en minutos
    print(f"{p:.0%} de las compras ocurren en ≤ {tiempo:.1f} minutos")
```

### Ejemplo en Compras: Tiempo entre Entregas

```python
# Tiempo entre entregas de proveedor: media 5 días
tasa_entregas = 1/5  # entregas por día
entregas = stats.expon(scale=5)

print("=== TIEMPO ENTRE ENTREGAS ===")
print(f"Tiempo medio: 5 días")
print(f"P(entrega en ≤3 días) = {entregas.cdf(3):.4f}")
print(f"P(tocar esperar >7 días) = {1 - entregas.cdf(7):.4f}")

# Planificación: ¿cada cuánto revisar?
for p in [0.80, 0.90, 0.95]:
    dias = entregas.ppf(p)
    print(f"{p:.0%} de entregas llegan en ≤ {dias:.1f} días")
```

---

## 8.5 Distribución Gamma

Generaliza la exponencial. Modela el **tiempo hasta que ocurren $\alpha$ eventos**.

### Parámetros
- $\alpha$ (shape): número de eventos
- $\beta$ (scale): escala ($1/\lambda$)

### Funciones
$$f(x) = \frac{1}{\Gamma(\alpha)\beta^\alpha} x^{\alpha-1} e^{-x/\beta}, \quad x \geq 0$$
$$E[X] = \alpha\beta, \quad Var(X) = \alpha\beta^2$$

```python
# Tiempo hasta recibir 5 entregas (cada entrega: exponencial con media 2 días)
alpha_entregas = 5  # número de entregas
beta_entrega = 2    # tiempo medio por entrega

gamma_entregas = stats.gamma(a=alpha_entregas, scale=beta_entrega)

print("=== DISTRIBUCIÓN GAMMA ===")
print(f"Tiempo hasta {alpha_entregas} entregas")
print(f"Tiempo esperado: {gamma_entregas.mean():.1f} días")
print(f"Desviación: {gamma_entregas.std():.1f} días")
print(f"P(tiempo ≤ 8 días) = {gamma_entregas.cdf(8):.4f}")
print(f"P(tiempo > 15 días) = {1 - gamma_entregas.cdf(15):.4f}")

# Tiempo para 95% de confianza
t95 = gamma_entregas.ppf(0.95)
print(f"95% de las veces las {alpha_entregas} entregas")
print(f"llegan en ≤ {t95:.1f} días (vs media de {gamma_entregas.mean():.1f})")
```

---

## 8.6 Distribución Beta

Modela proporciones y probabilidades en el intervalo $[0, 1]$. Muy útil en inferencia bayesiana.

### Parámetros
- $\alpha$ (shape1): éxitos + 1
- $\beta$ (shape2): fracasos + 1

### Funciones
$$f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}, \quad 0 \leq x \leq 1$$
$$E[X] = \frac{\alpha}{\alpha + \beta}$$

```python
# Probabilidad de conversión: observamos 30 éxitos en 100 intentos
alpha_prior = 30 + 1
beta_prior = 70 + 1

beta_dist = stats.beta(a=alpha_prior, b=beta_prior)

print("=== DISTRIBUCIÓN BETA (CREENCIA SOBRE CONVERSIÓN) ===")
print(f"Éxitos observados: {alpha_prior-1}, Fracasos: {beta_prior-1}")
print(f"Tasa de conversión esperada: {beta_dist.mean():.2%}")
print(f"Desviación: {beta_dist.std():.2%}")

# Intervalo de credibilidad
print(f"Percentil 2.5: {beta_dist.ppf(0.025):.2%}")
print(f"Percentil 97.5: {beta_dist.ppf(0.975):.2%}")
print(f"IC 95%: [{beta_dist.ppf(0.025):.2%}, {beta_dist.ppf(0.975):.2%}]")

# Probabilidad de que la conversión supere cierto umbral
print(f"\nP(conversión > 25%) = {1 - beta_dist.cdf(0.25):.2%}")
print(f"P(conversión > 35%) = {1 - beta_dist.cdf(0.35):.2%}")
print(f"P(conversión > 40%) = {1 - beta_dist.cdf(0.40):.2%}")
```

### Ejemplo: Probabilidad de Stockout (Bayesiana)

```python
# Estimación de probabilidad de stockout con datos históricos
# De 200 días, hubo stockout en 12
exitos = 12
fracasos = 188

beta_stockout = stats.beta(exitos + 1, fracasos + 1)
print("=== PROBABILIDAD DE STOCKOUT (ENFOQUE BAYESIANO) ===")
print(f"Días con stockout: {exitos}/200")
print(f"Tasa esperada: {beta_stockout.mean():.2%}")
print(f"IC 95%: [{beta_stockout.ppf(0.025):.2%}, {beta_stockout.ppf(0.975):.2%}]")
print(f"P(tasa real < 10%) = {beta_stockout.cdf(0.10):.2%}")
```

---

## 8.7 Distribución Chi-Cuadrado ($\chi^2$)

Relacionada con la varianza muestral. Fundamental en pruebas de hipótesis.

### Parámetros
- $k$: grados de libertad

$$E[X] = k, \quad Var(X) = 2k$$

```python
# Distribución chi-cuadrado con diferentes gl
for k in [1, 2, 5, 10]:
    chi2 = stats.chi2(df=k)
    print(f"χ²({k}): Media={chi2.mean()}, Var={chi2.var()}, P95={chi2.ppf(0.95):.2f}")
```

---

## 8.8 Distribución t de Student

Similar a la normal pero con colas más pesadas. Se usa cuando $\sigma$ es desconocido y se estima con $s$.

### Parámetros
- $\nu$: grados de libertad

```python
# Comparación t vs Normal
t5 = stats.t(df=5)
t30 = stats.t(df=30)
normal = stats.norm()

print("=== t-STUDENT vs NORMAL ESTÁNDAR ===")
print(f"{'Percentil':>10s} {'t(5)':>8s} {'t(30)':>8s} {'Normal':>8s}")
for p in [0.90, 0.95, 0.975, 0.99]:
    print(f"{p*100:>9.0f}% {t5.ppf(p):>8.3f} {t30.ppf(p):>8.3f} {normal.ppf(p):>8.3f}")

print("\nLas colas de t son más pesadas (valores extremos más probables)")
```

---

## 8.9 Distribución F de Fisher

Se usa en ANOVA y comparación de varianzas.

### Parámetros
- $d_1, d_2$: grados de libertad del numerador y denominador

```python
# Comparación de varianzas entre dos procesos
f_dist = stats.f(dfn=10, dfd=20)
print(f"F(10,20): Media={f_dist.mean():.2f}, P95={f_dist.ppf(0.95):.2f}, P99={f_dist.ppf(0.99):.2f}")
```

---

## 8.10 Ejemplo Integrador: Gestión de Inventarios con Distribuciones Continuas

```python
# Sistema completo de inventario con múltiples distribuciones
np.random.seed(42)

# 1. Demanda diaria: Normal(100, 20)
demanda = stats.norm(100, 20)

# 2. Lead time: Exponencial(7) - promedio 7 días
lead_time = stats.expon(scale=7)

# 3. Tiempo entre pedidos: Gamma(3, 5)
tiempo_pedido = stats.gamma(a=3, scale=5)

# Simulación
n_periodos = 500
stock = 800  # stock inicial
pedidos = []

for i in range(n_periodos):
    d = max(0, demanda.rvs())  # demanda del período
    stock -= d
    
    if stock < 200:  # punto de reorden
        lt = max(1, lead_time.rvs())  # lead time del pedido
        cantidad = 600
        pedidos.append({'periodo': i, 'llegada': i + lt, 'cantidad': cantidad})
        stock += cantidad

print("=== SIMULACIÓN COMPLETA DE INVENTARIO ===")
print(f"Períodos simulados: {n_periodos}")
print(f"Pedidos generados: {len(pedidos)}")
print(f"Días promedio entre pedidos: {n_periodos/len(pedidos):.1f}")

# Análisis de nivel de servicio
stock_final = stock
print(f"\nStock final: {stock:.0f} unidades")
print(f"Demanda promedio: {demanda.mean():.0f} unidades/día")
```

---

## 8.11 Resumen

| Distribución | Parámetros | Uso en Negocios |
|-------------|------------|-----------------|
| **Normal** | $\mu, \sigma$ | Demanda, precios, errores de pronóstico |
| **Uniforme** | $a, b$ | Tiempos de entrega sin información |
| **Exponencial** | $\lambda$ | Tiempo entre eventos, vida de componentes |
| **Gamma** | $\alpha, \beta$ | Tiempo hasta k eventos, colas de espera |
| **Beta** | $\alpha, \beta$ | Proporciones, tasas de conversión |
| **Chi-cuadrado** | $k$ | Pruebas de bondad de ajuste, varianza |
| **t-Student** | $\nu$ | Inferencia con varianza desconocida |
| **F-Fisher** | $d_1, d_2$ | ANOVA, comparación de varianzas |

---

## Ejercicios Propuestos

1. **Ventas**: Si las ventas mensuales siguen Normal(50000, 10000), ¿cuál es la probabilidad de vender más de $60,000?
2. **Compras**: El tiempo entre entregas de un proveedor es exponencial con media 4 días. ¿Probabilidad de que llegue en ≤ 2 días?
3. **Inventarios**: Modela la demanda en lead time con distribución Gamma si la demanda diaria es Gamma(5, 20) y el lead time es 7 días

---

[← Anterior](07-distribuciones-discretas.md) | [Índice](index.md) | [Siguiente →](09-teorema-limite-central.md)
