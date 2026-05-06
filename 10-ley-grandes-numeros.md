# Capítulo 10: Ley de los Grandes Números

[← Anterior](09-teorema-limite-central.md) | [Índice](index.md) | [Siguiente →](11-estimacion-puntual.md)

---

## 10.1 Introducción

La **Ley de los Grandes Números (LGN)** es un teorema fundamental que establece que el promedio de una muestra converge al valor esperado a medida que el tamaño de la muestra aumenta. En términos simples:

> **"A más datos, más cerca de la verdad"**

### Diferencia entre LGN y TLC

| Ley de Grandes Números | Teorema del Límite Central |
|------------------------|---------------------------|
| El promedio muestral converge al valor esperado | La distribución del promedio muestral se vuelve Normal |
| Dice **hacia dónde** va | Dice **cómo se distribuye** alrededor de ese valor |
| No requiere distribución normal | Sí requiere que la distribución de la media sea normal |
| Se cumple para cualquier n grande | La velocidad de convergencia depende de la forma |

---

## 10.2 Ley Débil vs Ley Fuerte

### Ley Débil (Convergencia en Probabilidad)

$$\lim_{n \to \infty} P(|\bar{X}_n - \mu| > \epsilon) = 0 \quad \forall \epsilon > 0$$

"La probabilidad de que el promedio esté lejos de la media tiende a cero"

### Ley Fuerte (Convergencia Casi Segura)

$$P\left(\lim_{n \to \infty} \bar{X}_n = \mu\right) = 1$$

"El promedio converge a la media con probabilidad 1"

### Implicación Práctica

Para propósitos prácticos:
- La ley débil dice: "eventualmente estaremos cerca"
- La ley fuerte dice: "eventualmente estaremos en el punto exacto"
- En la práctica, la diferencia es la probabilidad de desviaciones grandes pero raras

---

## 10.3 Demostración Visual

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

np.random.seed(42)

def demostrar_lgn(distribucion, nombre, valor_esperado, n_max=10000, n_simulaciones=5):
    """Muestra cómo el promedio converge al valor esperado"""
    plt.figure(figsize=(14, 6))
    
    # Múltiples trayectorias
    for sim in range(n_simulaciones):
        datos = distribucion.rvs(n_max)
        promedios = np.cumsum(datos) / np.arange(1, n_max + 1)
        plt.plot(promedios, alpha=0.6, linewidth=0.8, label=f'Simulación {sim+1}')
    
    plt.axhline(y=valor_esperado, color='red', linestyle='--', 
                linewidth=2, label=f'Valor esperado = μ = {valor_esperado}')
    plt.xlabel('Tamaño de muestra (n)', fontweight='bold')
    plt.ylabel('Promedio acumulado (X̄ₙ)', fontweight='bold')
    plt.title(f'Ley de Grandes Números - {nombre}', fontweight='bold')
    plt.xscale('log')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f'lgn_{nombre.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()

print("=== LEY DE GRANDES NÚMEROS - DEMOSTRACIONES ===\n")

# Distribución Bernoulli (lanzamiento de moneda)
print("1. Lanzamiento de moneda (Bernoulli p=0.5)")
demostrar_lgn(stats.bernoulli(0.5), "Moneda (p=0.5)", 0.5)

# Distribución Exponencial
print("2. Tiempo entre llegadas (Exponencial λ=2)")
demostrar_lgn(stats.expon(scale=0.5), "Exponencial(λ=2)", 0.5)

# Distribución Poisson
print("3. Demanda diaria (Poisson λ=25)")
demostrar_lgn(stats.poisson(mu=25), "Poisson(λ=25)", 25)

# Distribución Normal
print("4. Precios (Normal μ=100, σ=20)")
demostrar_lgn(stats.norm(100, 20), "Normal(μ=100, σ=20)", 100)
```

---

## 10.4 Velocidad de Convergencia

La convergencia ocurre a una tasa de $O(1/\sqrt{n})$:

$$|\bar{X}_n - \mu| \approx \frac{\sigma}{\sqrt{n}}$$

```python
def analizar_convergencia(distribucion, nombre, valor_esperado, n_max=100000):
    """Analiza la velocidad de convergencia"""
    datos = distribucion.rvs(n_max)
    promedios = np.cumsum(datos) / np.arange(1, n_max + 1)
    errores = np.abs(promedios - valor_esperado)
    
    # Teórico: σ/√n
    sigma = distribucion.std()
    n_values = np.arange(1, n_max + 1)
    error_teorico = sigma / np.sqrt(n_values)
    
    print(f"=== VELOCIDAD DE CONVERGENCIA - {nombre} ===")
    for n in [10, 100, 1000, 10000, 100000]:
        print(f"n={n:6d}: Error real={errores[n-1]:.6f}, Error teórico (σ/√n)={error_teorico[n-1]:.6f}")

analizar_convergencia(stats.bernoulli(0.5), "Bernoulli(0.5)", 0.5)
print()
analizar_convergencia(stats.expon(scale=1), "Exponencial(1)", 1)
```

### Gráfico de Convergencia

```python
plt.figure(figsize=(12, 6))

for nombre, dist, mu in [('Bernoulli(0.5)', stats.bernoulli(0.5), 0.5),
                           ('Poisson(20)', stats.poisson(20), 20),
                           ('Normal(0,1)', stats.norm(), 0)]:
    datos = dist.rvs(10000)
    errores = np.abs(np.cumsum(datos) / np.arange(1, 10001) - mu)
    plt.loglog(range(1, 10001), errores, label=nombre, alpha=0.8)

# Referencia: 1/√n
n = np.arange(1, 10001)
plt.loglog(n, 1/np.sqrt(n), 'k--', linewidth=2, label='1/√n (referencia)')

plt.xlabel('n (escala log)', fontweight='bold')
plt.ylabel('|X̄ₙ - μ| (escala log)', fontweight='bold')
plt.title('Velocidad de Convergencia de la LGN', fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('velocidad_convergencia_lgn.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 10.5 Ejemplo en Ventas: Tasa de Conversión

```python
# Simulación de tasa de conversión de un vendedor
np.random.seed(42)

# Cada lead tiene 25% de probabilidad de conversión
p_real = 0.25
n_leads_total = 5000

# Simulamos leads uno a uno
leads = stats.bernoulli.rvs(p_real, size=n_leads_total)
tasa_acumulada = np.cumsum(leads) / np.arange(1, n_leads_total + 1)

print("=== TASA DE CONVERSIÓN ACUMULADA ===")
print(f"Tasa real (p): {p_real:.2%}")

puntos = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
for n in puntos:
    print(f"Lead #{n:4d}: Tasa={tasa_acumulada[n-1]:.2%}, "
          f"Error=|{tasa_acumulada[n-1] - p_real:.2%}|")

# Implicación: ¿Cuántos leads necesito para tener una estimación confiable?
print(f"\n¿Cuántos leads para error < 2%?")
for n in [100, 500, 1000, 2000]:
    error = np.abs(tasa_acumulada[n-1] - p_real)
    print(f"  n={n:4d}: error={error:.2%}")
```

---

## 10.6 Ejemplo en Compras: Precio Promedio de Mercado

```python
# Determinación del precio justo de mercado mediante cotizaciones
np.random.seed(42)

# Los precios siguen una distribución con algunos valores extremos
precios_reales = stats.gamma.rvs(a=3, scale=30, size=2000) + 20  # Gamma(3,30)+20
precio_real_medio = precios_reales.mean()

print("=== PRECIO PROMEDIO DE MERCADO ===")
print(f"Precio real medio: ${precio_real_medio:.2f}")

# Simulamos el proceso de cotizar
cotizaciones_acumuladas = np.cumsum(np.random.choice(precios_reales, size=500, replace=True))
precios_promedio = cotizaciones_acumuladas / np.arange(1, 501)

for n in [3, 5, 10, 20, 50, 100, 200, 500]:
    error = abs(precios_promedio[n-1] - precio_real_medio)
    print(f"n={n:3d} cotizaciones: Promedio=${precios_promedio[n-1]:.2f}, Error=${error:.2f}")

# Regla práctica
print(f"\nRegla: con 10 cotizaciones el error suele ser < 10%")
print(f"Con 30 cotizaciones el error suele ser < 5%")
```

---

## 10.7 Ejemplo en Inventarios: Demanda Promedio

```python
# Estimación de la demanda promedio diaria
np.random.seed(42)

# Demanda real sigue Poisson(λ=35)
lambda_real = 35
dias_simulados = 2000

demandas = stats.poisson.rvs(mu=lambda_real, size=dias_simulados)
demanda_acum = np.cumsum(demandas) / np.arange(1, dias_simulados + 1)

print("=== ESTIMACIÓN DE DEMANDA PROMEDIO ===")
print(f"Demanda real (λ): {lambda_real} unidades/día")

periodos = [7, 14, 30, 60, 90, 180, 365, 1000]
for dia in periodos:
    print(f"Día {dia:4d}: Demanda prom={demanda_acum[dia-1]:.1f}, "
          f"Error={abs(demanda_acum[dia-1] - lambda_real):.2f}")

# Implicación: stock de seguridad
print(f"\nImplicación para inventario:")
for dias in [7, 30, 90]:
    demanda_estimada = demanda_acum[dias-1]
    error = abs(demanda_estimada - lambda_real)
    impacto_ss = error * 1.645 * 5  # Z * error * lead_time
    print(f"  Con {dias:2d} días: demanda≈{demanda_estimada:.0f}, "
          f"error≈{error:.2f}, sobrecosto SS≈${impacto_ss:.0f}")
```

---

## 10.8 La Falacia del Jugador (Gambler's Fallacy)

Un error común relacionado con la LGN es la **falacia del jugador**: creer que después de una racha de resultados, el siguiente será diferente para "compensar".

```python
def demostrar_falacia_jugador(n_lanzamientos=1000):
    """Demuestra que los eventos son independientes (falacia del jugador)"""
    resultados = np.random.choice(['Cara', 'Cruz'], n_lanzamientos)
    
    # Buscamos rachas de 3 o más caras consecutivas
    rachas = []
    racha_actual = 0
    
    for r in resultados:
        if r == 'Cara':
            racha_actual += 1
        else:
            if racha_actual >= 3:
                rachas.append(racha_actual)
            racha_actual = 0
    
    # Probabilidad de cruz después de racha de 3 caras
    caras_consecutivas = 0
    cruz_despues = []
    
    for r in resultados:
        if r == 'Cara':
            caras_consecutivas += 1
        else:
            if caras_consecutivas >= 3:
                cruz_despues.append(1)  # Sí, salió cruz
            else:
                cruz_despues.append(0)  # No aplica
            caras_consecutivas = 0
    
    print("=== FALACIA DEL JUGADOR ===")
    print(f"Total lanzamientos: {n_lanzamientos}")
    if rachas:
        print(f"Rachas de ≥3 caras: {len(rachas)}")
        print(f"Probabilidad de cruz después de 3+ caras: ", end="")
        if len(cruz_despues) > 0:
            prob = np.mean(cruz_despues)
            print(f"{prob:.2%} (debería ser ~50%)")
        else:
            print("No hay suficientes datos")
    print(f"La moneda no tiene memoria - cada lanzamiento es independiente")

demostrar_falacia_jugador(5000)
```

---

## 10.9 Relación entre LGN y TLC

```python
# Visualización conjunta: LGN (convergencia) y TLC (distribución)
np.random.seed(42)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

n_values = [5, 20, 100, 500, 2000, 10000]
n_sim = 5000
dist = stats.expon(scale=1)

for i, n in enumerate(n_values):
    ax = axes[i // 3, i % 3]
    
    # Generamos n_sim promedios de tamaño n
    muestras = dist.rvs(size=(n_sim, n))
    promedios = muestras.mean(axis=1)
    
    # Histograma
    ax.hist(promedios, bins=40, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    
    # Media teórica (siempre 1 para exponencial)
    ax.axvline(x=1, color='red', linestyle='--', linewidth=2)
    
    # Normal aproximada
    x = np.linspace(promedios.min(), promedios.max(), 100)
    ax.plot(x, stats.norm.pdf(x, 1, 1/np.sqrt(n)), 'g-', linewidth=2, label='Normal')
    
    ax.set_title(f'n = {n}', fontweight='bold')
    ax.set_xlabel('Media muestral')
    ax.set_ylabel('Densidad')

fig.suptitle('LGN + TLC: Convergencia y Distribución', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('lgn_tlc_conjunto.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 10.10 Aplicaciones Prácticas en Negocios

| Aplicación | Proceso | n mínimo | Decisión |
|-----------|---------|----------|----------|
| Tasa de conversión | Conteo de ventas/leads | 100-500 | Evaluar vendedores |
| Demanda promedio | Conteo de ventas diarias | 30-90 | Stock de seguridad |
| Precio de mercado | Cotizaciones a proveedores | 10-30 | Negociación |
| Tasa de defectos | Inspección de lotes | 100-500 | Aceptar/rechazar lote |
| Tiempo de entrega | Historial de pedidos | 20-50 | Lead time planificación |
| Satisfacción cliente | Encuestas | 50-200 | Métricas de calidad |

### Limitaciones de la LGN

1. **No acelera milagros**: Si la varianza es enorme, necesitas muchas más observaciones
2. **No elimina sesgos**: Si los datos tienen sesgo sistemático, más datos no ayudan
3. **No garantiza convergencia rápida**: Algunas distribuciones (Pareto, Cauchy) violan las condiciones
4. **Independencia**: Si los datos están correlacionados, la convergencia es más lenta

```python
# Demostración: la LGN falla con distribución de Cauchy (varianza infinita)
cauchy = stats.cauchy()
datos_cauchy = cauchy.rvs(10000)
promedios_cauchy = np.cumsum(datos_cauchy) / np.arange(1, 10001)

plt.figure(figsize=(12, 5))
plt.plot(promedios_cauchy, linewidth=0.8)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('n')
plt.ylabel('Promedio acumulado')
plt.title('LGN NO aplica: Cauchy (varianza infinita) - ¡El promedio NO converge!', 
          fontweight='bold')
plt.grid(alpha=0.3)
plt.savefig('lgn_cauchy.png', dpi=150, bbox_inches='tight')
plt.show()

print("La distribución de Cauchy no tiene media ni varianza definidas.")
print("La LGN NO se cumple. El promedio no converge.")
```

---

## 10.11 Resumen

- La **LGN** garantiza que el promedio muestral converge al valor esperado
- La convergencia es a tasa $O(1/\sqrt{n})$
- La **Ley Débil** dice que la probabilidad de gran error tiende a 0
- La **Ley Fuerte** dice que la convergencia es casi segura
- NO confundir con la falacia del jugador (independencia de eventos)
- NO aplica a distribuciones sin varianza finita (Cauchy)
- Es la base de los seguros, la calidad y la inferencia estadística

---

## Ejercicios Propuestos

1. **Ventas**: Simula la tasa de conversión acumulada de un lead con p=0.15. ¿Cuántos leads necesitas para error < 1%?
2. **Compras**: Genera 1000 precios de proveedor desde una Gamma(α=4, β=20) y muestra cómo el promedio converge al valor esperado
3. **Inventarios**: Usando demanda Poisson(λ=50), demuestra que la demanda promedio diaria converge a λ y calcula el error con n=7, 30, 365

---

[← Anterior](09-teorema-limite-central.md) | [Índice](index.md) | [Siguiente →](11-estimacion-puntual.md)
