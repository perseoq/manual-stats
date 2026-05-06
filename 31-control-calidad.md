# Capítulo 31: Control de Calidad (SPC)

[← Anterior](30-arboles-decision.md) | [Índice](index.md) | [Siguiente →](32-muestreo.md)

---

## 31.1 Introducción

El **Control Estadístico de Procesos (SPC)** utiliza métodos estadísticos para monitorear y controlar la calidad de procesos productivos y de servicios. Su objetivo es detectar **variaciones anormales** antes de que generen productos defectuosos.

### Tipos de Variación

| Tipo | Fuente | Característica |
|------|--------|---------------|
| **Causas Comunes** | Inherentes al proceso | Variación natural, estable |
| **Causas Especiales** | Factores externos | Variación anormal, detectable |

### Gráficos de Control

Monitorean la estabilidad del proceso mediante:

$$LSC = \mu + 3\sigma \quad \text{(Límite Superior)}$$
$$LIC = \mu - 3\sigma \quad \text{(Límite Inferior)}$$

---

## 31.2 Gráficos para Variables Continuas (X-bar, R, S)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

np.random.seed(42)

class GraficoControl:
    """Implementación de gráficos de control SPC"""
    
    @staticmethod
    def grafico_xbar_r(datos, n_muestra, nombre="Proceso"):
        """Gráfico X-bar y R para subgrupos"""
        # Organizar en subgrupos
        n_subgrupos = len(datos) // n_muestra
        subgrupos = datos[:n_subgrupos * n_muestra].reshape(n_subgrupos, n_muestra)
        
        medias = subgrupos.mean(axis=1)
        rangos = subgrupos.ptp(axis=1)
        
        # Límites de control
        media_general = np.mean(medias)
        rango_medio = np.mean(rangos)
        
        # Constantes para n_muestra (tabla A2, D3, D4)
        A2 = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}
        D3 = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}
        D4 = {2: 3.267, 3: 2.575, 4: 2.282, 5: 2.115, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}
        
        n_actual = min(n_muestra, max(A2.keys()))
        a2 = A2[n_actual]
        d3 = D3[n_actual]
        d4 = D4[n_actual]
        
        # Límites para X-bar
        lsc_x = media_general + a2 * rango_medio
        lic_x = media_general - a2 * rango_medio
        
        # Límites para R
        lsc_r = d4 * rango_medio
        lic_r = d3 * rango_medio
        
        # Gráficos
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # X-bar
        axes[0].plot(medias, 'b-', linewidth=1.5, marker='o', markersize=4)
        axes[0].axhline(y=media_general, color='green', linestyle='-', linewidth=2, label='Media')
        axes[0].axhline(y=lsc_x, color='red', linestyle='--', linewidth=2, label='LSC')
        axes[0].axhline(y=lic_x, color='red', linestyle='--', linewidth=2, label='LIC')
        axes[0].fill_between(range(len(medias)), lic_x, lsc_x, alpha=0.1, color='green')
        axes[0].set_title(f'Gráfico X-bar - {nombre}', fontweight='bold')
        axes[0].set_ylabel('Media del subgrupo')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # R
        axes[1].plot(rangos, 'b-', linewidth=1.5, marker='s', markersize=4)
        axes[1].axhline(y=rango_medio, color='green', linestyle='-', linewidth=2, label='R medio')
        axes[1].axhline(y=lsc_r, color='red', linestyle='--', linewidth=2, label='LSC')
        axes[1].axhline(y=lic_r, color='red', linestyle='--', linewidth=2, label='LIC')
        axes[1].fill_between(range(len(rangos)), lic_r, lsc_r, alpha=0.1, color='green')
        axes[1].set_title(f'Gráfico R - {nombre}', fontweight='bold')
        axes[1].set_xlabel('Subgrupo')
        axes[1].set_ylabel('Rango del subgrupo')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'spc_{nombre.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return {'media': media_general, 'rango': rango_medio, 
                'lsc_x': lsc_x, 'lic_x': lic_x, 'lsc_r': lsc_r, 'lic_r': lic_r}

# Datos de proceso (peso de producto en gramos)
np.random.seed(42)
proceso_estable = np.random.normal(500, 5, 200)  # Proceso bajo control

limites = GraficoControl.grafico_xbar_r(proceso_estable, 5, "Peso (estable)")
```

### Proceso Fuera de Control

```python
# Proceso con desviación (fuera de control)
np.random.seed(42)
proceso_inestable = np.concatenate([
    np.random.normal(500, 5, 50),   # Normal
    np.random.normal(515, 5, 50),   # Desviado hacia arriba
    np.random.normal(500, 5, 50),   # Vuelve a normal
    np.random.normal(490, 5, 50),   # Desviado hacia abajo
])

limites_inest = GraficoControl.grafico_xbar_r(proceso_inestable, 5, "Peso (inestable)")

# Reglas de detección de Western Electric
print("=== REGLAS DE DETECCIÓN (WESTERN ELECTRIC) ===")
print("1. Un punto fuera de los límites 3σ → FUERA DE CONTROL")
print("2. 2 de 3 puntos consecutivos fuera de 2σ → FUERA DE CONTROL")
print("3. 4 de 5 puntos consecutivos fuera de 1σ → FUERA DE CONTROL")
print("4. 8 puntos consecutivos del mismo lado de la media → FUERA DE CONTROL")
```

---

## 31.3 Gráficos para Atributos (p, np, c, u)

```python
# Gráfico p (proporción de defectos)
np.random.seed(42)

n_lotes = 30
tamano_lote = 200
p_objetivo = 0.03

defectos = np.random.binomial(tamano_lote, p_objetivo, n_lotes)
proporciones = defectos / tamano_lote

# Límites de control para gráfico p
p_promedio = np.mean(proporciones)
lsc_p = p_promedio + 3 * np.sqrt(p_promedio * (1-p_promedio) / tamano_lote)
lic_p = max(0, p_promedio - 3 * np.sqrt(p_promedio * (1-p_promedio) / tamano_lote))

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(proporciones, 'bo-', linewidth=1.5, markersize=6)
ax.axhline(y=p_promedio, color='green', linewidth=2, label=f'p̄ = {p_promedio:.3f}')
ax.axhline(y=lsc_p, color='red', linestyle='--', linewidth=2, label=f'LSC = {lsc_p:.3f}')
ax.axhline(y=lic_p, color='red', linestyle='--', linewidth=2, label=f'LIC = {lic_p:.3f}')
ax.set_title('Gráfico p - Proporción de Defectos', fontweight='bold')
ax.set_xlabel('Lote')
ax.set_ylabel('Proporción de defectos')
ax.legend()
ax.grid(alpha=0.3)
plt.savefig('grafico_p.png', dpi=150, bbox_inches='tight')
plt.show()

print("=== GRÁFICO p (DEFECTOS) ===")
print(f"Proporción promedio: {p_promedio:.2%}")
print(f"LSC: {lsc_p:.2%}")
print(f"LIC: {lic_p:.2%}")
```

---

## 31.4 Capacidad de Proceso (Cp, Cpk)

$$C_p = \frac{ES - EI}{6\sigma}$$
$$C_{pk} = \min\left(\frac{\bar{x} - EI}{3\sigma}, \frac{ES - \bar{x}}{3\sigma}\right)$$

```python
def capacidad_proceso(datos, ei, es, nombre="Proceso"):
    """Calcula índices de capacidad del proceso"""
    media = np.mean(datos)
    sigma = np.std(datos, ddof=1)
    
    Cp = (es - ei) / (6 * sigma)
    Cpk_superior = (es - media) / (3 * sigma)
    Cpk_inferior = (media - ei) / (3 * sigma)
    Cpk = min(Cpk_superior, Cpk_inferior)
    
    # PPM (partes por millón) fuera de especificación
    ppm_superior = (1 - stats.norm.cdf((es - media) / sigma)) * 1e6
    ppm_inferior = stats.norm.cdf((ei - media) / sigma) * 1e6
    ppm_total = ppm_superior + ppm_inferior
    
    print(f"=== CAPACIDAD DEL PROCESO - {nombre} ===")
    print(f"Especificaciones: [{ei}, {es}]")
    print(f"Media: {media:.2f}")
    print(f"Sigma: {sigma:.2f}")
    print(f"\nCp = {Cp:.3f}")
    print(f"Cpk = {Cpk:.3f}")
    print(f"PPM total estimado: {ppm_total:.0f}")
    
    # Interpretación
    if Cpk >= 1.67:
        print("→ Capacidad EXCELENTE")
    elif Cpk >= 1.33:
        print("→ Capacidad BUENA")
    elif Cpk >= 1.0:
        print("→ Capacidad ACEPTABLE")
    elif Cpk >= 0.67:
        print("→ Capacidad BAJA (requiere mejora)")
    else:
        print("→ Capacidad INACEPTABLE (requiere rediseño)")
    
    return {'Cp': Cp, 'Cpk': Cpk, 'ppm': ppm_total}

# Ejemplo: peso de producto (especificación: 500±10)
pesos = np.random.normal(502, 3, 200)
capacidad_proceso(pesos, 490, 510, "Peso de producto")
```

---

## 31.5 Gráfico de Control por Media Móvil (EWMA)

```python
def grafico_ewma(datos, lambda_=0.2, lsc_mult=3):
    """Gráfico EWMA (Exponentially Weighted Moving Average)"""
    n = len(datos)
    ewma = np.zeros(n)
    ewma[0] = np.mean(datos)
    
    for i in range(1, n):
        ewma[i] = lambda_ * datos[i] + (1 - lambda_) * ewma[i-1]
    
    # Límites de control
    media = np.mean(datos)
    sigma = np.std(datos, ddof=1)
    sigma_ewma = sigma * np.sqrt(lambda_ / (2 - lambda_))
    
    lsc = media + lsc_mult * sigma_ewma
    lic = media - lsc_mult * sigma_ewma
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(ewma, 'b-', linewidth=2, label='EWMA')
    ax.axhline(y=media, color='green', linewidth=1.5, label='Media')
    ax.axhline(y=lsc, color='red', linestyle='--', linewidth=2, label='LSC')
    ax.axhline(y=lic, color='red', linestyle='--', linewidth=2, label='LIC')
    ax.fill_between(range(n), lic, lsc, alpha=0.1, color='green')
    ax.set_title(f'Gráfico EWMA (λ={lambda_})', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.savefig('ewma.png', dpi=150, bbox_inches='tight')
    plt.show()

# Detectar cambios pequeños con EWMA
proceso_con_cambio = np.concatenate([
    np.random.normal(100, 3, 50),
    np.random.normal(103, 3, 50)  # Cambio pequeño
])
grafico_ewma(proceso_con_cambio, lambda_=0.15)
```

---

## 31.6 Ejemplo en Ventas: Calidad del Servicio

```python
# Control de calidad para tiempo de respuesta al cliente
np.random.seed(42)

# Tiempo de respuesta en minutos (especificación: < 30 min)
tiempos_respuesta = np.random.exponential(8, 100) + 2

# Gráfico de control
limites_resp = GraficoControl.grafico_xbar_r(tiempos_respuesta, 5, "Tiempo de Respuesta")

# Capacidad
capacidad_proceso(tiempos_respuesta, 0, 30, "Tiempo de Respuesta")
```

---

## 31.7 Ejemplo en Compras: Control de Calidad de Insumos

```python
# Control de calidad para insumos recibidos
np.random.seed(42)

lotes = 40
n_muestra = 50
mediciones = np.random.normal(100, 2, lotes * n_muestra)

# Añadir defectos en algunos lotes
for i in [3, 15, 28]:
    mediciones[i*n_muestra:(i+1)*n_muestra] = np.random.normal(95, 3, n_muestra)

limites_insumos = GraficoControl.grafico_xbar_r(mediciones, n_muestra, "Insumos")

# Evaluación de capacidad
ei, es = 95, 105
capacidad_proceso(mediciones, ei, es, "Insumos")
```

---

## 31.8 Ejemplo en Inventarios: Exactitud de Inventario

```python
# Control de exactitud de inventario (precisión de conteos)
np.random.seed(42)

# Diferencia entre inventario teórico y real (%)
diferencias = np.random.normal(0, 2, 150)  # % de error

# Límites de control
media_diff = np.mean(diferencias)
sigma_diff = np.std(diferencias, ddof=1)

print("=== EXACTITUD DE INVENTARIO ===")
print(f"Error promedio: {media_diff:.2f}%")
print(f"Desviación: {sigma_diff:.2f}%")
print(f"Límites 3σ: [{media_diff - 3*sigma_diff:.2f}%, {media_diff + 3*sigma_diff:.2f}%]")

# Exactitud del inventario (> 95% aceptable)
exactitud = 100 - np.abs(diferencias).mean()
print(f"Exactitud del inventario: {exactitud:.1f}%")
```

---

## 31.9 Resumen

| Gráfico | Tipo | Uso |
|---------|------|-----|
| **X-bar** | Variable | Monitorear media del proceso |
| **R** | Variable | Monitorear variabilidad |
| **S** | Variable | Monitorear desviación estándar |
| **p** | Atributo | Proporción de defectos |
| **np** | Atributo | Número de defectos |
| **c** | Atributo | Conteo de defectos por unidad |
| **u** | Atributo | Defectos por unidad |
| **EWMA** | Variable | Detectar cambios pequeños |

---

## Ejercicios Propuestos

1. **Ventas**: Crea un gráfico de control para el tiempo de respuesta al cliente. ¿Hay puntos fuera de control?
2. **Compras**: Evalúa la capacidad del proceso de un proveedor con especificación [48, 52] y datos con media=50.5, sigma=0.8
3. **Inventarios**: Calcula Cp y Cpk para la exactitud del inventario. ¿El proceso es capaz?

---

[← Anterior](30-arboles-decision.md) | [Índice](index.md) | [Siguiente →](32-muestreo.md)
