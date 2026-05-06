# Capítulo 27: Métodos Bootstrap

[← Anterior](26-simulacion-montecarlo.md) | [Índice](index.md) | [Siguiente →](28-analisis-multivariante.md)

---

## 27.1 Introducción

El **bootstrap** es una técnica de remuestreo que permite estimar la distribución de un estadístico sin asumir una distribución teórica. Inventado por Bradley Efron en 1979, es uno de los métodos más importantes de la estadística moderna.

### Principio

1. De una muestra original de tamaño n, extraer **B muestras bootstrap** con reemplazo
2. Calcular el estadístico de interés en cada muestra
3. La distribución de estos B estadísticos aproxima la distribución muestral real

### Ventajas

- ✅ No asume distribución normal
- ✅ Funciona con estadísticos complejos (mediana, correlación, etc.)
- ✅ Proporciona intervalos de confianza y errores estándar
- ✅ Fácil de implementar

### Fundamento Matemático

Dada una muestra original $X = \{x_1, x_2, ..., x_n\}$ de una distribución desconocida $F$, el bootstrap genera $B$ muestras $X^{*b}$ (con reemplazo) del mismo tamaño $n$. Para cada muestra, calculamos el estadístico de interés $\hat{\theta}^{*b} = s(X^{*b})$.

El **error estándar bootstrap** del estadístico $\hat{\theta}$ es:

$$SE_{boot}(\hat{\theta}) = \sqrt{\frac{1}{B-1} \sum_{b=1}^{B} (\hat{\theta}^{*b} - \bar{\theta}^*)^2}$$

Donde $\bar{\theta}^* = \frac{1}{B} \sum_{b=1}^{B} \hat{\theta}^{*b}$ es la media de las estimaciones bootstrap.

El **sesgo bootstrap** se estima como:

$$Bias_{boot} = \bar{\theta}^* - \hat{\theta}_{original}$$

El **intervalo de confianza por percentiles** al nivel $(1-\alpha)$ se construye tomando los percentiles $\alpha/2$ y $1-\alpha/2$ de la distribución bootstrap:

$$IC_{boot} = [\hat{\theta}^{*}_{(\alpha/2)}, \hat{\theta}^{*}_{(1-\alpha/2)}]$$

Estos fundamentos son la base de todas las aplicaciones bootstrap que veremos a continuación.

---

## 27.2 Bootstrap para el Error Estándar

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

np.random.seed(42)

class Bootstrap:
    """Implementación de bootstrap para estimación"""
    
    def __init__(self, datos, estadistico=np.mean, n_bootstrap=10000):
        self.datos = np.array(datos)
        self.estadistico = estadistico
        self.n_bootstrap = n_bootstrap
        self.n = len(datos)
        self.estimaciones = None
    
    def ejecutar(self):
        """Ejecuta el bootstrap"""
        self.estimaciones = np.zeros(self.n_bootstrap)
        
        for i in range(self.n_bootstrap):
            muestra_boot = np.random.choice(self.datos, self.n, replace=True)
            self.estimaciones[i] = self.estadistico(muestra_boot)
        
        return self
    
    @property
    def se(self):
        """Error estándar bootstrap"""
        return np.std(self.estimaciones, ddof=1)
    
    @property
    def sesgo(self):
        """Sesgo bootstrap"""
        return np.mean(self.estimaciones) - self.estadistico(self.datos)
    
    def ic_percentil(self, nivel=0.95):
        """Intervalo de confianza por percentiles"""
        alpha = 1 - nivel
        li = np.percentile(self.estimaciones, alpha/2 * 100)
        ls = np.percentile(self.estimaciones, (1 - alpha/2) * 100)
        return li, ls
    
    def ic_bca(self, nivel=0.95):
        """Intervalo BCa (Bias-Corrected and Accelerated) - aproximado"""
        return self.ic_percentil(nivel)  # Simplificación
    
    def resumen(self, nombre="Estadístico", nivel=0.95):
        """Resumen completo del bootstrap"""
        estimacion_original = self.estadistico(self.datos)
        ic = self.ic_percentil(nivel)
        
        print(f"=== BOOTSTRAP: {nombre} ===")
        print(f"Estimación original: {estimacion_original:.4f}")
        print(f"Media bootstrap: {np.mean(self.estimaciones):.4f}")
        print(f"Error estándar (SE): {self.se:.4f}")
        print(f"Sesgo: {self.sesgo:.4f}")
        print(f"IC {nivel:.0%} (percentil): [{ic[0]:.4f}, {ic[1]:.4f}]")
        print(f"B = {self.n_bootstrap} remuestras")
        
        return {'original': estimacion_original, 'media_boot': np.mean(self.estimaciones),
                'se': self.se, 'sesgo': self.sesgo, 'ic': ic}
    
    def graficar(self, titulo="Distribución Bootstrap"):
        """Grafica la distribución bootstrap"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histograma
        axes[0].hist(self.estimaciones, bins=50, density=True, 
                     alpha=0.7, color='steelblue', edgecolor='white')
        axes[0].axvline(self.estadistico(self.datos), color='red', 
                        linestyle='--', linewidth=2, label='Original')
        axes[0].axvline(np.mean(self.estimaciones), color='green', 
                        linestyle='--', linewidth=2, label='Media Bootstrap')
        ic = self.ic_percentil()
        axes[0].axvline(ic[0], color='orange', linestyle=':', linewidth=2, label='IC 95%')
        axes[0].axvline(ic[1], color='orange', linestyle=':', linewidth=2)
        axes[0].set_xlabel('Valor del estadístico', fontweight='bold')
        axes[0].set_ylabel('Densidad', fontweight='bold')
        axes[0].set_title(titulo, fontweight='bold')
        axes[0].legend()
        
        # Q-Q plot
        stats.probplot(self.estimaciones, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot (Normalidad)', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('bootstrap_distribucion.png', dpi=150, bbox_inches='tight')
        plt.show()

# --- Ejemplo 1: Media ---
ventas = np.random.gamma(2, 100, 50) + 50  # Datos asimétricos
boot_media = Bootstrap(ventas, np.mean, 10000).ejecutar()
boot_media.resumen("Media de Ventas")

**Interpretación del resultado bootstrap para la media:** El error estándar bootstrap (SE) nos dice cuánto varía la media muestral debido al azar del muestreo. Un SE pequeño indica que la media es estable y precisa; uno grande sugiere que necesitamos más datos. El sesgo bootstrap compara la media de las remuestras con la estimación original: si es cercano a cero, el estimador es insesgado. El intervalo de confianza al 95% nos da el rango plausible para la media poblacional: podemos decir que "con un 95% de confianza, la media real de ventas está entre estos dos valores". A diferencia del IC tradicional basado en la distribución Normal, este IC bootstrap no asume normalidad de los datos, lo que lo hace más robusto cuando trabajamos con datos asimétricos (como ingresos, tiempos de entrega o demanda).

# --- Ejemplo 2: Mediana ---
boot_mediana = Bootstrap(ventas, np.median, 10000).ejecutar()
boot_mediana.resumen("Mediana de Ventas")
boot_mediana.graficar("Distribución Bootstrap - Mediana")
```

---

## 27.3 Bootstrap para Correlación

```python
# Bootstrap para el coeficiente de correlación
x = np.random.normal(100, 15, 40)
y = x * 0.7 + np.random.normal(0, 15, 40)

r_original = stats.pearsonr(x, y)[0]

# Bootstrap de la correlación
n_boot = 5000
r_boot = np.zeros(n_boot)
n = len(x)

for i in range(n_boot):
    indices = np.random.choice(n, n, replace=True)
    r_boot[i] = stats.pearsonr(x[indices], y[indices])[0]

print("=== BOOTSTRAP DE CORRELACIÓN ===")
print(f"r original = {r_original:.4f}")
print(f"r bootstrap medio = {r_boot.mean():.4f}")
print(f"SE bootstrap = {r_boot.std():.4f}")
print(f"IC 95% (percentil): [{np.percentile(r_boot, 2.5):.4f}, {np.percentile(r_boot, 97.5):.4f}]")
```

---

## 27.4 Bootstrap para Diferencia de Medias

```python
# Comparar dos grupos con bootstrap
np.random.seed(42)
grupo_a = np.random.exponential(50, 30) + 20  # Media ≈ 70
grupo_b = np.random.exponential(60, 35) + 20  # Media ≈ 80

n_boot = 10000
diferencias = np.zeros(n_boot)
n_a, n_b = len(grupo_a), len(grupo_b)

for i in range(n_boot):
    muestra_a = np.random.choice(grupo_a, n_a, replace=True)
    muestra_b = np.random.choice(grupo_b, n_b, replace=True)
    diferencias[i] = np.mean(muestra_b) - np.mean(muestra_a)

print("=== BOOTSTRAP: DIFERENCIA DE MEDIAS ===")
print(f"Media A: {np.mean(grupo_a):.1f}")
print(f"Media B: {np.mean(grupo_b):.1f}")
print(f"Diferencia: {np.mean(grupo_b) - np.mean(grupo_a):.1f}")
print(f"IC 95% diferencia: [{np.percentile(diferencias, 2.5):.1f}, {np.percentile(diferencias, 97.5):.1f}]")
print(f"P(diferencia > 0) = {(diferencias > 0).mean():.1%}")
```

---

## 27.5 Bootstrap para Proporciones

```python
# Tasa de conversión con bootstrap
np.random.seed(42)
n = 200
conversiones = 45  # 22.5%

datos_binarios = np.array([1]*45 + [0]*155)

boot_prop = Bootstrap(datos_binarios, np.mean, 10000).ejecutar()
boot_prop.resumen("Tasa de Conversión")

# Comparación con fórmula asintótica
p_hat = 45/200
se_asint = np.sqrt(p_hat * (1-p_hat) / n)
ic_asint = (p_hat - 1.96*se_asint, p_hat + 1.96*se_asint)
print(f"\nFórmula asintótica: IC 95% = [{ic_asint[0]:.4f}, {ic_asint[1]:.4f}]")
print(f"Bootstrap:          IC 95% = [{boot_prop.ic_percentil()[0]:.4f}, {boot_prop.ic_percentil()[1]:.4f}]")
```

---

## 27.6 Bootstrap para Razones (Ratio)

Útil para ratios como eficiencia, margen, rotación:

```python
# Razón de margen (utilidad / ingreso)
np.random.seed(42)
n = 30
ingresos = np.random.normal(50000, 10000, n)
costos = ingresos * 0.6 + np.random.normal(0, 3000, n)
utilidades = ingresos - costos
margen = utilidades / ingresos

n_boot = 5000
margen_original = np.mean(margen)
margenes_boot = np.zeros(n_boot)

for i in range(n_boot):
    idx = np.random.choice(n, n, replace=True)
    margenes_boot[i] = np.mean(utilidades[idx] / ingresos[idx])

print("=== BOOTSTRAP PARA MARGEN DE UTILIDAD ===")
print(f"Margen original: {margen_original:.2%}")
print(f"IC 95%: [{np.percentile(margenes_boot, 2.5):.2%}, {np.percentile(margenes_boot, 97.5):.2%}]")
```

---

## 27.7 Bootstrap para Series Temporales (Block Bootstrap)

Para datos dependientes (series temporales), usamos **block bootstrap**:

```python
def block_bootstrap(serie, block_size=7, n_boot=1000):
    """Block bootstrap para series temporales"""
    n = len(serie)
    n_blocks = int(np.ceil(n / block_size))
    estimaciones = []
    
    for _ in range(n_boot):
        # Seleccionar bloques aleatorios
        indices_bloques = np.random.choice(n - block_size + 1, n_blocks)
        muestra_boot = []
        for idx in indices_bloques:
            muestra_boot.extend(serie[idx:idx + block_size])
        muestra_boot = np.array(muestra_boot[:n])
        estimaciones.append(np.mean(muestra_boot))
    
    return np.array(estimaciones)

# Serie temporal de demanda
np.random.seed(42)
n = 200
demanda = 50 + 0.1 * np.arange(n) + 5 * np.sin(2 * np.pi * np.arange(n) / 7) + \
          np.random.normal(0, 5, n)

est_boot = block_bootstrap(demanda, block_size=7, n_boot=2000)

print("=== BLOCK BOOTSTRAP (SERIE TEMPORAL) ===")
print(f"Demanda media: {np.mean(demanda):.1f}")
print(f"SE block bootstrap: {est_boot.std():.1f}")
print(f"IC 95%: [{np.percentile(est_boot, 2.5):.1f}, {np.percentile(est_boot, 97.5):.1f}]")
```

---

## 27.8 Bootstrap para Regresión

```python
# Bootstrap de coeficientes de regresión
from sklearn.linear_model import LinearRegression

np.random.seed(42)
n = 100
X = np.random.uniform(0, 10, n)
y = 5 + 2 * X + np.random.normal(0, 2, n)

n_boot = 5000
coefs_boot = np.zeros((n_boot, 2))

for i in range(n_boot):
    idx = np.random.choice(n, n, replace=True)
    X_boot = X[idx].reshape(-1, 1)
    y_boot = y[idx]
    modelo = LinearRegression().fit(X_boot, y_boot)
    coefs_boot[i] = [modelo.intercept_, modelo.coef_[0]]

print("=== BOOTSTRAP DE REGRESIÓN ===")
print(f"{'Coeficiente':15s} {'Original':>10s} {'Media Boot':>10s} {'SE':>10s} {'IC 95%':>30s}")
print("-" * 75)
for i, nombre in enumerate(['Intercepto', 'Pendiente']):
    orig = [5, 2][i]
    media_boot = coefs_boot[:, i].mean()
    se = coefs_boot[:, i].std()
    ic = np.percentile(coefs_boot[:, i], [2.5, 97.5])
    print(f"{nombre:15s} {orig:>10.2f} {media_boot:>10.4f} {se:>10.4f} [{ic[0]:>8.4f}, {ic[1]:>8.4f}]")
```

---

## 27.9 Ejemplo en Ventas: IC del Ticket Promedio

```python
# Intervalo de confianza bootstrap para el ticket promedio
np.random.seed(42)
tickets = np.random.gamma(3, 50, 100) + 20  # Datos asimétricos

boot_ticket = Bootstrap(tickets, np.mean, 10000).ejecutar()
boot_ticket.graficar("IC Bootstrap - Ticket Promedio")

# Comparar con IC tradicional
media = np.mean(tickets)
se_t = np.std(tickets, ddof=1) / np.sqrt(len(tickets))
ic_trad = (media - 1.96*se_t, media + 1.96*se_t)
ic_boot = boot_ticket.ic_percentil()

print(f"\nIC tradicional (Normal): [${ic_trad[0]:.2f}, ${ic_trad[1]:.2f}]")
print(f"IC Bootstrap (percentil): [${ic_boot[0]:.2f}, ${ic_boot[1]:.2f}]")
```

---

## 27.10 Ejemplo en Compras: Variabilidad de Precios

```python
# Evaluación de la variabilidad de precios de proveedores
np.random.seed(42)

precios_prov_a = np.random.normal(100, 5, 20)  # Estable
precios_prov_b = np.random.normal(105, 15, 20)  # Volátil

# Bootstrap para el CV
def cv(datos):
    return np.std(datos, ddof=1) / np.mean(datos)

boot_cv_a = Bootstrap(precios_prov_a, cv, 5000).ejecutar()
boot_cv_b = Bootstrap(precios_prov_b, cv, 5000).ejecutar()

print("=== BOOTSTRAP: COEFICIENTE DE VARIACIÓN ===")
boot_cv_a.resumen("CV Proveedor A")
print()
boot_cv_b.resumen("CV Proveedor B")
```

---

## 27.11 Ejemplo en Inventarios: Rotación de Inventario

```python
# Bootstrap para la rotación de inventario
np.random.seed(42)
rotacion = np.random.exponential(10, 50) + 2  # Días entre rotaciones

boot_rot = Bootstrap(rotacion, np.mean, 10000).ejecutar()
boot_rot.resumen("Rotación de Inventario (días)")

# Interpretación
ic_rot = boot_rot.ic_percentil()
print(f"\nCon 95% de confianza, la rotación promedio")
print(f"está entre {ic_rot[0]:.1f} y {ic_rot[1]:.1f} días")
```

---

## 27.12 Resumen

| Aplicación | Estadístico | Bootstrap útil? |
|-----------|-------------|-----------------|
| Media | Media | ✓ Útil si datos no normales |
| Mediana | Mediana | ✓ **Muy** útil (no hay fórmula simple) |
| Correlación | r de Pearson | ✓ Útil para IC |
| Proporción | p̂ | ✓ Similar a fórmula asintótica |
| Diferencia de medias | d | ✓ Alternativa a t-test |
| Ratio | a/b | ✓ Ideal (ratios son complejos) |
| Regresión | β | ✓ Evaluar estabilidad |

---

## Ejercicios Propuestos

1. **Ventas**: Usa bootstrap para obtener IC 95% de la mediana de tickets de venta (datos asimétricos)
2. **Compras**: Compara el precio promedio de dos proveedores con bootstrap de la diferencia
3. **Inventarios**: Calcula el IC bootstrap para la rotación de inventario (días entre salidas)

---

[← Anterior](26-simulacion-montecarlo.md) | [Índice](index.md) | [Siguiente →](28-analisis-multivariante.md)
