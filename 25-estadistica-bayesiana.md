# Capítulo 25: Estadística Bayesiana

[← Anterior](24-pronosticos.md) | [Índice](index.md) | [Siguiente →](26-simulacion-montecarlo.md)

---

## 25.1 Introducción

La **estadística bayesiana** es un enfoque que actualiza la probabilidad de una hipótesis a medida que se obtiene nueva evidencia. A diferencia de la estadística frecuentista, incorpora **conocimiento previo** (prior) y lo actualiza con datos observados para obtener una **distribución posterior**.

### Paradigma Bayesiano

$$P(\theta | datos) = \frac{P(datos | \theta) \cdot P(\theta)}{P(datos)}$$

| Término | Nombre | Significado |
|---------|--------|-------------|
| $P(\theta)$ | **Prior** | Creencia inicial sobre θ |
| $P(datos|\theta)$ | **Verosimilitud** | Probabilidad de los datos dado θ |
| $P(\theta|datos)$ | **Posterior** | Creencia actualizada sobre θ |
| $P(datos)$ | **Evidencia** | Probabilidad marginal de los datos |

### Frecuentista vs Bayesiano

| Aspecto | Frecuentista | Bayesiano |
|---------|-------------|-----------|
| Parámetro θ | Fijo, desconocido | Variable aleatoria |
| Probabilidad | Frecuencia a largo plazo | Grado de creencia |
| Resultado | IC: "95% de los intervalos contienen θ" | IC: "95% de probabilidad de que θ esté en el intervalo" |
| Información previa | No se usa | Se incorpora explícitamente |

---

## 25.2 Inferencia Bayesiana con la Distribución Beta

Para **proporciones** (tasas de conversión, defectos, etc.), usamos:

- **Prior**: Beta(α, β)
- **Verosimilitud**: Binomial(n, p)
- **Posterior**: Beta(α + éxitos, β + fracasos)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

np.random.seed(42)

class InferenciaBayesianaBeta:
    """Inferencia bayesiana para proporciones usando Beta-Binomial"""
    
    def __init__(self, alpha_prior=1, beta_prior=1):
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.prior = stats.beta(alpha_prior, beta_prior)
    
    def actualizar(self, exitos, fracasos):
        """Actualiza la distribución posterior"""
        self.alpha_post = self.alpha_prior + exitos
        self.beta_post = self.beta_prior + fracasos
        self.posterior = stats.beta(self.alpha_post, self.beta_post)
        return self
    
    def resumen(self):
        """Resumen de la distribución posterior"""
        media = self.posterior.mean()
        mediana = self.posterior.median()
        std = self.posterior.std()
        ic_95 = (self.posterior.ppf(0.025), self.posterior.ppf(0.975))
        
        print("=== INFERENCIA BAYESIANA - PROPORCIÓN ===")
        print(f"Prior: Beta({self.alpha_prior}, {self.beta_prior})")
        print(f"Posterior: Beta({self.alpha_post}, {self.beta_post})")
        print(f"\nMedia posterior: {media:.4f} ({media*100:.2f}%)")
        print(f"Mediana posterior: {mediana:.4f}")
        print(f"Desviación estándar: {std:.4f}")
        print(f"IC 95% (Intervalo de Credibilidad): [{ic_95[0]:.4f}, {ic_95[1]:.4f}]")
        print(f"P(proporción > 0.3) = {1 - self.posterior.cdf(0.3):.4f}")
        
        return {'media': media, 'mediana': mediana, 'std': std, 'ic_95': ic_95}
    
    def graficar(self, titulo="Distribuciones Beta"):
        """Grafica prior y posterior"""
        x = np.linspace(0, 1, 500)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x, self.prior.pdf(x), 'b--', linewidth=2, label='Prior')
        ax.plot(x, self.posterior.pdf(x), 'r-', linewidth=2, label='Posterior')
        ax.axvline(self.prior.mean(), color='blue', linestyle=':', alpha=0.5)
        ax.axvline(self.posterior.mean(), color='red', linestyle=':', alpha=0.5)
        ax.fill_between(x, 0, self.posterior.pdf(x), 
                        where=(x >= self.posterior.ppf(0.025)) & 
                              (x <= self.posterior.ppf(0.975)),
                        color='red', alpha=0.2, label='IC 95%')
        ax.set_xlabel('Proporción', fontweight='bold')
        ax.set_ylabel('Densidad', fontweight='bold')
        ax.set_title(titulo, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.savefig('bayes_beta.png', dpi=150, bbox_inches='tight')
        plt.show()

# Ejemplo: Tasa de conversión
# Prior informativo (40 éxitos en 200 intentos previos → Beta(40, 160))
bayes = InferenciaBayesianaBeta(alpha_prior=40, beta_prior=160)
bayes.actualizar(exitos=25, fracasos=75)  # Nueva campaña: 25 ventas en 100 leads
bayes.resumen()
bayes.graficar("Actualización Bayesiana - Tasa de Conversión")

# Comparación con enfoque frecuentista
p_hat = 25/100
se = np.sqrt(p_hat * (1-p_hat) / 100)
ic_frec = (p_hat - 1.96*se, p_hat + 1.96*se)
print(f"\nFrecuentista: p̂ = {p_hat:.2%}, IC 95% = [{ic_frec[0]:.2%}, {ic_frec[1]:.2%}]")
```

---

## 25.3 Actualización Secuencial

La estadística bayesiana permite actualizar creencias a medida que llegan nuevos datos:

```python
# Actualización paso a paso
bayes_sec = InferenciaBayesianaBeta(alpha_prior=1, beta_prior=1)

# Datos de 4 semanas de campaña
semanas = [(8, 32), (12, 28), (10, 30), (15, 25)]  # (exitos, fracasos)

print("=== ACTUALIZACIÓN SECUENCIAL ===")
for i, (exitos, fracasos) in enumerate(semanas, 1):
    bayes_sec.actualizar(exitos, fracasos)
    res = bayes_sec.resumen()
    print(f"\n--- Semana {i}: {exitos}/{exitos+fracasos} = {exitos/(exitos+fracasos):.1%} ---")
    print(f"Posterior: Beta({bayes_sec.alpha_post}, {bayes_sec.beta_post})")
    print(f"Media: {res['media']:.2%}, IC 95%: [{res['ic_95'][0]:.2%}, {res['ic_95'][1]:.2%}]")
```

---

## 25.4 Comparación de Dos Grupos (A/B Testing)

```python
def comparar_grupos_bayesiano(exitos_a, total_a, exitos_b, total_b,
                                alpha_prior=1, beta_prior=1):
    """Compara dos grupos usando inferencia bayesiana"""
    # Posteriors
    post_a = stats.beta(alpha_prior + exitos_a, beta_prior + total_a - exitos_a)
    post_b = stats.beta(alpha_prior + exitos_b, beta_prior + total_b - exitos_b)
    
    # Simular diferencias
    n_sim = 100000
    muestras_a = post_a.rvs(n_sim)
    muestras_b = post_b.rvs(n_sim)
    diferencias = muestras_b - muestras_a
    
    # Probabilidad de que B sea mejor que A
    p_b_mejor = np.mean(diferencias > 0)
    
    print("=== A/B TESTING BAYESIANO ===")
    print(f"Grupo A: {exitos_a}/{total_a} = {exitos_a/total_a:.1%}")
    print(f"Grupo B: {exitos_b}/{total_b} = {exitos_b/total_b:.1%}")
    print(f"P(Conversión A): media={post_a.mean():.2%}, IC95=[{post_a.ppf(0.025):.2%}, {post_a.ppf(0.975):.2%}]")
    print(f"P(Conversión B): media={post_b.mean():.2%}, IC95=[{post_b.ppf(0.025):.2%}, {post_b.ppf(0.975):.2%}]")
    print(f"Diferencia esperada: {diferencias.mean():.2%}")
    print(f"P(B mejor que A) = {p_b_mejor:.2%}")
    
    if p_b_mejor > 0.95:
        print("→ Alta probabilidad de que B sea superior")
    elif p_b_mejor < 0.05:
        print("→ Alta probabilidad de que A sea superior")
    else:
        print("→ No hay suficiente evidencia para concluir")
    
    return {'post_a': post_a, 'post_b': post_b, 'p_b_mejor': p_b_mejor}

# Ejemplo: Dos versiones de landing page
comparar_grupos_bayesiano(exitos_a=45, total_a=500, exitos_b=62, total_b=500)
```

---

## 25.5 Inferencia Bayesiana para la Media (Normal)

Cuando la media es desconocida y usamos un prior Normal:

```python
def inferencia_media_bayesiana(datos, mu_prior=0, sigma_prior=100):
    """Inferencia bayesiana para la media de una Normal"""
    n = len(datos)
    media_muestral = np.mean(datos)
    sigma_datos = np.std(datos, ddof=1)
    
    # Posterior: Normal
    precision_prior = 1 / sigma_prior**2
    precision_datos = n / sigma_datos**2
    
    mu_post = (mu_prior * precision_prior + media_muestral * precision_datos) / \
              (precision_prior + precision_datos)
    sigma_post = np.sqrt(1 / (precision_prior + precision_datos))
    
    posterior = stats.norm(mu_post, sigma_post)
    
    print("=== INFERENCIA BAYESIANA - MEDIA ===")
    print(f"Prior: Normal(μ={mu_prior}, σ={sigma_prior})")
    print(f"Datos: n={n}, media={media_muestral:.2f}")
    print(f"Posterior: Normal(μ={mu_post:.2f}, σ={sigma_post:.2f})")
    print(f"IC 95%: [{posterior.ppf(0.025):.2f}, {posterior.ppf(0.975):.2f}]")
    
    return posterior

# Datos de ventas
ventas = stats.norm.rvs(loc=500, scale=50, size=30)
post_media = inferencia_media_bayesiana(ventas, mu_prior=450, sigma_prior=100)
```

---

## 25.6 Introducción a MCMC con PyMC

Para modelos más complejos, usamos **MCMC** (Markov Chain Monte Carlo):

```python
try:
    import pymc as pm
    print("PyMC disponible para modelos bayesianos avanzados")
    
    # Modelo simple con PyMC
    np.random.seed(42)
    datos_obs = stats.norm.rvs(loc=100, scale=15, size=50)
    
    with pm.Model() as modelo:
        # Prior
        mu = pm.Normal('mu', mu=0, sigma=100)
        sigma = pm.HalfNormal('sigma', sigma=50)
        
        # Verosimilitud
        observaciones = pm.Normal('obs', mu=mu, sigma=sigma, observed=datos_obs)
        
        # MCMC
        traza = pm.sample(1000, tune=500, chains=2, progressbar=False)
    
    print("\n=== RESULTADOS MCMC - PyMC ===")
    print(pm.summary(traza).round(3))
    
    # Media posterior
    mu_post = traza['mu'].mean()
    ic_post = np.percentile(traza['mu'], [2.5, 97.5])
    print(f"\nMedia posterior: {mu_post:.2f}")
    print(f"IC 95%: [{ic_post[0]:.2f}, {ic_post[1]:.2f}]")
    
except ImportError:
    print("PyMC no está instalado. Ejecuta: pip install pymc")
    print("Los conceptos bayesianos se explican con scipy.stats")
```

---

## 25.7 Ejemplo en Ventas: Tasa de Conversión por Segmento

```python
# Estimación bayesiana de conversión por segmento de cliente
np.random.seed(42)

segmentos = {
    'Premium': (80, 20),    # 80% conversión
    'Estándar': (40, 60),   # 40% conversión
    'Económico': (15, 85),  # 15% conversión
}

print("=== TASA DE CONVERSIÓN POR SEGMENTO (BAYESIANO) ===")
for segmento, (exitos, total) in segmentos.items():
    fracasos = total - exitos
    posterior = stats.beta(1 + exitos, 1 + fracasos)
    
    print(f"\n{segmento}: {exitos}/{total} observados")
    print(f"  Media posterior: {posterior.mean():.1%}")
    print(f"  IC 95%: [{posterior.ppf(0.025):.1%}, {posterior.ppf(0.975):.1%}]")

# Probabilidad de que Premium > Estándar
post_premium = stats.beta(1 + 80, 1 + 20)
post_estandar = stats.beta(1 + 40, 1 + 60)
p_superior = np.mean(post_premium.rvs(50000) > post_estandar.rvs(50000))
print(f"\nP(Premium > Estándar) = {p_superior:.1%}")
```

---

## 25.8 Ejemplo en Compras: Probabilidad de Defectos

```python
# Estimación bayesiana de tasa de defectos
np.random.seed(42)

# Prior: históricamente 2% de defectos (2 en 100)
alpha_prior = 2 + 1
beta_prior = 98 + 1

# Nuevo lote: 3 defectos en 200 unidades
posterior_def = stats.beta(alpha_prior + 3, beta_prior + 197)

print("=== ESTIMACIÓN DE DEFECTOS (BAYESIANO) ===")
print(f"Prior: 2% (2 defectos en 100)")
print(f"Nuevo lote: 3 defectos en 200")
print(f"Tasa frecuentista: {3/200:.2%}")
print(f"Tasa bayesiana: {posterior_def.mean():.2%}")
print(f"IC 95%: [{posterior_def.ppf(0.025):.2%}, {posterior_def.ppf(0.975):.2%}]")
print(f"P(tasa real < 3%) = {posterior_def.cdf(0.03):.2%}")
```

---

## 25.9 Ejemplo en Inventarios: Demanda con Prior

```python
# Estimación bayesiana de demanda con información previa
np.random.seed(42)

# Prior: demanda ~ Normal(100, 20)
mu_prior = 100
sigma_prior = 20

# Observaciones recientes
demanda_obs = stats.norm.rvs(loc=110, scale=25, size=15)

# Actualización bayesiana
n = len(demanda_obs)
media_obs = np.mean(demanda_obs)
sigma_obs = np.std(demanda_obs, ddof=1)

precision_prior = 1 / sigma_prior**2
precision_obs = n / sigma_obs**2
mu_post = (mu_prior * precision_prior + media_obs * precision_obs) / \
          (precision_prior + precision_obs)
sigma_post = np.sqrt(1 / (precision_prior + precision_obs))

print("=== ESTIMACIÓN DE DEMANDA (BAYESIANO) ===")
print(f"Prior: Normal(μ={mu_prior}, σ={sigma_prior})")
print(f"Observado: media={media_obs:.1f}, n={n}")
print(f"Posterior: Normal(μ={mu_post:.1f}, σ={sigma_post:.1f})")

# Stock de seguridad bayesiano
z = 1.645
ss_bayes = z * sigma_post * np.sqrt(7)  # lead time 7 días
print(f"\nStock de seguridad (bayesiano): {ss_bayes:.0f} unidades")
```

---

## 25.10 Resumen

| Concepto | Fórmula | Uso |
|----------|---------|-----|
| **Teorema de Bayes** | $P(\theta|D) \propto P(D|\theta) P(\theta)$ | Actualizar creencias |
| **Prior** | $P(\theta)$ | Conocimiento inicial |
| **Verosimilitud** | $P(D|\theta)$ | Evidencia de los datos |
| **Posterior** | $P(\theta|D)$ | Creencia actualizada |
| **Beta-Binomial** | Posterior = Beta(α+éxitos, β+fracasos) | Proporciones |
| **Normal-Normal** | Posterior es Normal | Medias |
| **MCMC** | Muestreo de la posterior | Modelos complejos |

---

## Ejercicios Propuestos

1. **Ventas**: Estima la tasa de conversión con prior informativo (30 ventas en 200 leads previos) y nuevos datos (20 ventas en 80 leads)
2. **Compras**: Compara dos proveedores con A/B testing bayesiano. ¿Cuál tiene menor tasa de defectos?
3. **Inventarios**: Actualiza la estimación de demanda con prior Normal(200, 30) y 20 días de observaciones con media 220

---

[← Anterior](24-pronosticos.md) | [Índice](index.md) | [Siguiente →](26-simulacion-montecarlo.md)
