# Manual Completo de Probabilidad y Estadística Aplicada

## Sector: Ventas, Compras e Inventarios

---

Bienvenido al manual más extenso y detallado de probabilidad y estadística aplicada al mundo de los negocios. Este manual cubre **35 temas fundamentales** con explicaciones teóricas profundas, implementaciones prácticas en Python (NumPy, SciPy, Pandas, StatsModels, Scikit-learn, PyMC, Matplotlib, Seaborn, Lifelines) y ejemplos reales en los sectores de **ventas**, **compras** e **inventarios**.

Cada capítulo incluye código reproducible, fórmulas matemáticas, interpretación de resultados y ejercicios prácticos.

---

## Estructura del Manual

### 📐 Parte I: Fundamentos Estadísticos

| # | Capítulo | Descripción |
|---|----------|-------------|
| 01 | [Introducción a la Estadística](01-intro-estadistica.md) | Conceptos fundamentales, población y muestra |
| 02 | [Introducción a la Probabilidad](02-intro-probabilidad.md) | Axiomas, probabilidad condicional, Bayes |
| 03 | [Estadística Descriptiva](03-estadistica-descriptiva.md) | Tablas, frecuencias, asimetría, outliers |
| 04 | [Medidas de Tendencia Central](04-medidas-tendencia-central.md) | Media, mediana, moda, ponderada, geométrica |
| 05 | [Medidas de Dispersión](05-medidas-dispersion.md) | Varianza, desviación estándar, IQR, CV |
| 06 | [Visualización de Datos](06-visualizacion-datos.md) | Matplotlib, Seaborn, dashboards ejecutivos |

### 🎲 Parte II: Distribuciones de Probabilidad

| # | Capítulo | Descripción |
|---|----------|-------------|
| 07 | [Distribuciones Discretas](07-distribuciones-discretas.md) | Binomial, Poisson, Geométrica, Hipergeométrica |
| 08 | [Distribuciones Continuas](08-distribuciones-continuas.md) | Normal, Exponencial, Gamma, Beta, t-Student |
| 09 | [Teorema del Límite Central](09-teorema-limite-central.md) | TLC, error estándar, simulaciones |
| 10 | [Ley de los Grandes Números](10-ley-grandes-numeros.md) | Ley débil y fuerte, convergencia, falacia del jugador |

### 🔬 Parte III: Inferencia Estadística

| # | Capítulo | Descripción |
|---|----------|-------------|
| 11 | [Estimación Puntual](11-estimacion-puntual.md) | MLE, método de momentos, MSE, sesgo |
| 12 | [Intervalos de Confianza](12-intervalos-confianza.md) | IC para media, proporción, varianza, bootstrap |
| 13 | [Pruebas de Hipótesis](13-pruebas-hipotesis.md) | Errores tipo I/II, p-valor, potencia, tamaño del efecto |
| 14 | [Pruebas Paramétricas](14-pruebas-parametricas.md) | t-student, t pareada, pruebas Z, ANOVA |
| 15 | [Pruebas No Paramétricas](15-pruebas-no-parametricas.md) | Mann-Whitney, Wilcoxon, Kruskal-Wallis, Friedman |
| 16 | [Prueba Chi-Cuadrado](16-chi-cuadrado.md) | Bondad de ajuste, independencia, homogeneidad, Fisher |

### 📈 Parte IV: Relaciones y Regresión

| # | Capítulo | Descripción |
|---|----------|-------------|
| 17 | [Correlación](17-correlacion.md) | Pearson, Spearman, Kendall, correlación parcial |
| 18 | [Regresión Lineal Simple](18-regresion-lineal.md) | MCO, interpretación, diagnóstico de residuos |
| 19 | [Regresión Lineal Múltiple](19-regresion-multiple.md) | Selección de variables, VIF, interacciones, regularización |
| 20 | [Regresión Logística](20-regresion-logistica.md) | Clasificación binaria, odds ratio, AUC-ROC |

### 📊 Parte V: Series Temporales

| # | Capítulo | Descripción |
|---|----------|-------------|
| 21 | [Introducción a Series Temporales](21-series-temporales-intro.md) | Componentes, descomposición, estacionariedad, ACF/PACF |
| 22 | [Modelos ARIMA](22-modelos-arima.md) | ARIMA, SARIMA, identificación, pronóstico |
| 23 | [Suavización Exponencial](23-suavizacion-exponencial.md) | SES, Holt, Holt-Winters, ETS |
| 24 | [Pronósticos de Ventas](24-pronosticos.md) | Métodos comparados, backtesting, presupuestos |

### 🧠 Parte VI: Técnicas Avanzadas

| # | Capítulo | Descripción |
|---|----------|-------------|
| 25 | [Estadística Bayesiana](25-estadistica-bayesiana.md) | Teorema de Bayes, Beta-Binomial, MCMC, PyMC |
| 26 | [Simulación Montecarlo](26-simulacion-montecarlo.md) | Simulación de inventario, VaR, optimización |
| 27 | [Métodos Bootstrap](27-bootstrap.md) | Remuestreo, IC bootstrap, block bootstrap |
| 28 | [Análisis Multivariante](28-analisis-multivariante.md) | PCA, análisis factorial, reducción de dimensionalidad |
| 29 | [Clustering](29-clustering.md) | K-means, jerárquico, DBSCAN, segmentación |
| 30 | [Árboles de Decisión](30-arboles-decision.md) | Clasificación, regresión, Random Forest |

### 🏭 Parte VII: Aplicaciones Industriales

| # | Capítulo | Descripción |
|---|----------|-------------|
| 31 | [Control de Calidad (SPC)](31-control-calidad.md) | Gráficos X-bar, R, p, Cp, Cpk, EWMA |
| 32 | [Muestreo](32-muestreo.md) | MAS, estratificado, conglomerados, tamaño muestral |
| 33 | [Análisis de Varianza (ANOVA)](33-analisis-varianza.md) | Unifactorial, bifactorial, medidas repetidas, MANOVA |
| 34 | [Distribuciones Muestrales](34-distribuciones-muestrales.md) | Media, proporción, varianza, t-Student, F-Fisher |
| 35 | [Análisis de Supervivencia](35-analisis-supervivencia.md) | Kaplan-Meier, Cox PH, Log-Rank, CLV |

---

## Librerías de Python Utilizadas

| Librería | Propósito | Documentación |
|----------|-----------|---------------|
| **NumPy** | Cómputo numérico, arreglos | [numpy.org](https://numpy.org) |
| **SciPy** | Distribuciones, tests estadísticos | [scipy.org](https://scipy.org) |
| **Pandas** | Manipulación y análisis de datos | [pandas.pydata.org](https://pandas.pydata.org) |
| **Matplotlib** | Visualización de datos | [matplotlib.org](https://matplotlib.org) |
| **Seaborn** | Visualización estadística | [seaborn.pydata.org](https://seaborn.pydata.org) |
| **StatsModels** | Modelos estadísticos clásicos | [statsmodels.org](https://statsmodels.org) |
| **Scikit-learn** | Machine learning | [scikit-learn.org](https://scikit-learn.org) |
| **PyMC** | Inferencia bayesiana | [pymc.io](https://pymc.io) |
| **Lifelines** | Análisis de supervivencia | [lifelines.readthedocs.io](https://lifelines.readthedocs.io) |

---

## Convenciones del Manual

Cada capítulo sigue la misma estructura:

1. **Fundamento teórico** con fórmulas matemáticas en LaTeX
2. **Implementación en Python** con código listo para ejecutar
3. **Ejemplo en Ventas** aplicación directa al departamento comercial
4. **Ejemplo en Compras** aplicación al aprovisionamiento
5. **Ejemplo en Inventarios** aplicación a la gestión de stocks
6. **Interpretación** de resultados y conclusiones de negocio

---

## Requisitos Técnicos

```bash
pip install numpy scipy pandas matplotlib seaborn statsmodels scikit-learn pymc lifelines
```

---

## Navegación Rápida

| Anterior | Índice | Siguiente |
|----------|--------|-----------|
| — | [Índice](index.md) | [01 →](01-intro-estadistica.md) |

---

*Manual generado con propósitos educativos y de aplicación profesional.*
