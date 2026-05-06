# Capítulo 35: Análisis de Supervivencia

[← Anterior](34-distribuciones-muestrales.md) | [Índice](index.md) | [Inicio ↑](index.md)

---

## 35.1 Introducción

El **análisis de supervivencia** estudia el tiempo hasta que ocurre un evento de interés. Originalmente desarrollado para estudios médicos, tiene múltiples aplicaciones en negocios:

### Aplicaciones en Negocios

| Área | Evento | Tiempo |
|------|--------|--------|
| **Ventas** | Pérdida de cliente (churn) | Tiempo como cliente activo |
| **Compras** | Cambio de proveedor | Duración de relación con proveedor |
| **Inventarios** | Rotura de stock | Tiempo hasta próximo stockout |
| **RRHH** | Renuncia de empleado | Tiempo de permanencia |

### Conceptos Clave

- **Evento**: Lo que queremos predecir (muerte, falla, churn)
- **Censura**: No observamos el evento (cliente sigue activo al final del estudio)
- **Función de Supervivencia**: $S(t) = P(T > t)$
- **Función de Riesgo** (Hazard): $h(t) = \lim_{dt\to0} \frac{P(t \leq T < t+dt | T \geq t)}{dt}$

---

## 35.2 Estimador Kaplan-Meier

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# --- Implementación manual de Kaplan-Meier ---
class KaplanMeier:
    """Estimador de Kaplan-Meier para la función de supervivencia"""
    
    def fit(self, tiempos, eventos):
        """
        tiempos: tiempo hasta evento o censura
        eventos: 1 si ocurrió el evento, 0 si censurado
        """
        df = pd.DataFrame({'tiempo': tiempos, 'evento': eventos})
        df = df.sort_values('tiempo').reset_index(drop=True)
        
        n = len(df)
        n_riesgo = n
        self.tiempos_ = []
        self.supervivencia_ = []
        s = 1.0
        
        for tiempo, grupo in df.groupby('tiempo'):
            n_eventos = grupo['evento'].sum()
            n_riesgo_actual = n_riesgo
            
            if n_eventos > 0:
                s *= (1 - n_eventos / n_riesgo_actual)
                self.tiempos_.append(tiempo)
                self.supervivencia_.append(s)
            
            n_riesgo -= len(grupo)
        
        self.tiempos_ = np.array(self.tiempos_)
        self.supervivencia_ = np.array(self.supervivencia_)
        return self
    
    def plot(self, label='Estimación'):
        plt.step(self.tiempos_, self.supervivencia_, where='post', 
                linewidth=2, label=label)
        plt.xlabel('Tiempo')
        plt.ylabel('Probabilidad de Supervivencia S(t)')
        plt.grid(alpha=0.3)

# --- Simular datos de churn de clientes ---
def simular_churn(n=500, tasa_base=0.02, efecto_contrato=True):
    """Simula datos de churn de clientes"""
    tiempos = []
    eventos = []
    
    for _ in range(n):
        t = 0
        while True:
            riesgo = tasa_base
            if efecto_contrato and t > 12:  # Después de 12 meses
                riesgo *= 0.5  # Menor riesgo con contrato
            if np.random.random() < riesgo:
                tiempos.append(t)
                eventos.append(1)
                break
            t += 1
            if t >= 36:  # Censura a los 36 meses
                tiempos.append(36)
                eventos.append(0)
                break
    
    return np.array(tiempos), np.array(eventos)

# Datos de churn
tiempos_churn, eventos_churn = simular_churn(300, tasa_base=0.03)

# Kaplan-Meier
km_churn = KaplanMeier()
km_churn.fit(tiempos_churn, eventos_churn)

# Con librería lifelines (si está disponible)
try:
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    kmf.fit(tiempos_churn, event_observed=eventos_churn)
    
    print("=== ANÁLISIS DE SUPERVIVENCIA (CHURN) ===")
    print(f"Clientes totales: {len(tiempos_churn)}")
    print(f"Eventos (churn): {eventos_churn.sum()}")
    print(f"Censurados: {(1-eventos_churn).sum()}")
    print(f"Tasa de churn observada: {eventos_churn.mean():.1%}")
    
    # Tiempo medio de supervivencia
    mediana = kmf.median_survival_time_
    print(f"Tiempo mediano de retención: {mediana:.0f} meses")
    
    # Probabilidad de retención a 12 y 24 meses
    for t in [12, 24]:
        s_t = kmf.predict(t)
        print(f"P(retención ≥ {t} meses) = {s_t:.2%}")
    
    # Graficar
    fig, ax = plt.subplots(figsize=(10, 6))
    kmf.plot_survival_function(ax=ax)
    ax.set_title('Curva de Supervivencia - Retención de Clientes', fontweight='bold')
    ax.set_xlabel('Tiempo (meses)')
    ax.set_ylabel('Probabilidad de Retención S(t)')
    ax.grid(alpha=0.3)
    plt.savefig('curva_supervivencia.png', dpi=150, bbox_inches='tight')
    plt.show()
    
except ImportError:
    print("Para análisis avanzado de supervivencia: pip install lifelines")
    print("\n=== KAPLAN-MEIER (Manual) ===")
    fig, ax = plt.subplots(figsize=(10, 6))
    km_churn.plot()
    plt.title('Curva de Supervivencia - Retención (Manual)', fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('curva_supervivencia_manual.png', dpi=150, bbox_inches='tight')
    plt.show()
```

---

## 35.3 Comparación de Grupos (Log-Rank Test)

```python
try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    
    # Comparar dos grupos de clientes
    np.random.seed(42)
    
    # Grupo A: sin contrato
    t_a, e_a = simular_churn(200, tasa_base=0.04, efecto_contrato=False)
    # Grupo B: con contrato
    t_b, e_b = simular_churn(200, tasa_base=0.04, efecto_contrato=True)
    
    # Kaplan-Meier por grupo
    kmf_a = KaplanMeierFitter()
    kmf_b = KaplanMeierFitter()
    
    kmf_a.fit(t_a, event_observed=e_a, label='Sin contrato')
    kmf_b.fit(t_b, event_observed=e_b, label='Con contrato')
    
    # Log-rank test
    resultados = logrank_test(t_a, t_b, event_observed_A=e_a, event_observed_B=e_b)
    
    print("=== COMPARACIÓN DE GRUPOS (LOG-RANK) ===")
    print(f"Grupo A (sin contrato, n={len(t_a)}): churn={e_a.mean():.1%}")
    print(f"Grupo B (con contrato, n={len(t_b)}): churn={e_b.mean():.1%}")
    print(f"\nLog-Rank Test:")
    print(f"  Estadístico: {resultados.test_statistic:.2f}")
    print(f"  p-valor: {resultados.p_value:.4f}")
    print(f"  {'Diferencias significativas' if resultados.p_value < 0.05 else 'No hay diferencias significativas'}")
    
    # Graficar
    fig, ax = plt.subplots(figsize=(10, 6))
    kmf_a.plot_survival_function(ax=ax, linewidth=2)
    kmf_b.plot_survival_function(ax=ax, linewidth=2)
    ax.set_title('Comparación de Supervivencia por Grupo', fontweight='bold')
    ax.set_xlabel('Tiempo (meses)')
    ax.set_ylabel('Probabilidad de Retención')
    ax.grid(alpha=0.3)
    plt.savefig('comparacion_supervivencia.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Tiempos medios
    print(f"\nTiempo mediano retención:")
    print(f"  Sin contrato: {kmf_a.median_survival_time_:.1f} meses")
    print(f"  Con contrato: {kmf_b.median_survival_time_:.1f} meses")
    
except ImportError:
    print("Para análisis avanzado: pip install lifelines")
```

---

## 35.4 Modelo de Riesgos Proporcionales (Cox)

```python
try:
    from lifelines import CoxPHFitter
    
    np.random.seed(42)
    
    # Datos de clientes con múltiples variables
    n = 500
    datos_cox = pd.DataFrame({
        'tiempo': np.zeros(n),
        'evento': np.ones(n),
        'edad': np.random.randint(20, 70, n),
        'ingreso': np.random.normal(50000, 20000, n),
        'contrato_largo': np.random.choice([0, 1], n, p=[0.4, 0.6]),
        'reclamos_previos': np.random.poisson(1, n),
        'satisfaccion': np.random.uniform(1, 5, n),
        'canal_online': np.random.choice([0, 1], n)
    })
    
    # Generar tiempos de supervivencia basados en predictores
    for i in range(n):
        riesgo_base = 0.02
        hazard = (riesgo_base * 
                  np.exp(-0.03 * datos_cox.loc[i, 'edad'] +
                         0.2 * datos_cox.loc[i, 'reclamos_previos'] +
                         -0.3 * datos_cox.loc[i, 'satisfaccion'] +
                         -0.5 * datos_cox.loc[i, 'contrato_largo']))
        
        t = 0
        while True:
            if np.random.random() < hazard:
                datos_cox.loc[i, 'tiempo'] = t
                datos_cox.loc[i, 'evento'] = 1
                break
            t += 1
            if t >= 36:
                datos_cox.loc[i, 'tiempo'] = 36
                datos_cox.loc[i, 'evento'] = 0
                break
    
    # Modelo de Cox
    cph = CoxPHFitter()
    cph.fit(datos_cox, duration_col='tiempo', event_col='evento')
    
    print("=== MODELO DE RIESGOS PROPORCIONALES (COX) ===")
    print(cph.summary)
    
    # Interpretación
    print("\n=== INTERPRETACIÓN (Hazard Ratios) ===")
    hr = np.exp(cph.params_)
    for var, hazard_ratio in hr.items():
        if hazard_ratio > 1:
            print(f"  {var}: HR={hazard_ratio:.3f} → Aumenta el riesgo de churn")
        else:
            print(f"  {var}: HR={hazard_ratio:.3f} → Disminuye el riesgo de churn")
    
    # Graficar importancia
    fig, ax = plt.subplots(figsize=(10, 6))
    cph.plot()
    ax.set_title('Hazard Ratios del Modelo Cox', fontweight='bold')
    plt.savefig('cox_hr.png', dpi=150, bbox_inches='tight')
    plt.show()
    
except ImportError:
    print("Para modelos Cox: pip install lifelines")
```

---

## 35.5 Ejemplo en Ventas: Customer Lifetime Value (CLV)

```python
# Estimación de Customer Lifetime Value con análisis de supervivencia
np.random.seed(42)

try:
    from lifelines import KaplanMeierFitter
    
    # Simular datos de clientes
    n = 1000
    meses_seguimiento = 36
    
    t_clientes, e_clientes = simular_churn(n, tasa_base=0.025)
    
    # Ingreso mensual por cliente
    ingreso_mensual = np.random.gamma(3, 30, n) + 20  # $20-120/mes
    
    # Kaplan-Meier
    kmf_clv = KaplanMeierFitter()
    kmf_clv.fit(t_clientes, event_observed=e_clientes)
    
    # CLV estimado
    tasa_descuento = 0.01  # 1% mensual
    horizonte = 24  # 24 meses
    
    clv_estimado = 0
    for mes in range(1, horizonte + 1):
        s_mes = kmf_clv.predict(mes)
        ingreso_esperado = ingreso_mensual.mean() * s_mes
        clv_estimado += ingreso_esperado / (1 + tasa_descuento)**mes
    
    print("=== CUSTOMER LIFETIME VALUE (CLV) ESTIMADO ===")
    print(f"Clientes analizados: {n}")
    print(f"Tasa de descuento mensual: {tasa_descuento:.1%}")
    print(f"Horizonte: {horizonte} meses")
    print(f"CLV promedio estimado: ${clv_estimado:.2f}")
    print(f"Ingreso mensual promedio: ${ingreso_mensual.mean():.2f}")
    
    # Segmentación por CLV
    clvs = []
    for i in range(10):
        ingreso_i = ingreso_mensual[i]
        clv_i = 0
        for mes in range(1, horizonte + 1):
            s_mes = kmf_clv.predict(mes)
            clv_i += ingreso_i * s_mes / (1 + tasa_descuento)**mes
        clvs.append(clv_i)
    
    print(f"\nTop 5% clientes: CLV > ${np.percentile(clvs, 95):.2f}")
    
except ImportError:
    print("Para CLV completo: pip install lifelines")
```

---

## 35.6 Ejemplo en Compras: Duración de Relación con Proveedores

```python
# Análisis de supervivencia para relaciones con proveedores
np.random.seed(42)

n_prov = 100
meses_relacion = np.random.exponential(24, n_prov) + 3
eventos_prov = np.random.choice([0, 1], n_prov, p=[0.3, 0.7])

try:
    from lifelines import KaplanMeierFitter
    kmf_prov = KaplanMeierFitter()
    kmf_prov.fit(meses_relacion, event_observed=eventos_prov)
    
    print("=== RELACIÓN CON PROVEEDORES ===")
    print(f"Proveedores: {n_prov}")
    print(f"Relación mediana: {kmf_prov.median_survival_time_:.0f} meses")
    print(f"P(relación > 12 meses) = {kmf_prov.predict(12):.1%}")
    print(f"P(relación > 24 meses) = {kmf_prov.predict(24):.1%}")
    print(f"P(relación > 36 meses) = {kmf_prov.predict(36):.1%}")
    
except ImportError:
    print("Para análisis: pip install lifelines")
```

---

## 35.7 Ejemplo en Inventarios: Tiempo entre Stockouts

```python
# Tiempo entre eventos de stockout
np.random.seed(42)
n_periodos = 200
tiempo_entre_stockouts = np.random.exponential(45, n_periodos)  # días entre stockouts

try:
    from lifelines import KaplanMeierFitter
    kmf_stockout = KaplanMeierFitter()
    kmf_stockout.fit(tiempo_entre_stockouts, event_observed=np.ones(n_periodos))
    
    print("=== TIEMPO ENTRE STOCKOUTS ===")
    print(f"Tiempo mediano entre stockouts: {kmf_stockout.median_survival_time_:.0f} días")
    print(f"P(> 30 días sin stockout) = {kmf_stockout.predict(30):.1%}")
    print(f"P(> 60 días sin stockout) = {kmf_stockout.predict(60):.1%}")
    
except ImportError:
    print("Media entre stockouts:", tiempo_entre_stockouts.mean())
```

---

## 35.8 Resumen

| Método | Función | Uso |
|--------|---------|-----|
| **Kaplan-Meier** | $S(t)$ no paramétrica | Estimar retención/churn |
| **Log-Rank Test** | Comparar grupos | A/B testing de retención |
| **Cox PH** | $h(t|X) = h_0(t)e^{\beta X}$ | Identificar factores de riesgo |
| **CLV** | $\sum \frac{ingreso_t \cdot S(t)}{(1+r)^t}$ | Valor del cliente |

---

## Ejercicios Propuestos

1. **Ventas**: Estima la curva de retención de clientes con Kaplan-Meier. ¿Cuál es la probabilidad de retener un cliente 12 meses?
2. **Compras**: Compara la duración de relaciones con proveedores locales vs internacionales usando log-rank test
3. **Inventarios**: Modela el tiempo entre stockouts con Cox PH. ¿Qué factores (lead time, demanda, stock seguridad) afectan el riesgo?

---

[← Anterior](34-distribuciones-muestrales.md) | [Índice](index.md) | [Inicio ↑](index.md)
