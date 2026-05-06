# Capítulo 2: Introducción a la Probabilidad

[← Anterior](01-intro-estadistica.md) | [Índice](index.md) | [Siguiente →](03-estadistica-descriptiva.md)

---

## 2.1 Conceptos Fundamentales

La **probabilidad** es la medida numérica de la posibilidad de que ocurra un evento. Su valor está entre 0 (imposible) y 1 (certeza absoluta).

### Experimentos y Eventos

- **Experimento aleatorio**: Proceso cuyo resultado no se puede predecir con certeza
  - Ej: *Lanzar un dado, la demanda del próximo mes*
- **Espacio muestral (Ω)**: Conjunto de todos los resultados posibles
  - Ej: *Ω = {0, 1, 2, ...} unidades vendidas mañana*
- **Evento (E)**: Subconjunto del espacio muestral
  - Ej: *E = "vender más de 100 unidades"*

```
Experimento: "Observar la demanda diaria de un producto"
Espacio muestral: Ω = {0, 1, 2, 3, ...}
Evento A: "Demanda ≥ 50 unidades" = {50, 51, 52, ...}
Evento B: "Demanda entre 20 y 30" = {20, 21, ..., 30}
```

### Axiomas de Probabilidad (Kolmogorov)

1. **No negatividad**: $P(E) \geq 0$ para todo evento E
2. **Certeza**: $P(\Omega) = 1$
3. **Aditividad**: Si $E_1, E_2, ...$ son mutuamente excluyentes, entonces $P(\bigcup_i E_i) = \sum_i P(E_i)$

---

## 2.2 Tipos de Probabilidad

| Enfoque | Definición | Ejemplo |
|---------|-----------|---------|
| **Clásica** | $P(E) = \frac{\text{casos favorables}}{\text{casos posibles}}$ | Probabilidad de que un producto sea defectuoso si 5 de 100 lo son: $P = 5/100 = 0.05$ |
| **Frecuencial** | $P(E) = \lim_{n \to \infty} \frac{\text{ocurrencias de }E}{n}$ | Probabilidad de stockout basada en 1000 días de observación |
| **Subjetiva** | Grado de creencia personal | "Tengo un 80% de confianza en que las ventas superarán el millón" |

---

## 2.3 Reglas de Probabilidad

### Regla de la Adición

Para eventos **no excluyentes**:
$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

Si son **mutuamente excluyentes** ($A \cap B = \emptyset$):
$$P(A \cup B) = P(A) + P(B)$$

### Regla de la Multiplicación

Para eventos **dependientes**:
$$P(A \cap B) = P(A) \cdot P(B|A)$$

Para eventos **independientes**:
$$P(A \cap B) = P(A) \cdot P(B)$$

### Probabilidad Condicional

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

*"Probabilidad de A dado que ocurrió B"*

---

## 2.4 Teorema de Bayes

Uno de los teoremas más importantes en la toma de decisiones empresariales:

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

Donde:
- $P(A|B)$: Probabilidad **posterior** (actualizada con nueva evidencia)
- $P(A)$: Probabilidad **prior** (creencia inicial)
- $P(B|A)$: **Verosimilitud**
- $P(B)$: **Evidencia** o probabilidad marginal

---

## 2.5 Implementación con Python

### Cálculos Básicos de Probabilidad

```python
import numpy as np
from scipy import stats
import itertools
import pandas as pd

# --- Probabilidad Clásica ---
# En un lote de 200 productos, 12 son defectuosos
total = 200
defectuosos = 12
p_defecto = defectuosos / total
print(f"P(defectuoso) = {p_defecto:.3f}")

# Probabilidad de NO defectuoso
p_bueno = 1 - p_defecto
print(f"P(no defectuoso) = {p_bueno:.3f}")
```

### Simulación de Frecuencias Relativas

```python
# Simulamos el lanzamiento de un dado 10,000 veces
np.random.seed(42)
n_simulaciones = 10000
dados = np.random.randint(1, 7, n_simulaciones)

# Probabilidad de obtener un 6
p_6 = np.mean(dados == 6)
print(f"P(6) simulada = {p_6:.4f} (teórica = {1/6:.4f})")

# Probabilidad de obtener par (2, 4, 6)
p_par = np.mean(dados % 2 == 0)
print(f"P(par) simulada = {p_par:.4f} (teórica = {0.5:.4f})")
```

### Probabilidad Condicional en Datos de Ventas

```python
# Datos históricos de ventas y clima
np.random.seed(42)
n_dias = 1000

datos = pd.DataFrame({
    'dia': range(n_dias),
    'lluvia': np.random.choice([True, False], n_dias, p=[0.3, 0.7]),
    'ventas_altas': np.random.choice([True, False], n_dias, p=[0.4, 0.6])
})

# Ajustamos: si llueve, menor probabilidad de ventas altas
for i in range(n_dias):
    if datos.loc[i, 'lluvia']:
        datos.loc[i, 'ventas_altas'] = np.random.choice([True, False], p=[0.25, 0.75])
    else:
        datos.loc[i, 'ventas_altas'] = np.random.choice([True, False], p=[0.45, 0.55])

# Calculamos probabilidades
P_lluvia = datos['lluvia'].mean()
P_ventas_altas = datos['ventas_altas'].mean()
P_lluvia_y_ventas = (datos['lluvia'] & datos['ventas_altas']).mean()

# P(ventas_altas | lluvia)
condicional = datos[datos['lluvia']]['ventas_altas'].mean()

print(f"P(lluvia) = {P_lluvia:.3f}")
print(f"P(ventas altas) = {P_ventas_altas:.3f}")
print(f"P(ventas altas | lluvia) = {condicional:.3f}")

# Verificación con fórmula de Bayes
P_lluvia_dado_ventas = P_lluvia_y_ventas / P_ventas_altas
print(f"P(lluvia | ventas altas) = {P_lluvia_dado_ventas:.3f}")
```

---

## 2.6 Ejemplo: Probabilidad en Ventas

**Problema:** Un equipo de ventas tiene las siguientes probabilidades:
- P(cerrar trato con lead caliente) = 0.6
- P(cerrar trato con lead frío) = 0.1
- 30% de los leads son calientes, 70% son fríos

¿Cuál es la probabilidad de cerrar un trato? Si se cerró un trato, ¿qué probabilidad hay de que el lead fuera caliente?

```python
# Datos del problema
P_caliente = 0.3
P_frio = 0.7
P_cierre_given_caliente = 0.6
P_cierre_given_frio = 0.1

# Probabilidad total de cierre
P_cierre = (P_caliente * P_cierre_given_caliente) + (P_frio * P_cierre_given_frio)
print(f"P(cierre) = {P_cierre:.3f}")

# Teorema de Bayes: P(caliente | cierre)
P_caliente_dado_cierre = (P_cierre_given_caliente * P_caliente) / P_cierre
print(f"P(caliente | cierre) = {P_caliente_dado_cierre:.3f}")
print(f"Interpretación: Si se cierra un trato, hay un {P_caliente_dado_cierre*100:.1f}% de probabilidad de que el lead fuera caliente")
```

---

## 2.7 Ejemplo: Probabilidad en Compras

**Problema:** Un comprador evalúa 3 proveedores. Las probabilidades de entrega a tiempo son:
- Proveedor A: 95% a tiempo, 40% de los pedidos
- Proveedor B: 90% a tiempo, 35% de los pedidos
- Proveedor C: 85% a tiempo, 25% de los pedidos

Si un pedido llegó a tiempo, ¿cuál es la probabilidad de que fuera del Proveedor A?

```python
# Probabilidades previas (distribución de pedidos)
P_A = 0.40
P_B = 0.35
P_C = 0.25

# Probabilidades de entrega a tiempo por proveedor
P_tiempo_given_A = 0.95
P_tiempo_given_B = 0.90
P_tiempo_given_C = 0.85

# Probabilidad total de entrega a tiempo
P_tiempo = P_A * P_tiempo_given_A + P_B * P_tiempo_given_B + P_C * P_tiempo_given_C
print(f"P(entrega a tiempo) = {P_tiempo:.4f}")

# Bayes: P(A | entrega a tiempo)
P_A_dado_tiempo = (P_tiempo_given_A * P_A) / P_tiempo
P_B_dado_tiempo = (P_tiempo_given_B * P_B) / P_tiempo
P_C_dado_tiempo = (P_tiempo_given_C * P_C) / P_tiempo

print(f"\nProbabilidades posteriores de cada proveedor DADO que llegó a tiempo:")
print(f"P(A | tiempo) = {P_A_dado_tiempo:.4f} ({P_A_dado_tiempo*100:.1f}%)")
print(f"P(B | tiempo) = {P_B_dado_tiempo:.4f} ({P_B_dado_tiempo*100:.1f}%)")
print(f"P(C | tiempo) = {P_C_dado_tiempo:.4f} ({P_C_dado_tiempo*100:.1f}%)")
```

---

## 2.8 Ejemplo: Probabilidad en Inventarios

**Problema:** Un almacén tiene 500 SKUs.
- 20% son categoría A (alto valor)
- 30% son categoría B (medio valor)  
- 50% son categoría C (bajo valor)

Probabilidad de que un SKU tenga rotación lenta:
- Categoría A: 5%
- Categoría B: 15%
- Categoría C: 30%

Se selecciona un SKU al azar y resulta tener rotación lenta. ¿Probabilidad de que sea categoría C?

```python
P_A = 0.20
P_B = 0.30
P_C = 0.50

P_lenta_given_A = 0.05
P_lenta_given_B = 0.15
P_lenta_given_C = 0.30

P_lenta = (P_A * P_lenta_given_A + P_B * P_lenta_given_B + P_C * P_lenta_given_C)
print(f"P(rotación lenta) = {P_lenta:.3f}")

P_C_dado_lenta = (P_lenta_given_C * P_C) / P_lenta
print(f"P(C | rotación lenta) = {P_C_dado_lenta:.3f}")
print(f"\n→ Hay un {P_C_dado_lenta*100:.1f}% de probabilidad de que un producto")
print(f"  de rotación lenta pertenezca a la categoría C")
```

---

## 2.9 Independencia de Eventos

Dos eventos A y B son independientes si:
$$P(A \cap B) = P(A) \cdot P(B)$$

O equivalentemente: $P(A|B) = P(A)$

```python
# Verificación de independencia
def son_independientes(A, B, datos):
    """Verifica si dos eventos son independientes"""
    P_A = datos[A].mean()
    P_B = datos[B].mean()
    P_A_y_B = (datos[A] & datos[B]).mean()
    
    producto = P_A * P_B
    print(f"P({A}) = {P_A:.4f}")
    print(f"P({B}) = {P_B:.4f}")
    print(f"P({A} ∩ {B}) = {P_A_y_B:.4f}")
    print(f"P({A}) × P({B}) = {producto:.4f}")
    
    if np.isclose(P_A_y_B, producto, atol=0.01):
        print("→ Los eventos son independientes")
    else:
        print("→ Los eventos NO son independientes")

# Ejemplo: ¿Ventas altas y lluvia son independientes?
son_independientes('lluvia', 'ventas_altas', datos)
```

---

## 2.10 Variable Aleatoria

Una **variable aleatoria (VA)** es una función que asigna un valor numérico a cada resultado de un experimento aleatorio.

### Tipos

| Tipo | Descripción | Ejemplo en Ventas |
|------|-------------|-------------------|
| **Discreta** | Toma valores contables | N° de pedidos recibidos hoy |
| **Continua** | Toma valores en un intervalo | Precio de venta de un producto |

### Función de Probabilidad

- **Discreta (PMF)**: $P(X = x)$ probabilidad de que X tome el valor exacto x
- **Continua (PDF)**: $f(x)$ densidad de probabilidad, $P(a \leq X \leq b) = \int_a^b f(x)dx$
- **Acumulada (CDF)**: $F(x) = P(X \leq x)$

---

## 2.11 Valor Esperado y Varianza

### Valor Esperado

**Discreta**: $E[X] = \sum_i x_i \cdot P(X = x_i)$

**Continua**: $E[X] = \int_{-\infty}^{\infty} x \cdot f(x) dx$

### Varianza

$$Var(X) = E[(X - \mu)^2] = E[X^2] - E[X]^2$$

### Propiedades

- $E[aX + b] = aE[X] + b$
- $Var(aX + b) = a^2 Var(X)$
- $E[X + Y] = E[X] + E[Y]$

```python
# Cálculo de valor esperado y varianza empírica
ventas_diarias = np.random.poisson(lam=20, size=365)
print(f"Media empírica de ventas diarias: {ventas_diarias.mean():.2f}")
print(f"Varianza empírica: {ventas_diarias.var():.2f}")
print(f"Desviación estándar: {ventas_diarias.std():.2f}")
print(f"Valor esperado teórico (Poisson): λ = 20")
```

---

## 2.12 Resumen

- La probabilidad cuantifica la incertidumbre entre 0 y 1
- Existen 3 enfoques: clásico, frecuencial, subjetivo
- Bayes permite actualizar creencias con nueva evidencia
- Las variables aleatorias mapean resultados a números
- El valor esperado y la varianza son medidas fundamentales

---

## Ejercicios Propuestos

1. **Ventas**: Si P(venta > $1000) = 0.3 y P(cliente frecuente) = 0.4, y P(venta > $1000 ∩ cliente frecuente) = 0.15, ¿son independientes?
2. **Compras**: Usando los datos de 3 proveedores, si un pedido llegó tarde, ¿cuál proveedor es el más probable culpable?
3. **Inventarios**: Calcula el valor esperado de la demanda diaria si P(demanda=10)=0.2, P(20)=0.5, P(30)=0.3

---

[← Anterior](01-intro-estadistica.md) | [Índice](index.md) | [Siguiente →](03-estadistica-descriptiva.md)
