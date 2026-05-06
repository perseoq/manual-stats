# Capítulo 1: Introducción a la Estadística

[← Índice](index.md) | [Índice](index.md) | [Siguiente →](02-intro-probabilidad.md)

---

## 1.1 ¿Qué es la Estadística?

La **estadística** es la ciencia que se ocupa de recolectar, organizar, analizar, interpretar y presentar datos para facilitar la toma de decisiones en condiciones de incertidumbre. En el contexto empresarial, la estadística permite transformar datos crudos en información accionable.

### Ramas de la Estadística

| Rama | Descripción | Ejemplo en Negocios |
|------|-------------|---------------------|
| **Descriptiva** | Resume y describe características de un conjunto de datos | Calcular el promedio de ventas diarias |
| **Inferencial** | Extrae conclusiones sobre una población a partir de una muestra | Estimar la demanda total del año basado en una semana |
| **Probabilidad** | Mide la incertidumbre de eventos aleatorios | Calcular la probabilidad de stockout |

---

## 1.2 Conceptos Fundamentales

### Población y Muestra

- **Población (N)**: Conjunto total de elementos que se desea estudiar
  - Ej: *Todos los productos del catálogo (10,000 SKUs)*
- **Muestra (n)**: Subconjunto representativo de la población
  - Ej: *500 productos seleccionados aleatoriamente*

```
Población: Todos los clientes de la empresa (50,000)
Muestra: 1,000 clientes encuestados
```

### Parámetros vs Estadísticos

| Concepto | Población | Muestra |
|----------|-----------|---------|
| Media | $\mu$ (parámetro) | $\bar{x}$ (estadístico) |
| Varianza | $\sigma^2$ (parámetro) | $s^2$ (estadístico) |
| Desviación | $\sigma$ (parámetro) | $s$ (estadístico) |
| Proporción | $p$ (parámetro) | $\hat{p}$ (estadístico) |

### Tipos de Variables

```
Variables
├── Cualitativas (Categóricas)
│   ├── Nominales (sin orden): Categoría de producto, región
│   └── Ordinales (con orden): Nivel de urgencia, rating
└── Cuantitativas (Numéricas)
    ├── Discretas (conteos): N° de unidades vendidas, N° de pedidos
    └── Continuas (mediciones): Precio, peso, tiempo de entrega
```

---

## 1.3 Escalas de Medición

| Escala | Operaciones | Ejemplo en Ventas |
|--------|-------------|-------------------|
| **Nominal** | Igualdad/desigualdad | Código de producto, categoría |
| **Ordinal** | Mayor/menor que | Clasificación A-B-C de productos |
| **Intervalo** | Suma/resta | Temperatura en cadena de frío |
| **Razón** | Multiplicación/división | Ingresos, costos, unidades (el cero es absoluto) |

---

## 1.4 Implementación con Python

### Configuración del Entorno

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configuración de visualización
plt.style.use('seaborn-v0_8')
sns.set_theme()
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
```

### Carga y Exploración de Datos de Ventas

```python
# Simulamos datos de ventas para una tienda minorista
np.random.seed(42)

datos_ventas = pd.DataFrame({
    'fecha': pd.date_range('2024-01-01', periods=365, freq='D'),
    'producto': np.random.choice(['Laptop', 'Mouse', 'Teclado', 'Monitor', 'Audífonos'], size=365),
    'categoria': np.random.choice(['Electrónica', 'Accesorios', 'Periféricos'], size=365),
    'unidades_vendidas': np.random.poisson(15, 365),
    'precio_unitario': np.random.uniform(10, 1500, 365).round(2),
    'region': np.random.choice(['Norte', 'Sur', 'Este', 'Oeste'], size=365),
    'canal': np.random.choice(['Online', 'Tienda'], size=365, p=[0.6, 0.4])
})

# Calculamos el ingreso total
datos_ventas['ingreso'] = datos_ventas['unidades_vendidas'] * datos_ventas['precio_unitario']

print("=== PRIMERAS 5 FILAS ===")
print(datos_ventas.head())

print("\n=== INFORMACIÓN DEL DATASET ===")
print(datos_ventas.info())

print("\n=== ESTADÍSTICAS DESCRIPTIVAS BÁSICAS ===")
print(datos_ventas[['unidades_vendidas', 'precio_unitario', 'ingreso']].describe())
```

**Salida esperada:**
```
=== PRIMERAS 5 FILAS ===
       fecha producto    categoria  unidades_vendidas  precio_unitario  region  canal    ingreso
0 2024-01-01   Laptop  Electrónica                 17          712.54   Norte  Online   12113.18
1 2024-01-02    Mouse   Accesorios                 11          829.46     Sur  Tienda    9124.06
2 2024-01-03  Teclado  Periféricos                 14          432.78   Este  Online    6058.92
3 2024-01-04  Monitor  Electrónica                  9          811.85  Oeste  Online    7306.65
4 2024-01-05  Audífonos  Accesorios                16          737.54   Norte  Online   11800.64
```

### Análisis Exploratorio Rápido

```python
# Ventas por categoría
print("=== VENTAS POR CATEGORÍA ===")
print(datos_ventas.groupby('categoria')['ingreso'].agg(['sum', 'mean', 'count']))

# Ventas por región
print("\n=== VENTAS POR REGIÓN ===")
print(datos_ventas.groupby('region')['unidades_vendidas'].sum())

# Comparativa Online vs Tienda
print("\n=== CANAL DE VENTA ===")
print(datos_ventas.groupby('canal')['ingreso'].sum())
```

---

## 1.5 Ejemplo Aplicado: Diagnóstico de Ventas

**Problema:** El gerente comercial quiere entender el comportamiento de ventas del último año para detectar patrones y anomalías.

```python
# Resumen ejecutivo para la gerencia
resumen = {
    'Ingreso Total Anual': f"${datos_ventas['ingreso'].sum():,.2f}",
    'Unidades Vendidas Totales': f"{datos_ventas['unidades_vendidas'].sum():,}",
    'Precio Promedio': f"${datos_ventas['precio_unitario'].mean():.2f}",
    'Ticket Promedio': f"${datos_ventas['ingreso'].mean():.2f}",
    'Días con Datos': len(datos_ventas),
    'Productos Distintos': datos_ventas['producto'].nunique(),
    'Categorías': datos_ventas['categoria'].nunique(),
    'Regiones': datos_ventas['region'].nunique()
}

print("=== DASHBOARD EJECUTIVO ===")
for k, v in resumen.items():
    print(f"{k:30s}: {v}")
```

---

## 1.6 Ejemplo Aplicado: Análisis de Compras

**Problema:** El departamento de compras necesita caracterizar el comportamiento de los proveedores.

```python
# Simulamos datos de compras
datos_compras = pd.DataFrame({
    'proveedor': np.random.choice(['Prov_A', 'Prov_B', 'Prov_C', 'Prov_D'], size=200),
    'dias_entrega': np.random.normal(7, 2, 200).round(0).clip(1, 20),
    'costo_unitario': np.random.uniform(5, 500, 200).round(2),
    'calidad': np.random.choice(['A', 'B', 'C'], size=200, p=[0.7, 0.2, 0.1]),
    'volumen_pedido': np.random.randint(10, 1000, 200)
})

print("=== ANÁLISIS DE PROVEEDORES ===")
print(datos_compras.groupby('proveedor').agg({
    'dias_entrega': ['mean', 'std'],
    'costo_unitario': 'mean',
    'volumen_pedido': 'sum'
}))
```

---

## 1.7 Ejemplo Aplicado: Análisis de Inventarios

**Problema:** El equipo de logística quiere entender la rotación de inventario.

```python
# Simulamos datos de inventario
datos_inventario = pd.DataFrame({
    'sku': [f'SKU_{i:04d}' for i in range(100)],
    'categoria': np.random.choice(['A', 'B', 'C'], size=100, p=[0.2, 0.3, 0.5]),
    'stock_actual': np.random.randint(0, 500, 100),
    'stock_seguridad': np.random.randint(10, 100, 100),
    'demanda_diaria': np.random.exponential(5, 100).round(1),
    'lead_time': np.random.choice([1, 2, 3, 5, 7], size=100, p=[0.3, 0.3, 0.2, 0.1, 0.1]),
    'costo_almacenamiento': np.random.uniform(0.5, 5, 100).round(2)
})

# Identificamos productos con riesgo de stockout
datos_inventario['riesgo_stockout'] = datos_inventario['stock_actual'] < datos_inventario['stock_seguridad']
print("=== PRODUCTOS EN RIESGO DE STOCKOUT ===")
print(f"Productos en riesgo: {datos_inventario['riesgo_stockout'].sum()} de {len(datos_inventario)}")
print(f"Porcentaje: {(datos_inventario['riesgo_stockout'].mean() * 100):.1f}%")

# Días de cobertura
datos_inventario['dias_cobertura'] = datos_inventario['stock_actual'] / datos_inventario['demanda_diaria'].clip(lower=0.1)
print("\n=== DÍAS DE COBERTURA POR CATEGORÍA ===")
print(datos_inventario.groupby('categoria')['dias_cobertura'].describe())
```

---

## 1.8 Importancia de la Estadística en los Negocios

### Aplicaciones Clave

| Área | Aplicación Estadística |
|------|----------------------|
| **Ventas** | Pronósticos de demanda, segmentación de clientes, análisis de elasticidad precio |
| **Compras** | Evaluación de proveedores, optimización de lotes, análisis de costos |
| **Inventarios** | Punto de reorden, stock de seguridad, clasificación ABC |
| **Marketing** | A/B testing, análisis de campañas, ROI |
| **Finanzas** | Riesgo, valoración, predicción de flujo de caja |
| **Operaciones** | Control de calidad, capacidad, productividad |

---

## 1.9 Flujo de Trabajo Estadístico

```
1. Definir el problema de negocio
   ↓
2. Recolectar datos relevantes
   ↓
3. Limpiar y preparar los datos
   ↓
4. Análisis exploratorio (EDA)
   ↓
5. Aplicar métodos estadísticos
   ↓
6. Interpretar resultados
   ↓
7. Tomar decisiones de negocio
   ↓
8. Monitorear y retroalimentar
```

---

## 1.10 Resumen

- La estadística convierte datos en decisiones
- Se divide en descriptiva e inferencial
- Población ≠ Muestra; Parámetro ≠ Estadístico
- Las variables pueden ser cualitativas o cuantitativas
- Python + Pandas permite análisis estadístico potente

---

## Ejercicios Propuestos

1. **Ventas**: Calcula el ingreso total por producto y encuentra el top 3
2. **Compras**: ¿Qué proveedor tiene el menor tiempo de entrega promedio?
3. **Inventarios**: ¿Cuántos SKUs tienen menos de 5 días de cobertura?

---

← [Índice](index.md) | [↑ Arriba](#capítulo-1-introducción-a-la-estadística) | [Siguiente →](02-intro-probabilidad.md)
