import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor


# =========================
# 1. Cargar datos
# =========================
df = pd.read_csv("data/data.csv")

print("📊 Dataset cargado:")
print(df.head())
print("\nColumnas:")
print(df.columns)


# =========================
# 2. Limpieza inicial
# =========================
if "date" in df.columns:
    df = df.drop("date", axis=1)

for col in df.columns:
    if "id" in col.lower():
        df = df.drop(col, axis=1)


# =========================
# 3. Definir target
# =========================
TARGET = "price" if "price" in df.columns else df.columns[-1]

print(f"\n🎯 Variable objetivo: {TARGET}")


# =========================
# 4. Asegurar target numérico
# =========================
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET])


# =========================
# 5. Eliminar outliers del precio
# =========================
low_price = df[TARGET].quantile(0.02)
high_price = df[TARGET].quantile(0.98)

df = df[(df[TARGET] > low_price) & (df[TARGET] < high_price)]
df = df[df[TARGET] < 1_500_000]
df = df[df[TARGET] > 50_000]


# =========================
# 6. Filtrar valores absurdos
# =========================
filters = {
    "bedrooms": (1, 10),
    "bathrooms": (1, 10),
    "sqft_living": (200, 10000),
    "sqft_lot": (500, 500000),
    "floors": (1, 5),
    "yr_built": (1800, 2025),
}

for col, (min_val, max_val) in filters.items():
    if col in df.columns:
        df = df[(df[col] >= min_val) & (df[col] <= max_val)]


# =========================
# 7. Feature engineering
# =========================
if "yr_built" in df.columns:
    df["house_age"] = 2025 - df["yr_built"]

if "yr_renovated" in df.columns:
    df["was_renovated"] = (df["yr_renovated"] > 0).astype(int)

if "sqft_living" in df.columns and "sqft_lot" in df.columns:
    df["total_size"] = df["sqft_living"] + df["sqft_lot"]

if "bathrooms" in df.columns and "bedrooms" in df.columns:
    df["bath_per_room"] = df["bathrooms"] / (df["bedrooms"] + 1)

if "sqft_living" in df.columns and "bedrooms" in df.columns:
    df["sqft_per_bedroom"] = df["sqft_living"] / (df["bedrooms"] + 1)

if "sqft_above" in df.columns and "sqft_basement" in df.columns:
    df["basement_ratio"] = df["sqft_basement"] / (df["sqft_above"] + 1)


# =========================
# 8. Separar X / y
# =========================
X = df.drop(TARGET, axis=1)
y = df[TARGET]

# Transformación logarítmica del precio
y_log = np.log1p(y)


# =========================
# 9. Limpieza de variables
# =========================
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

X[num_cols] = X[num_cols].fillna(X[num_cols].median())

for col in cat_cols:
    if X[col].mode().empty:
        X[col] = X[col].fillna("Unknown")
    else:
        X[col] = X[col].fillna(X[col].mode()[0])

X = pd.get_dummies(X, drop_first=True)


# =========================
# 10. División train/test
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)


# =========================
# 11. Modelos
# =========================
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=25,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42
    ),

    "Extra Trees": ExtraTreesRegressor(
    n_estimators=300,
    max_depth=25,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
}

results = {}


# =========================
# 12. Entrenar y evaluar
# =========================
for name, model in models.items():
    print(f"\n🔧 Entrenando: {name}")

    model.fit(X_train, y_train)

    y_pred_log = model.predict(X_test)

    # Regresar predicciones a precio real
    y_pred = np.expm1(y_pred_log)
    y_test_real = np.expm1(y_test)

    mae = mean_absolute_error(y_test_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred))
    r2 = r2_score(y_test_real, y_pred)

    results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    print(f"📊 {name}")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")


# =========================
# 13. Elegir mejor modelo
# =========================
best_model_name = max(results, key=lambda x: results[x]["R2"])
best_model = models[best_model_name]

print(f"\n🏆 Mejor modelo: {best_model_name}")

# =========================
# 13.1 Gráficas y análisis
# =========================

results_df = pd.DataFrame(results).T
results_df = results_df.sort_values("R2", ascending=False)

print("\n📊 Comparación de modelos:")
print(results_df)

# -------------------------
# Gráfica MAE y RMSE
# -------------------------
plt.figure(figsize=(10, 6))
results_df[["MAE", "RMSE"]].plot(kind="bar")
plt.title("Comparación de modelos - Errores")
plt.xlabel("Modelo")
plt.ylabel("Error")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("outputs/comparacion_errores.png")
plt.show()

# -------------------------
# Gráfica R² (separada)
# -------------------------
plt.figure(figsize=(8, 5))
plt.bar(results_df.index, results_df["R2"])
plt.title("Comparación de modelos - R²")
plt.xlabel("Modelo")
plt.ylabel("R²")
plt.xticks(rotation=20)
plt.ylim(0, 1)  # para que se vea bien la escala
plt.tight_layout()
plt.savefig("outputs/comparacion_r2.png")
plt.show()

# -------------------------
# Entrenamiento vs prueba
# -------------------------
train_pred_log = best_model.predict(X_train)
test_pred_log = best_model.predict(X_test)

train_pred = np.expm1(train_pred_log)
test_pred = np.expm1(test_pred_log)

y_train_real = np.expm1(y_train)
y_test_real = np.expm1(y_test)

r2_train = r2_score(y_train_real, train_pred)
r2_test = r2_score(y_test_real, test_pred)

print("\n📈 Entrenamiento vs prueba:")
print(f"R² entrenamiento: {r2_train:.4f}")
print(f"R² prueba       : {r2_test:.4f}")

plt.figure(figsize=(7, 5))
plt.bar(["Entrenamiento", "Prueba"], [r2_train, r2_test])
plt.title("R² Entrenamiento vs Prueba")
plt.ylabel("R²")
plt.tight_layout()
plt.savefig("outputs/entrenamiento_vs_prueba.png")
plt.show()

# -------------------------
# Importancia de variables
# -------------------------
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    features = X.columns

    indices = np.argsort(importances)[-10:]

    plt.figure(figsize=(9, 6))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.title("Top 10 variables más importantes")
    plt.xlabel("Importancia")
    plt.tight_layout()
    plt.savefig("outputs/importancia_variables.png")
    plt.show()

# -------------------------
# Predicción vs valores reales
# -------------------------
plt.figure(figsize=(7, 7))
plt.scatter(y_test_real, test_pred, alpha=0.5)

plt.plot(
    [y_test_real.min(), y_test_real.max()],
    [y_test_real.min(), y_test_real.max()]
)

plt.title("Predicción vs Valores Reales")
plt.xlabel("Precio real")
plt.ylabel("Precio predicho")
plt.tight_layout()
plt.savefig("outputs/prediccion_vs_real.png")
plt.show()

print("\n✅ Gráficas guardadas:")
print("- outputs/comparacion_errores.png")
print("- outputs/comparacion_r2.png")
print("- outputs/entrenamiento_vs_prueba.png")
print("- outputs/importancia_variables.png")
print("- outputs/prediccion_vs_real.png")

# -------------------------
# Gráfica de residuos
# -------------------------
residuos = y_test_real - test_pred

plt.figure(figsize=(7, 5))
plt.scatter(test_pred, residuos, alpha=0.5)

plt.axhline(y=0)  # línea en 0

plt.title("Residuos vs Predicción")
plt.xlabel("Predicción")
plt.ylabel("Residuo (Error)")
plt.tight_layout()
plt.savefig("outputs/residuos.png")
plt.show()

plt.figure(figsize=(7, 5))
plt.hist(residuos, bins=30)
plt.title("Distribución de residuos")
plt.xlabel("Error")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig("outputs/distribucion_residuos.png")
plt.show()

# =========================
# 14. Guardar archivos
# =========================
joblib.dump(best_model, "models/model.pkl")
joblib.dump(X.columns.tolist(), "models/columns.pkl")
joblib.dump(True, "models/uses_log_target.pkl")

# Guardar ejemplos reales para validar después
test_examples = X_test.copy()
test_examples["real_price"] = np.expm1(y_test)

test_examples.to_csv("validation/test_examples.csv", index=False)

print("✅ Ejemplos de prueba guardados en validation/test_examples.csv")

print("\n✅ Modelo, columnas y configuración guardados correctamente")