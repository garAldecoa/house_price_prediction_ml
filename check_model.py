import pandas as pd
import numpy as np
import joblib

model = joblib.load("model.pkl")

df = pd.read_csv("test_examples.csv")

X = df.drop("real_price", axis=1)
y_real = df["real_price"]

pred_log = model.predict(X)
pred = np.expm1(pred_log)
pred = np.clip(pred, 50000, 1500000)

results = pd.DataFrame({
    "Precio real": y_real,
    "Predicción": pred,
    "Diferencia": abs(y_real - pred),
    "Error %": (abs(y_real - pred) / y_real) * 100
})

print("\n📊 Primeras 15 predicciones:")
print(results.head(15))

print("\n📈 Resumen del modelo:")
print(f"Error promedio: ${results['Diferencia'].mean():,.2f}")
print(f"Error porcentual promedio: {results['Error %'].mean():.2f}%")
print(f"Error porcentual mediano: {results['Error %'].median():.2f}%")

print("\n🏠 Mejores predicciones:")
print(results.sort_values("Error %").head(5))

print("\n⚠️ Peores predicciones:")
print(results.sort_values("Error %", ascending=False).head(5))