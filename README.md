# 🏠 House Price Prediction

Proyecto de Machine Learning para predecir precios de viviendas.

## 🚀 Tecnologías
- Python
- Scikit-learn
- Pandas
- Streamlit

## 📊 Modelos usados
- Linear Regression
- Random Forest
- Gradient Boosting
- Extra Trees

## 🏆 Mejor modelo
Gradient Boosting (R² ≈ 0.79)

Aquí tienes la sección de cómo ejecutar el proyecto lista para pegar en tu README.md 👇

## ▶️ Cómo ejecutar el proyecto### 1. Clonar repositorio```bashgit clone https://github.com/TU_USUARIO/house_price_prediction_ml.gitcd house_price_prediction_ml

2. Crear entorno virtual (recomendado)
python -m venv venv
Activar entorno:
Windows
venv\Scripts\activate
Mac / Linux
source venv/bin/activate

3. Instalar dependencias
pip install -r requirements.txt

4. Entrenar el modelo
python train_model.py
Esto generará:


modelo entrenado (models/model.pkl)


columnas (models/columns.pkl)


gráficas en outputs/



5. Validar el modelo
python check_model.py
Mostrará:


predicciones vs valores reales


error promedio


error porcentual



6. Ejecutar la aplicación
python -m streamlit run app.py
Luego abre en el navegador:
http://localhost:8501

📊 Notas


El modelo utiliza transformación logarítmica para mejorar precisión


Se aplicó limpieza de datos y eliminación de outliers


El mejor modelo es Gradient Boosting
