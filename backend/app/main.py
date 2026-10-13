import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Inicializar la aplicación FastAPI
app = FastAPI(
    title="API de Clasificación de Flores ML",
    description="Servicio para predecir la especie de flor Iris usando un modelo de Machine Learning.",
    version="1.0.0"
)

# Cargar el modelo guardado
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'trained_model.pkl')

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

# Mapeo de las clases predichas a los nombres reales de las flores Iris
SPECIES_MAP = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}


# Esquema de datos de entrada para la predicción
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def read_root():
    return {"message": "API de Machine Learning activa. Visita /docs para probar la interfaz."}


@app.post("/predict")
def predict(data: IrisInput):
    if model is None:
        raise HTTPException(status_code=500, detail="El modelo no se encuentra cargado.")

    # Convertir los datos recibidos a la estructura requerida por scikit-learn
    features = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    prediction = model.predict(features)[0]

    return {
        "prediction_class": int(prediction),
        "species": SPECIES_MAP.get(int(prediction), "Desconocida")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)