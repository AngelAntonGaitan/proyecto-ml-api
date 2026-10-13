# 🌸 Machine Learning API con FastAPI

API RESTful desarrollada con **FastAPI** para realizar predicciones de clasificación sobre el dataset Iris utilizando un modelo de Machine Learning entrenado con **scikit-learn**.

---

## 🛠️ Tecnologías utilizadas

* **Python 3**
* **FastAPI**: Framework web moderno y de alto rendimiento.
* **Uvicorn**: Servidor ASGI para la ejecución de la API.
* **scikit-learn**: Entrenamiento y serialización del modelo ML.
* **Pandas & NumPy**: Procesamiento y manipulación de datos.
* **Joblib**: Carga del modelo persistido.

---

## 🚀 Instalación y ejecución local

### 1. Clonar el repositorio
```bash
git clone [https://github.com/AngelAntonGaitan/proyecto-ml-api.git](https://github.com/AngelAntonGaitan/proyecto-ml-api.git)
cd proyecto-ml-api
```

### 2. Crear y activar el entorno virtual
```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Iniciar el servidor
```bash
uvicorn backend.main:app --reload
```
La API estará disponible en http://127.0.0.1:8000.

## 📖 Documentación interactiva
Una vez en ejecución, puedes acceder a la documentación interactiva de Swagger UI en:

* Swagger UI: http://127.0.0.1:8000/docs

* ReDoc: http://127.0.0.1:8000/redoc

## 📩 Endpoint principal
* POST /predict

Ejemplo de solicitud (JSON):
```bash
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```
Ejemplo de respuesta:
```bash
{
  "prediction": "setosa"
}
```

