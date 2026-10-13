import os
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


def train_and_save_model():
    # Cargar un dataset de ejemplo
    iris = load_iris()
    X, y = iris.data, iris.target

    # Entrenar un modelo básico de Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Definir la ruta donde se guardará el modelo (fuera de app/, dentro de backend/)
    model_path = os.path.join(os.path.dirname(__file__), '..', 'trained_model.pkl')

    # Guardar el modelo entrenado
    joblib.dump(model, model_path)
    print(f"Modelo entrenado y guardado correctamente en: {model_path}")


if __name__ == "__main__":
    train_and_save_model()