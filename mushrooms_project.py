# ============================================================
# PROYECTO IIA - CLASIFICACIÓN DE HONGOS
# Dataset: devzohaib/mushroom-edibility-classification (Kaggle)
# Modelos: SVM, MLP, Árbol tipo ID3-like (entropy)
#
# Requisitos (instalar antes):
#   pip install kagglehub scikit-learn pandas numpy
# ============================================================

import os
import kagglehub
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ------------------------------------------------------------
# 1. Descargar y cargar el dataset
# ------------------------------------------------------------

def descargar_y_cargar_dataset():
    """
    Descarga el dataset de Kaggle usando kagglehub y carga el CSV
    en un DataFrame de pandas.
    """
    print("Descargando dataset de Kaggle (devzohaib/mushroom-edibility-classification)...")
    path = kagglehub.dataset_download("devzohaib/mushroom-edibility-classification")
    print("Carpeta de descarga:", path)
    print("Archivos encontrados:", os.listdir(path))

    # Buscar el primer .csv disponible
    csv_files = [f for f in os.listdir(path) if f.lower().endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No se encontró ningún archivo CSV en la carpeta descargada.")

    csv_path = os.path.join(path, csv_files[0])
    print("Usando archivo CSV:", csv_path)

    df = pd.read_csv(csv_path, sep=';')
    print("Shape del dataset:", df.shape)
    print("Primeras filas:")
    print(df.head())
    print("Columnas del dataset:")
    print(df.columns)

    return df


# ------------------------------------------------------------
# 2. Preparar datos (target, features, split, preprocesador)
# ------------------------------------------------------------

def preparar_datos(df):
    """
    Define la columna objetivo 'class', la mapea a 0/1,
    separa X e y, hace train/test split y construye
    el preprocesador (OneHotEncoder para categóricas).
    """
    print("\nColumnas disponibles en el DataFrame:")
    print(df.columns)

    # En este dataset la columna objetivo es 'class' (e = edible, p = poisonous)
    TARGET_COL = "class"
    if TARGET_COL not in df.columns:
        raise ValueError(f"No se encontró la columna objetivo '{TARGET_COL}' en el dataset.")

    X = df.drop(columns=[TARGET_COL])
    y_raw = df[TARGET_COL]

    print("\nValores únicos originales de la clase:")
    print(y_raw.value_counts())

    # Mapear e/p a 0/1 (ejemplo:
    #   0 = edible
    #   1 = poisonous
    # Puedes invertirlo si quieres, mientras seas consistente.
    y = y_raw.map({'e': 0, 'p': 1})
    if y.isna().any():
        raise ValueError("Se encontraron valores de clase que no son 'e' ni 'p'. Revisa el dataset.")

    y = y.astype(int)
    print("\nValores únicos de y después de mapear (0/1):")
    print(y.value_counts())

    # Train/test split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTamaños de los conjuntos:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)

    # Todas las columnas de X son categóricas en este dataset
    categorical_features = X.columns.tolist()

    # OneHotEncoder para todas las columnas categóricas
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    return X_train, X_test, y_train, y_test, preprocessor


# ------------------------------------------------------------
# 3. Definición de modelos y configuraciones (diseño de experimento)
# ------------------------------------------------------------

def construir_modelos(preprocessor):
    """
    Define las configuraciones de modelos (SVM, MLP, Árbol tipo ID3)
    que se van a probar y devuelve una lista de (nombre, pipeline).
    Cada pipeline incluye el preprocesador + modelo.
    """
    model_configs = []

    # SVM - 3 variaciones importantes de parámetros
    svm_linear = SVC(kernel="linear", C=1.0, probability=True, random_state=42)
    svm_rbf_c1 = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
    svm_rbf_c10 = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)

    # MLP - 3 variaciones (arquitectura y activación)
    mlp_1_50_relu = MLPClassifier(
        hidden_layer_sizes=(50,),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42
    )

    mlp_2_50_20_relu = MLPClassifier(
        hidden_layer_sizes=(50, 20),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42
    )

    mlp_1_50_tanh = MLPClassifier(
        hidden_layer_sizes=(50,),
        activation="tanh",
        solver="adam",
        max_iter=500,
        random_state=42
    )

    # Árbol de decisión tipo ID3-like (criterion="entropy")
    tree_id3_like = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=None,     # sin límite para observar posible sobreajuste
        random_state=42
    )

    # Construir pipelines (preprocesador + modelo)
    def make_pipeline(name, estimator):
        return (
            name,
            Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("model", estimator)
                ]
            )
        )

    model_configs.append(make_pipeline("SVM_linear", svm_linear))
    model_configs.append(make_pipeline("SVM_rbf_C1", svm_rbf_c1))
    model_configs.append(make_pipeline("SVM_rbf_C10", svm_rbf_c10))

    model_configs.append(make_pipeline("MLP_1capa_50relu", mlp_1_50_relu))
    model_configs.append(make_pipeline("MLP_2capas_50_20relu", mlp_2_50_20_relu))
    model_configs.append(make_pipeline("MLP_1capa_50tanh", mlp_1_50_tanh))

    model_configs.append(make_pipeline("ID3_like_entropy", tree_id3_like))

    return model_configs


# ------------------------------------------------------------
# 4. Entrenamiento y evaluación de un modelo
# ------------------------------------------------------------

def evaluar_modelo(nombre_modelo, pipeline, X_train, X_test, y_train, y_test):
    """
    Entrena el pipeline (preprocesador + modelo) y devuelve un diccionario
    con las métricas de desempeño.
    """
    print("\n" + "=" * 70)
    print(f"Entrenando y evaluando modelo: {nombre_modelo}")
    print("=" * 70)

    # Entrenar
    pipeline.fit(X_train, y_train)

    # Predecir en test
    y_pred = pipeline.predict(X_test)

    # Métricas (asumiendo clase positiva = 1 = 'poisonous')
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f} (clase venenoso)")
    print(f"Recall   : {rec:.4f} (clase venenoso)")
    print(f"F1-score : {f1:.4f} (clase venenoso)")

    print("\nReporte de clasificación:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["edible (0)", "poisonous (1)"]
    ))

    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    # Devolver métricas para tabla comparativa
    res = {
        "modelo": nombre_modelo,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }
    return res


# ------------------------------------------------------------
# 5. Función principal
# ------------------------------------------------------------

def main():
    # 1) Descargar y cargar dataset
    df = descargar_y_cargar_dataset()

    # 2) Preparar datos (X_train, X_test, y_train, y_test, preprocesador)
    X_train, X_test, y_train, y_test, preprocessor = preparar_datos(df)

    # 3) Construir modelos (SVM, MLP, Árbol tipo ID3-like)
    modelos = construir_modelos(preprocessor)

    # 4) Ejecutar experimentos
    resultados = []
    for nombre, pipe in modelos:
        res = evaluar_modelo(nombre, pipe, X_train, X_test, y_train, y_test)
        resultados.append(res)

    # 5) Tabla comparativa de resultados
    resultados_df = pd.DataFrame(resultados)
    print("\n" + "#" * 70)
    print("TABLA COMPARATIVA DE RESULTADOS (ordenada por accuracy)")
    print("#" * 70)
    print(resultados_df.sort_values(by="accuracy", ascending=False))

    # 6) Guardar tabla a CSV (para usar en el informe/presentación)
    out_path = "resultados_mushroom_modelos.csv"
    resultados_df.to_csv(out_path, index=False)
    print(f"\nResultados guardados en: {os.path.abspath(out_path)}")


# ------------------------------------------------------------
# Punto de entrada
# ------------------------------------------------------------

if __name__ == "__main__":
    main()
