# ============================================================
# Mushroom Edibility Classification - ML Pipeline (SVM / MLP / ID3-like)
# Versión optimizada (rápida)
# Requisitos:
#   - Python 3.9+
#   - pip install pandas numpy scikit-learn kagglehub
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd

# Paquetes de ML
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

# Descarga de Kaggle (sin API key local, usa kagglehub)
try:
    import kagglehub
except Exception as e:
    raise ImportError(
        "No se pudo importar 'kagglehub'. Instala con: pip install kagglehub"
    ) from e


# ------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------
def _autodetect_read_csv(csv_path: str) -> pd.DataFrame:
    """
    Lee un CSV con autodetección de separador. Si falla, intenta con coma.
    """
    try:
        df = pd.read_csv(csv_path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(csv_path)
    return df


# ------------------------------------------------------------
# Carga del dataset
# ------------------------------------------------------------
def descargar_y_cargar_dataset() -> pd.DataFrame:
    """
    Descarga el dataset 'devzohaib/mushroom-edibility-classification' con kagglehub
    y retorna un DataFrame con limpieza mínima (reemplazo de '?').
    """
    print("Descargando dataset: devzohaib/mushroom-edibility-classification ...")
    path = kagglehub.dataset_download("devzohaib/mushroom-edibility-classification")
    print("Carpeta de descarga:", path)

    files = os.listdir(path)
    print("Archivos encontrados:", files)

    csv_files = [f for f in files if f.lower().endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(
            "No se encontró ningún CSV en la carpeta descargada. Revisa el dataset."
        )

    # Toma el primer CSV encontrado
    csv_path = os.path.join(path, csv_files[0])
    print("Archivo CSV seleccionado:", csv_path)

    # Autodetección del separador
    df = _autodetect_read_csv(csv_path)

    # Limpieza ligera: convertir '?' a una categoría explícita
    df = df.replace("?", "unknown")

    print("Shape del dataset:", df.shape)
    print("Columnas:", list(df.columns))
    print("Primeras filas:")
    print(df.head())

    return df


# ------------------------------------------------------------
# Preparación de datos
# ------------------------------------------------------------
def preparar_datos(df: pd.DataFrame):
    """
    Separa X, y, hace split estratificado y arma el preprocesador OneHotEncoder.
    """
    TARGET_COL = "class"
    if TARGET_COL not in df.columns:
        raise ValueError(f"No se encontró la columna objetivo '{TARGET_COL}' en el dataset.")

    X = df.drop(columns=[TARGET_COL])
    y_raw = df[TARGET_COL]

    print("\nDistribución original de la clase:")
    print(y_raw.value_counts())

    # Mapear edible (e) -> 0, poisonous (p) -> 1
    y = y_raw.map({"e": 0, "p": 1}).astype(int)
    if y.isna().any():
        raise ValueError("Se encontraron valores de clase distintos de 'e' o 'p'.")

    print("\nDistribución (0=edible, 1=poisonous):")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("\nShapes:")
    print("X_train:", X_train.shape, "X_test:", X_test.shape)

    categorical_features = X.columns.tolist()

    # Encoder DENSO y más liviano en memoria para acelerar MLP
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    except TypeError:
        # scikit-learn < 1.2
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)

    preprocessor = ColumnTransformer(
        transformers=[("cat", ohe, categorical_features)]
    )

    return X_train, X_test, y_train, y_test, preprocessor


# ------------------------------------------------------------
# Modelos y pipelines
# ------------------------------------------------------------
def construir_modelos(preprocessor: ColumnTransformer):
    """
    Devuelve una lista de (nombre, pipeline) con SVM (linear/rbf), MLP (variantes) e ID3-like.
    Optimizaciones para velocidad:
      - LinearSVC en vez de SVC(linear)
      - SVC RBF sin probability=True
      - MLP con early_stopping
    """
    modelos = []

    # --- SVM (rápidos) ---
    # LinearSVC es mucho más veloz que SVC(kernel="linear") y expone decision_function.
    svm_linear_fast = LinearSVC(C=1.0, random_state=42)

    # RBF sin probability=True para evitar Platt scaling (muy costoso)
    svm_rbf_c1_fast = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42, cache_size=1000)
    svm_rbf_c10_fast = SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42, cache_size=1000)

    # --- MLP (multicapa) con early stopping ---
    mlp_1_50_relu = MLPClassifier(
        hidden_layer_sizes=(50,),
        activation="relu",
        solver="adam",
        max_iter=400,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.1,
        batch_size=256,
        random_state=42,
    )
    mlp_2_50_20_relu = MLPClassifier(
        hidden_layer_sizes=(50, 20),
        activation="relu",
        solver="adam",
        max_iter=400,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.1,
        batch_size=256,
        random_state=42,
    )
    mlp_1_50_tanh = MLPClassifier(
        hidden_layer_sizes=(50,),
        activation="tanh",
        solver="adam",
        max_iter=400,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.1,
        batch_size=256,
        random_state=42,
    )

    # --- Árbol (ID3-like con entropía) ---
    tree_id3_like = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=None,
        random_state=42
    )

    def make_pipeline(name, estimator):
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
        return (name, pipe)

    modelos += [
        make_pipeline("SVM_linear_fast", svm_linear_fast),
        make_pipeline("SVM_rbf_C1_fast", svm_rbf_c1_fast),
        make_pipeline("SVM_rbf_C10_fast", svm_rbf_c10_fast),
        make_pipeline("MLP_1capa_50relu_ES", mlp_1_50_relu),
        make_pipeline("MLP_2capas_50_20relu_ES", mlp_2_50_20_relu),
        make_pipeline("MLP_1capa_50tanh_ES", mlp_1_50_tanh),
        make_pipeline("ID3_like_entropy", tree_id3_like),
    ]
    return modelos


# ------------------------------------------------------------
# Evaluación
# ------------------------------------------------------------
def evaluar_modelo(nombre_modelo: str,
                   pipeline: Pipeline,
                   X_train: pd.DataFrame,
                   X_test: pd.DataFrame,
                   y_train: pd.Series,
                   y_test: pd.Series) -> dict:
    """
    Entrena y evalúa un modelo. Devuelve un diccionario con métricas y los hiperparámetros.
    """
    print("\n" + "=" * 70)
    print(f"Entrenando y evaluando: {nombre_modelo}")
    print("=" * 70)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    # Scores para métricas AUC
    y_score = None
    # predict_proba solo existe en SVC si probability=True y en MLP/árbol sí existe
    if hasattr(pipeline, "predict_proba"):
        try:
            y_score = pipeline.predict_proba(X_test)[:, 1]
        except Exception:
            y_score = None
    # decision_function existe en LinearSVC y SVC
    if y_score is None and hasattr(pipeline, "decision_function"):
        try:
            y_score = pipeline.decision_function(X_test)
        except Exception:
            y_score = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec  = recall_score(y_test, y_pred, pos_label=1)
    f1   = f1_score(y_test, y_pred, pos_label=1)

    roc_auc = roc_auc_score(y_test, y_score) if y_score is not None else np.nan
    pr_auc  = average_precision_score(y_test, y_score) if y_score is not None else np.nan

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}  (clase venenoso)")
    print(f"Recall   : {rec:.4f}  (clase venenoso)")
    print(f"F1-score : {f1:.4f}  (clase venenoso)")
    if y_score is not None:
        print(f"ROC-AUC  : {roc_auc:.4f}")
        print(f"PR-AUC   : {pr_auc:.4f}")

    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=["edible (0)", "poisonous (1)"]))
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    # Log de hiperparámetros simples del estimador
    est = pipeline.named_steps["model"]
    hyperparams = {
        k: v for k, v in est.get_params().items()
        if isinstance(v, (int, float, str, bool, tuple, type(None)))
    }

    return {
        "modelo": nombre_modelo,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "hyperparams": str(hyperparams),
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    df = descargar_y_cargar_dataset()
    X_train, X_test, y_train, y_test, preprocessor = preparar_datos(df)
    modelos = construir_modelos(preprocessor)

    resultados = []
    for nombre, pipe in modelos:
        res = evaluar_modelo(nombre, pipe, X_train, X_test, y_train, y_test)
        resultados.append(res)

    resultados_df = pd.DataFrame(resultados)
    resultados_df = resultados_df.sort_values(by=["f1", "accuracy"], ascending=False)

    print("\n" + "#" * 70)
    print("TABLA COMPARATIVA (orden: F1 luego Accuracy)")
    print("#" * 70)
    print(resultados_df)

    out_csv = "resultados_mushroom_modelos.csv"
    resultados_df.to_csv(out_csv, index=False)
    print(f"\nResultados guardados en: {os.path.abspath(out_csv)}")


if __name__ == "__main__":
    main()
