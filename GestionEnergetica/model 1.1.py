import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, RocCurveDisplay
import joblib


def generate_energy_management_model():
    """
    Crea y entrena un modelo de Machine Learning para la detección de anomalías en el consumo de energía.
    """
    try:
        # Cargar los datos modificados.
        # Es crucial que este archivo tenga las anomalías que has simulado.
        df = pd.read_csv('data_modificada_15.csv')

    except FileNotFoundError:
        print("Error: El archivo 'data_modificada_15.csv' no se encontró.")
        print("Asegúrate de que el archivo esté en la misma carpeta que este script.")
        return

    # 1. Preparación de los datos: Creación de la variable objetivo 'anomalia'
    # Basado en la descripción, el consumo máximo modificado es 1.12 veces el valor máximo original.
    # El umbral para identificar anomalías se establecerá en base a la descripción del usuario.
    # Se considera anómalo un consumo total que exceda un umbral alto o que esté activo en horas no operativas.

    # Asumimos que los datos originales tenían un 'total_act_power' máximo.
    # Para simular las anomalías, el usuario aumentó los valores en 1.12x el máximo.
    # Podríamos encontrar un umbral basado en el 99.9 percentil de los datos originales
    # para evitar falsos positivos de picos de consumo normales.

    # Para este ejemplo, creamos las etiquetas basándonos en un umbral simple.
    # En un caso real, la lógica para crear 'anomalia' debe ser muy precisa.

    # Supongamos que cualquier valor superior a un umbral muy alto es una anomalía
    # (por ejemplo, el percentil 99.9 de los datos originales).
    df['anomalia'] = 0
    # Aquí se crea una etiqueta basada en un umbral. El usuario mencionó 1.12 veces el máximo.
    # Para este ejemplo, consideramos anómalo un valor que supere un valor alto.
    # Si sabes el valor del máximo original, puedes usarlo. Por ejemplo:
    # max_original = df['total_act_power'].max() / 1.12
    # threshold = max_original * 1.11
    # df.loc[df['total_act_power'] > threshold, 'anomalia'] = 1

    # Dado que no tenemos el valor original, se usa un percentil alto como umbral
    # para detectar los valores que el usuario modificó.
    threshold = df['total_act_power'].quantile(0.99)
    df.loc[df['total_act_power'] > threshold, 'anomalia'] = 1

    # También se deben etiquetar los consumos "fantasma" en horarios no operativos.
    # El usuario mencionó que el equipo debía estar apagado en ciertos horarios.
    # Por ejemplo, si el horario no operativo es de 0:00 a 6:00, y hay consumo, es una anomalía.
    # Asumimos que 'hora_del_dia' es la hora en formato 24h.
    # Asumimos que el equipo debe estar apagado de 23h a 6h (hora_del_dia de 23 a 5).
    df.loc[(df['hora_del_dia'].isin([23, 0, 1, 2, 3, 4, 5])) & (df['total_act_power'] > 0), 'anomalia'] = 1

    # 2. División de los datos
    # Seleccionamos las características (X) y la variable objetivo (y).
    features = ['total_act_power', 'hora_del_dia', 'dia_de_la_semana', 'hora_sin', 'hora_cos', 'dia_sin', 'dia_cos']
    X = df[features]
    y = df['anomalia']

    # Dividir el conjunto de datos en entrenamiento (80%) y prueba (20%).
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Número de anomalías en el conjunto de entrenamiento: {y_train.sum()}")
    print(f"Número de anomalías en el conjunto de prueba: {y_test.sum()}")

    # 3. Entrenamiento del modelo
    # Se usará un Random Forest Classifier, que es robusto para este tipo de problemas.
    print("\n--- ENTRENANDO EL MODELO ---")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    print("Entrenamiento completado.")

    # 4. Evaluación del modelo
    print("\n--- EVALUACIÓN DEL MODELO ---")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Reporte de clasificación
    print("\n--- REPORTE DE CLASIFICACIÓN ---")
    print("Precisión, Recall y F1-score del modelo:")
    print(classification_report(y_test, y_pred))

    # Matriz de Confusión
    print("\n--- MATRIZ DE CONFUSIÓN ---")
    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=['Normal (0)', 'Anomalía (1)'],
                         columns=['Predicho Normal (0)', 'Predicho Anomalía (1)'])
    print(df_cm)

    # Visualización de la Matriz de Confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_cm, annot=True, fmt='g', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.show()

    # Curva ROC y AUC
    # La curva ROC es excelente para evaluar el rendimiento de un clasificador binario.
    print("\n--- CURVA ROC Y AUC ---")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    print(f"Área bajo la curva ROC (AUC): {roc_auc:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (área = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.show()

    print("\nAnálisis de la Matriz de Confusión:")
    print(f"Verdaderos Positivos (VP): {cm[1, 1]} - Anomalías detectadas correctamente.")
    print(f"Falsos Positivos (FP): {cm[0, 1]} - Consumo normal clasificado erróneamente como anomalía.")
    print(f"Verdaderos Negativos (VN): {cm[0, 0]} - Consumo normal detectado correctamente.")
    print(f"Falsos Negativos (FN): {cm[1, 0]} - Anomalías que no fueron detectadas.")
    joblib.dump(model, "modelo_anomalias.pkl")
    print("✅ Modelo guardado como 'modelo_anomalias.pkl'")

# Llama a la función principal para ejecutar el script
if __name__ == "__main__":
    generate_energy_management_model()