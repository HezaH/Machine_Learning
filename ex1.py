import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                             mean_absolute_error, mean_squared_error, roc_curve,
                             accuracy_score, classification_report, roc_auc_score)
from utils import (
    get_model_scores, plot_confusion_matrix_from_values)
import os
# ===========================
# CARREGAMENTO E PRÉ-PROCESSAMENTO
# ===========================

name_columns = ["ESCT", "NDEP", "RENDA", "TIPOR", "VBEM", "NPARC", "VPARC", "TEL", "IDADE", "RESMS", "ENTRADA", "CLASSE"]
target_column = "CLASSE"

path_base = os.path.dirname(os.path.realpath(__file__))
df_train = pd.read_csv(os.path.join(path_base, 'credtrain.txt'), sep='\t', header=None)
df_test = pd.read_csv(os.path.join(path_base, 'credtest.txt'), sep='\t', header=None)

df_train.columns = name_columns
df_test.columns = name_columns

X_test, y_test = df_test.drop(target_column, axis=1), df_test[target_column]
X_train, y_train = df_train.drop(target_column, axis=1), df_train[target_column]

X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, train_size=.5)
df_train_split = pd.DataFrame(X_train_split, columns=name_columns[:-1])
df_train_split[target_column] = y_train_split

df_val = pd.DataFrame(X_val, columns=name_columns[:-1])
df_val[target_column] = y_val

list_data_stats = [set(df_train[col]) for col in df_train.columns]
for col, values in zip(df_train.columns, list_data_stats):
    print(f"Column: {col}, Min: {min(values)}, Max: {max(values)}")

# Define model configurations; include tuning keys for models that require parameter tuning.
model_configurations = [
    {
        'class_name': "KNeighborsClassifier",
        'model_class': KNeighborsClassifier
    },
    {
        'class_name': "LogisticRegression",
        'model_class': LogisticRegression
    },
    {
        'class_name': "GradientBoostingClassifier",
        'model_class': GradientBoostingClassifier,
    }
]

# Define scaler configurations
scaler_configs = [
    {"scaler_name": "MinMaxScaler", 'scaler_class': MinMaxScaler},
    {"scaler_name": "StandardScaler", 'scaler_class': StandardScaler},
    {"scaler_name": "RobustScaler", 'scaler_class': RobustScaler}
]

results = []

# Loop principal
for model_config in model_configurations:
    model_class_name = model_config['class_name']
    model_class = model_config['model_class']

    # Listas para armazenar dados das curvas ROC
    roc_data = []

    for scaler_config in scaler_configs:
        scaler_name = scaler_config['scaler_name']
        scaler_class = scaler_config['scaler_class']
        set_name = f"{model_class_name}_{scaler_name}"

        print(f"\nRodando: Modelo={model_class_name}, Scaler={scaler_name}")

        # Instanciar scaler e modelo
        scaler = scaler_class()
        model = model_class()

        # Obter scores de validação
        y_val_scores, y_val = get_model_scores(model, scaler, df_train_split, df_val, target_column=target_column)

        # Calcular curva ROC e AUC
        fpr, tpr, thresholds = roc_curve(y_val, y_val_scores)
        auc = roc_auc_score(y_val, y_val_scores)

        # Calcular G-Mean e melhor limiar
        gmeans = np.sqrt(tpr * (1 - fpr))
        ix = np.argmax(gmeans)
        best_threshold = thresholds[ix]
        print(f"Melhor limiar (G-Mean): {best_threshold:.4f}, G-Mean: {gmeans[ix]:.4f}")

        # Armazenar dados da curva ROC
        roc_data.append({
            'scaler_name': scaler_name,
            'fpr': fpr,
            'tpr': tpr,
            'auc': auc,
            'best_threshold': best_threshold,
            'best_point': (fpr[ix], tpr[ix])
        })

        # Avaliação no conjunto de teste
        y_test_scores, y_test = get_model_scores(model, scaler, df_train, df_test, target_column=target_column)
        y_test_pred = (y_test_scores >= best_threshold).astype(int)

        # Matriz de confusão e métricas
        cm = confusion_matrix(y_test, y_test_pred)
        tn, fp, fn, tp = cm.ravel()
        plot_confusion_matrix_from_values(tp, tn, fp, fn, title=set_name + ' Matriz de Confusão')

        cr = classification_report(y_test, y_test_pred)
        acc = accuracy_score(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)
        mse = mean_squared_error(y_test, y_test_pred)
        rmse = np.sqrt(mse)

        print(f"Matriz de Confusão para {set_name}:\n{cm}")
        print(f"Relatório de Classificação para {set_name}:\n{cr}")

        results.append({
            "Model": set_name,
            "Accuracy": acc,
            "Confusion_Matrix": cm,
            "Classification_Report": cr,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "G-Mean": gmeans[ix],
        })

    # Plotar todas as curvas ROC após iterar sobre os scalers
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', label='Aleatório')

    for data in roc_data:
        plt.plot(data['fpr'], data['tpr'], label=f"{data['scaler_name']} (AUC = {data['auc']:.2f})")
        plt.scatter(*data['best_point'], marker='o', color='black', label=f"Melhor limiar: {data['scaler_name']}")

    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title(f'Curva ROC - {model_class_name}')
    plt.legend()
    plt.grid()
    plt.show()

# Criar DataFrame de resultados
df_res = pd.DataFrame(results)