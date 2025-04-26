import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score,
                            precision_score,
                            recall_score,
                            f1_score,
                            roc_curve,
                            roc_auc_score,
                            confusion_matrix)
from utils import (
    get_model_scores, plot_confusion_matrix_from_values)
import os
# ===========================
# CARREGAMENTO E PRÉ-PROCESSAMENTO
# ===========================
# Definição das colunas
name_columns = ["ESCT", "NDEP", "RENDA", "TIPOR", "VBEM", "NPARC", "VPARC", "TEL", "IDADE", "RESMS", "ENTRADA", "CLASSE"]
cat_columns = ["ESCT", "NDEP", "TIPOR", "TEL"]  # OBS.: 'TIPOR' e 'TEL' já vem com valores 0 e 1
num_columns = ["RENDA", "VBEM", "NPARC", "VPARC", "IDADE", "RESMS", "ENTRADA"]
target_column = "CLASSE"  # 1 se o cliente pagou a dívida

# Carregamento dos dados de treino e teste
path_base = os.path.dirname(os.path.realpath(__file__))
df_train = pd.read_csv(os.path.join(path_base, 'credtrain.txt'), sep='\t', header=None)
df_test = pd.read_csv(os.path.join(path_base, 'credtest.txt'), sep='\t', header=None)

df_train.columns = name_columns
df_test.columns = name_columns

# Separação entre features e target
X_test, y_test = df_test.drop(target_column, axis=1), df_test[target_column]
X_train, y_train = df_train.drop(target_column, axis=1), df_train[target_column]

# Divisão adicional: treino e validação
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, train_size=.5)
df_train_split = pd.DataFrame(X_train_split, columns=name_columns[:-1])
df_train_split[target_column] = y_train_split

df_val = pd.DataFrame(X_val, columns=name_columns[:-1])
df_val[target_column] = y_val

# Criação do pré-processador utilizando ColumnTransformer:
# - OneHotEncoder para as colunas categóricas.
# - StandardScaler para as colunas numéricas.
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_columns),
        ('num', StandardScaler(), num_columns)
    ]
)

# Ajuste do pré-processador com os dados de treino (apenas as features) e aplicação nas partições
X_train_split_transformed = preprocessor.fit_transform(X_train_split)
X_val_transformed = preprocessor.transform(X_val)
X_test_transformed = preprocessor.transform(X_test)

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

# Lista para armazenar os resultados
results = []

# Loop pelos modelos
for config in model_configurations:
    model_class_name = config['class_name']
    print(f"Processando modelo: {model_class_name}")
    
    # Instanciar o modelo com os parâmetros default e criar o pipeline
    clf = config['model_class']()
    pipeline_clf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    # Treina o modelo com os dados de treino split
    pipeline_clf.fit(X_train_split, y_train_split)
    
    # Predições e probabilidades para os conjuntos de validação e teste
    y_pred_val   = pipeline_clf.predict(X_val)
    y_proba_val  = pipeline_clf.predict_proba(X_val)[:, 1]
    
    y_pred_test  = pipeline_clf.predict(X_test)
    y_proba_test = pipeline_clf.predict_proba(X_test)[:, 1]
    
    # Cálculo da curva ROC para o conjunto de validação
    fpr_val, tpr_val, thresholds_val = roc_curve(y_val, y_proba_val)
    roc_auc_val = roc_auc_score(y_val, y_proba_val)
    
    # Cálculo da curva ROC para o conjunto de teste
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_proba_test)
    roc_auc_test = roc_auc_score(y_test, y_proba_test)
    
    # Calcular G-Mean e melhor limiar com dados de validação
    gmeans = np.sqrt(tpr_val * (1 - fpr_val))
    ix = np.argmax(gmeans)
    best_threshold = thresholds_val[ix]
    print(f"Melhor limiar (Validação) para {model_class_name}: {best_threshold:.4f}, G-Mean: {gmeans[ix]:.4f}")

    # Matriz de confusão calculada com os dados de teste (usando threshold 0.5)
    cm = confusion_matrix(y_test, (y_proba_test >= 0.5).astype(int))
    tn, fp, fn, tp = cm.ravel()
    plot_confusion_matrix_from_values(tp, tn, fp, fn, title=f"{model_class_name} - Matriz de Confusão (Teste)")
    
    # Cálculo das métricas com base na validação
    accuracy  = accuracy_score(y_val, y_pred_val)
    precision = precision_score(y_val, y_pred_val, zero_division=0)
    recall    = recall_score(y_val, y_pred_val, zero_division=0)
    f1        = f1_score(y_val, y_pred_val, zero_division=0)
    
    results.append({
        'class_name': model_class_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc_val': roc_auc_val,
        'roc_auc_test': roc_auc_test
    })
    
    # Plotar as curvas ROC para os dois conjuntos (Validação e Teste) no mesmo gráfico
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Tangente')
    plt.plot(fpr_val, tpr_val, label=f"Validação (AUC = {roc_auc_val:.2f})")
    plt.plot(fpr_test, tpr_test, label=f"Teste (AUC = {roc_auc_test:.2f})")
    plt.scatter(fpr_val[ix], tpr_val[ix], marker='o', color='black', 
                label=f"Melhor limiar (Validação): {best_threshold:.2f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Curva ROC - {model_class_name}")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

# Identificação do melhor modelo com base no AUC do conjunto de teste
# Você pode optar por outro critério combinando várias métricas se desejar.
best_model = max(results, key=lambda item: item['roc_auc_test'])
print("------------------------------------------------------")
print("Melhor modelo identificado:")
print(f"Modelo: {best_model['class_name']}")
print(f"Acurácia (Validação): {best_model['accuracy']:.4f}")
print(f"Precisão (Validação): {best_model['precision']:.4f}")
print(f"Recall (Validação): {best_model['recall']:.4f}")
print(f"F1 Score (Validação): {best_model['f1']:.4f}")
print(f"ROC AUC (Validação): {best_model['roc_auc_val']:.4f}")
print(f"ROC AUC (Teste): {best_model['roc_auc_test']:.4f}")




