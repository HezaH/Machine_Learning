import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, 
    mean_absolute_error, mean_squared_error
)
from utils import (
    calculate_metrics, find_optimal_cost_threshold, preprocess_data, find_closest_threshold_idx, plot_confusion_matrices)

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

X_train, y_train = df_train.drop(target_column, axis=1), df_train[target_column]
X_test, y_test = df_test.drop(target_column, axis=1), df_test[target_column]

list_data_stats = [set(df_train[col]) for col in df_train.columns]
for col, values in zip(df_train.columns, list_data_stats):
    print(f"Column: {col}, Min: {min(values)}, Max: {max(values)}")

# ===========================
# SCALERS
# ===========================
# Normalização
list_norm = [*preprocess_data(MinMaxScaler(), df_train, df_test, target_column, fit=True), "MinMaxScaler"]
# Padronização
list_standard = [*preprocess_data(StandardScaler(), df_train, df_test, target_column, fit=True), "StandardScaler"]
# RobustScaler
list_robust = [*preprocess_data(RobustScaler(), df_train, df_test, target_column, fit=True), "RobustScaler"]

# ===========================
# MODELAGEM INICIAL
# ===========================

# Dicionários para armazenar resultados
predictions = {}
residuals = {}
results = {}
calculateMetrics = []

# Custos hipotéticos definidos
cost_scenarios = {
    "Equal Costs (FN = FP)": 1.0,
    "FN twice as costly as FP": 2.0,
    "FN five times as costly as FP": 5.0,
    "FP twice as costly as FN": 0.5,
    "FP five times as costly as FN": 0.2,
}

# Criação dos plots: ROC e Precision-Recall
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
ax.set_title("Precision-Recall Curves")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.legend(loc='lower left')
ax.grid(True)

# Logistic Regression para os dados normalizados, padronizados e robustos
for list_data in [list_norm, list_standard, list_robust]:

    # Separar os dados
    X_train_model, X_test_model, y_train_model, y_test_model, method = list_data
    print(f"Method: {method}")

    # Treinar modelo
    model = LogisticRegression()
    model.fit(X_train_model, y_train_model)

    # Coeficientes
    print("Coefficients:\n", model.coef_)

    # Predições e probabilidades
    y_pred = model.predict(X_test_model)
    y_scores = model.predict_proba(X_test_model)[:, 1]

    # ROC Curve
    fpr, tpr, thresholds_roc = roc_curve(y_test_model, y_scores)
    roc_auc = auc(fpr, tpr)

    # Precision-Recall Curve
    precision, recall, thresholds_pr = precision_recall_curve(y_test_model, y_scores)
    ap = average_precision_score(y_test_model, y_scores)

    # Plot ROC no ax1
    fig_roc, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(fpr, tpr, lw=2, label=f'{method} (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'ROC Curve with Cost-Based Thresholds ({method})')
    ax1.grid(alpha=0.3)

    # Plot PRC no ax
    ax.plot(recall, precision, lw=2, label=f'{method} (AP = {ap:.2f})')

    # Cálculo dos limiares ótimos para cada cenário de custo
    optimal_cost_thresholds = {}
    for name, cost_ratio in cost_scenarios.items():
        opt_threshold, costs, thresholds_eval, cm_cost = find_optimal_cost_threshold(
            y_test_model, y_scores, cost_ratio, thresholds_roc
        )
        optimal_cost_thresholds[name] = {
            "threshold": opt_threshold,
            "costs": costs,
            "eval_thresholds": thresholds_eval,
            "conf_matrix": cm_cost
        }

    # Índices mais próximos do threshold ideal para plotagem
    threshold_indices = {
        name: find_closest_threshold_idx(thresholds_roc, data["threshold"])
        for name, data in optimal_cost_thresholds.items()
    }

    # Cores para destaque dos thresholds
    colors = ['red', 'green', 'purple', 'orange', 'brown']
    for i, (name, idx) in enumerate(threshold_indices.items()):
        ax1.scatter(fpr[idx], tpr[idx], marker='o', color=colors[i],
                    label=f"{method} - {name} (t={optimal_cost_thresholds[name]['threshold']:.3f})")

    # Confusion matrix padrão (para o threshold 0.5)
    cm_default = confusion_matrix(y_test_model, y_pred)
    tn, fp, fn, tp = cm_default.ravel()

    # Plot das matrizes de confusão para cada cenário
    plot_confusion_matrices(cm_default, cost_scenarios, optimal_cost_thresholds, method)

    # Tabela comparativa das métricas por cenário
    results = {}
    for name, data in optimal_cost_thresholds.items():
        threshold = data["threshold"]
        cm = data["conf_matrix"]
        tn, fp, fn, tp = cm.ravel()
        metrics = calculate_metrics(tn, fp, fn, tp)
        results[name] = {"Threshold": threshold, **metrics}

    # Impressão formatada
    print("\nComparison of Different Cost-Based Thresholds:")
    print("-" * 100)
    metrics_to_display = ['Threshold', 'Sensitivity (TPR)', 'Specificity (TNR)', 'Precision', 'F1 Score', 'Accuracy']
    
    print(f"{'Cost Scenario':<25}", end=" ")
    for metric in metrics_to_display:
        print(f"{metric:<15}", end=" ")
    print("\n" + "-" * 100)

    for name, metrics in results.items():
        print(f"{name:<25}", end="")
        for metric in metrics_to_display:
            print(f"{metrics[metric]:<15.3f}", end="")
        print()

    # Armazenar previsões e resíduos
    predictions[method] = (y_test_model, y_pred)
    residuals[method] = y_test_model - y_pred

    # Erros
    print(f"\nMAE:  {mean_absolute_error(y_test_model, y_pred):.3f}")
    print(f"MSE:  {mean_squared_error(y_test_model, y_pred):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test_model, y_pred)):.3f}")
    print('\n' + '-'*80 + '\n')

# Finalização dos gráficos
ax1.plot([0, 1], [0, 1], color='gray', linestyle='--')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve with Cost-Based Optimal Thresholds')
ax1.legend(loc="lower right")
ax1.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Plot all residual histograms in one figure
# Quando você vê um pico muito alto em resíduos 0, isso significa que muitas previsões foram exatas. Ja os picos em +1 (Falso Negativo) e -1 (Falso Positivo) correspondem a previsões erradas.
plt.figure(figsize=(12, 6))

for method, resid in residuals.items():
    sns.histplot(resid, bins=50, kde=True, label=method, stat='density', element='step')

plt.title('Residuals Distribution - Logistic Regression')
plt.xlabel('Residuals')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()







