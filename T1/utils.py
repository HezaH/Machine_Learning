import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (confusion_matrix, classification_report,
                             mean_absolute_error, mean_squared_error, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import KFold

# ===========================
# MÉTRICAS E AVALIAÇÕES
# ===========================

def calculate_metrics(tn, fp, fn, tp):
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        'Sensitivity (TPR)': sensitivity,
        'Specificity (TNR)': specificity,
        'Precision': precision,
        'F1 Score': f1,
        'Accuracy': accuracy,
        'TP': tp, 
        'FP': fp, 
        'TN': tn, 
        'FN': fn
    }

def find_optimal_cost_threshold(y_true, y_scores, cost_ratio, thresholds=None):
    if thresholds is None:
        evaluated_thresholds = np.linspace(0.01, 0.99, 100)
    else:
        evaluated_thresholds = thresholds

    total_costs = []
    n_samples = len(y_true)
    n_pos = np.sum(y_true)
    n_neg = n_samples - n_pos

    best_confusion_matrix = None

    for threshold in evaluated_thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        confusion_matrix_current = np.array([[tn, fp], [fn, tp]])

        cost_fp = fp / n_neg if n_neg > 0 else 0
        cost_fn = fn / n_pos if n_pos > 0 else 0
        total_cost = cost_fp + cost_ratio * cost_fn

        total_costs.append(total_cost)

        if total_cost == min(total_costs):
            best_confusion_matrix = confusion_matrix_current

    optimal_idx = np.argmin(total_costs)
    optimal_threshold = evaluated_thresholds[optimal_idx]

    return optimal_threshold, np.array(total_costs), evaluated_thresholds, best_confusion_matrix

def find_closest_threshold_idx(thresholds, target_threshold):
    return np.abs(thresholds - target_threshold).argmin()

def plot_confusion_matrices(
    conf_matrices: dict,         # e.g. {"Scenario A": cm_a, "Scenario B": cm_b, ...}
    method_name: str,            # e.g. "GradientBoosting"
    thresholds: dict = None,     # opcionalmente {"Scenario A": 0.3, ...}
    labels: list = None          # opcionalmente ['Negative','Positive']
):
    """
    Plota, em uma grade automática, uma heatmap para cada matriz de confusão
    em `conf_matrices`. Usa títulos baseados em method_name e keys do dict.
    
    - conf_matrices: dict de {cenário: confusion_matrix 2×2}
    - method_name: nome do modelo/preprocessing, exibido no título
    - thresholds: dict de {cenário: limiar ótimo} para mostrar junto do título
    - labels: lista de rótulos das classes; se None, usa [0,1]
    """
    scenarios = list(conf_matrices.keys())
    n = len(scenarios)
    # definir grid (até 3 colunas)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, scenario in zip(axes, scenarios):
        cm = conf_matrices[scenario]
        # normalizar
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        title = f"{method_name}\n{scenario}"
        if thresholds and scenario in thresholds:
            title += f"\nThreshold={thresholds[scenario]:.3f}"
        ax.set_title(title)
        
        # ticks e labels
        cls_labels = labels or [0, 1]
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(cls_labels); ax.set_yticklabels(cls_labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

        # anotações
        for i in (0,1):
            for j in (0,1):
                ax.text(j, i,
                        f"{cm[i,j]}\n({cm_norm[i,j]:.1%})",
                        ha="center", va="center",
                        color="white" if cm[i,j] > cm.max()/2 else "black")

    # desligar eixos extras
    for ax in axes[len(scenarios):]:
        ax.axis('off')

    plt.tight_layout()
    return fig


def regressor_plot(y_test, y_pred, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Predicted vs Actual
    ax1.scatter(y_test, y_pred)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    ax1.set_title(f"{title} - Predicted vs Actual")
    ax1.grid(True)

    # 2. Residuals Plot
    residuals = y_test - y_pred
    ax2.scatter(y_pred, residuals)
    ax2.axhline(0, color='red', linestyle='--')
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Residuals")
    ax2.set_title(f"{title} - Residual Plot")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, model_name=None, labels=None, normalize=False, cmap="Blues", cm = None):
    """
    Plota a matriz de confusão para classificação binária.

    Parâmetros:
    -----------
    y_true : array-like
        Rótulos verdadeiros (0 ou 1).
    y_pred : array-like
        Rótulos previstos (0 ou 1).
    model_name : str ou None
        Título opcional do gráfico (nome do modelo).
    labels : list de str ou None
        Nomes das classes. Ex.: ['Negative', 'Positive'].
        Se None, usa ['0', '1'].
    normalize : bool
        Se True, normaliza as células para proporções por linha.
    cmap : str
        Colormap para o heatmap.
    """
    # Define os labels
    if labels is None:
        labels = ['0', '1']

    # Calcula a matriz
    if cm is None:
        cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
        # Para exibir números absolutos, remova normalize ou use normalize=None

    # Cria a figura
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                     cmap=cmap, cbar=False,
                     xticklabels=labels, yticklabels=labels)

    # Configurações de eixo e título
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    title = "Confusion Matrix"
    if model_name:
        title += f" — {model_name}"
    if normalize:
        title += " (Normalized)"
    ax.set_title(title, fontsize=14)

    # plt.tight_layout()
    # plt.show()

#função para criar a matriz de confusão e relatório de classificação
def MatConf(verdadeiros, previstos, titulo, rotulos_x="AxisX", rotulos_y="AxisY"):
    conf_matrix = confusion_matrix(verdadeiros, previstos)
    plt.figure(figsize=(6, 5))
    s = sns.heatmap(conf_matrix, annot=True, cmap="Greens",
                    xticklabels=rotulos_x, yticklabels=rotulos_y,
                    fmt=".2f")
    s.set(xlabel="Rótulo Previsto", ylabel="Rótulo Verdadeiro", title=titulo)
    # plt.tight_layout()
    plt.show()

def preprocess_data(scaler, df_train, df_test, target_column, fit=True):
    # Drop target column and scale features
    if fit:
        X_train = scaler.fit_transform(df_train.drop(target_column, axis=1))
        X_test  = scaler.transform(df_test.drop(target_column, axis=1))
    else:
        X_train = scaler.transform(df_train.drop(target_column, axis=1))
        X_test  = scaler.transform(df_test.drop(target_column, axis=1))
    y_train = df_train[target_column]
    y_test  = df_test[target_column]
    return X_train, X_test, y_train, y_test

def get_model_scores(model, scaler, df_train, df_test, target_column='target'):
    """
    Pré-processa os dados, treina o modelo e retorna os scores (probabilidades ou decision function)
    para os dados de teste.

    Parâmetros:
    - model: instância do classificador
    - scaler: instância do escalonador
    - df_train: DataFrame de treino
    - df_test: DataFrame de teste
    - target_column: nome da coluna alvo

    Retorna:
    - y_scores: scores do modelo para df_test
    - y_true: rótulos reais de df_test
    """
    X_train, X_test, y_train, y_test = preprocess_data(
        scaler, df_train, df_test, target_column=target_column, fit=True
    )
    model.fit(X_train, y_train)
    if hasattr(model, "predict_proba"):
        # Se o modelo tem predict_proba, usamos para obter scores (ex.: classificadores)
        y_scores = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        # Se tiver decision_function, usamos ele
        y_scores = model.decision_function(X_test)
    else:
        # Para outros modelos, como os de regressão, usamos predict
        y_scores = model.predict(X_test)
    return y_scores, y_test

def pipeline_model(spling_configs, clf, X_train, y_train, X_test, y_test):
    
    for config in spling_configs:
        name = config['spling_name']
        method = config['spling_class']
        
        print(f"\n--- Método: {name} ---")
        
        if name == "Threshold":
            # Treina o modelo com dados normais
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)[:, 1]

            for thresh in method:
                y_pred = (probs >= thresh).astype(int)
                print(f"\nLimiar: {thresh:.2f}")
               
        
        elif name == "WithOut":
            y_pred = clf.predict(X_test)

        else:
            # Aplica técnica de balanceamento
            X_resampled, y_resampled = method.fit_resample(X_train, y_train)

            
            clf.fit(X_resampled, y_resampled)
            y_pred = clf.predict(X_test)

        print(classification_report(y_test, y_pred))

def cross_validate_models(model_configurations, scaler_configs, spling_configs, dataset, target_column, cv=5,
                          fit_scaler=True, scoring_type="accuracy"):
    """
    Performs cross-validation on a dataset, evaluating a set of machine learning models
    with different scaling/preprocessing methods. Additionally, computes evaluation metrics
    including a configurable scoring value, confusion matrix, classification report, MAE, MSE, and RMSE.

    Parameters:
        model_configurations (list of dict): List of dictionaries containing model classes
            and their parameters. To tune a parameter, include:
            - 'tuning_parameter': The name of the parameter to tune (e.g., 'n_neighbors')
            - 'parameter_range': List or range of values to test.
            Example:
                {'class_name': "KNeighborsClassifier",
                 'model_class': KNeighborsClassifier,
                 'tuning_parameter': 'n_neighbors',
                 'parameter_range': list(range(1, 40))}
        scaler_configs (list of dict): List of dictionaries containing scaler classes and names.
            Example:
                {'scaler_name': "MinMaxScaler", 'scaler_class': MinMaxScaler}
        dataset (pd.DataFrame): The complete dataset for training (including target column).
        target_column (str): Name of the target column.
        cv (int): Number of folds for cross-validation.
        fit_scaler (bool): If True, fits the scaler on training data; otherwise, only transforms.
        scoring_type (str): Metric to be observed: "accuracy", "f1", "precision", or "recall".
            Default is "accuracy".

    Returns:
        pd.DataFrame: DataFrame containing aggregated scores and evaluation metrics for each model/scaler combination.
    """

    def compute_scoring(y_true, y_pred, scoring_type):
        """
        Compute the scoring metric based on scoring_type.
        """
        if scoring_type == "accuracy":
            return np.mean(y_true == y_pred)
        elif scoring_type == "f1":
            # Using weighted average to account for multiple classes
            return f1_score(y_true, y_pred, average='weighted')
        elif scoring_type == "precision":
            return precision_score(y_true, y_pred, average='weighted')
        elif scoring_type == "recall":
            return recall_score(y_true, y_pred, average='weighted')
        else:
            # Default to accuracy
            return np.mean(y_true == y_pred)

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []  # list to store scores across folds

    fold = 1
    for train_index, test_index in kf.split(dataset):
        df_train = dataset.iloc[train_index]
        df_test  = dataset.iloc[test_index]
        print(f"Processing fold {fold}...")

        for scaler_config in scaler_configs:
            scaler = scaler_config['scaler_class']()
            scaler_name = scaler_config['scaler_name']

            X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(scaler, df_train, df_test, target_column, fit=fit_scaler)

            for model_config in model_configurations:
                model_class = model_config['model_class']
                model_class_name = model_config['class_name']
                # Extract base parameters (excluding meta keys)
                base_model_params = {k: v for k, v in model_config.items()
                                     if k not in ['model_class', 'class_name', 'tuning_parameter', 'parameter_range']}
                
                # Check if tuning is specified for this model.
                if 'tuning_parameter' in model_config and 'parameter_range' in model_config:
                    tuning_param = model_config['tuning_parameter']
                    best_score = -np.inf
                    best_model_params = None
                    best_evaluation = {}  # To store all evaluation metrics from the best parameter
                    # Loop over the range of parameter values
                    for param_testue in model_config['parameter_range']:
                        current_params = base_model_params.copy()
                        current_params[tuning_param] = param_testue
                        model = model_class(**current_params)
                        
                        y_pred = pipeline_model(spling_configs, model, X_train_scaled, y_train, X_test_scaled, y_test)
                        
                        score_testue = compute_scoring(y_test, y_pred, scoring_type)

                        # If this configuration gives a better score, save it
                        if score_testue > best_score:
                            best_score = score_testue
                            best_model_params = current_params.copy()
                            # Compute additional metrics for the best configuration
                            if "Regressor" in model_class_name:
                                # Applying discretization for regression
                                bins = np.linspace(y_test.min(), y_test.max(), num=5)  # Adjust the number of bins as needed
                                y_test_cat = np.digitize(y_test, bins=bins, right=False)
                                y_pred_cat = np.digitize(y_pred, bins=bins, right=False)

                                # For regression models, we can use MAE, MSE, RMSE
                                best_evaluation = {
                                    "Confusion_Matrix": confusion_matrix(y_test_cat, y_pred_cat).tolist(),
                                    "MAE": mean_absolute_error(y_test, y_pred),
                                    "MSE": mean_squared_error(y_test, y_pred),
                                    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
                                }
                                MatConf(y_test_cat, y_pred_cat, f"{model_class_name}_{scaler_name}")
                                regressor_plot(y_test, y_pred, f"{model_class_name}_{scaler_name}")
                            else:
                                best_evaluation = {
                                    "Confusion_Matrix": confusion_matrix(y_test, y_pred).tolist(),
                                    "Classification_Report": classification_report(y_test, y_pred, output_dict=True),
                                    "MAE": mean_absolute_error(y_test, y_pred),
                                    "MSE": mean_squared_error(y_test, y_pred),
                                    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
                                }
                                MatConf(y_test, y_pred, f"{model_class_name}_{scaler_name}")

                    model_identifier = f"{model_class_name}_{scaler_name}_{tuning_param}={param_testue}"
                    score_record = {
                        "Model_Scaler": f"{model_class_name}_{scaler_name}",
                        "Scoring": best_score,
                        "Parameters": best_model_params,
                        "Evaluation": best_evaluation
                    }
                    scores.append(score_record)
                else:
                    # No tuning; use base parameters
                    model = model_class(**base_model_params)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)

                    
                    MatConf(y_test, y_pred, f"{model_class_name}_{scaler_name}")

                    score_testue = compute_scoring(y_test, y_pred, scoring_type)
                    evaluation_metrics = {
                        "Confusion_Matrix": confusion_matrix(y_test, y_pred).tolist(),
                        "Classification_Report": classification_report(y_test, y_pred, output_dict=True),
                        "MAE": mean_absolute_error(y_test, y_pred),
                        "MSE": mean_squared_error(y_test, y_pred),
                        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
                    }

                    model_identifier = f"{model_class_name}_{scaler_name}"
                    score_record = {
                        "Model_Scaler": model_identifier,
                        "Scoring": score_testue,
                        "Parameters": base_model_params,
                        "Evaluation": evaluation_metrics
                    }
                    scores.append(score_record)
        fold += 1
    
    df_scores = pd.DataFrame(scores)
    
    return df_scores

