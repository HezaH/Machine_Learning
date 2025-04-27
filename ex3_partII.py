import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import time

def predict_precipitation_final(classifier, regressor, x):
    x_formatted = np.array(x).reshape(1, -1)
    class_pred = classifier.predict(x_formatted)[0]
    if class_pred == 0:
        return 0
    else:
        return regressor.predict(x_formatted)[0]
    
def plot_results(counter, result):
    """
    Plots, side by side, the confusion matrix and the classification report.
    
    Parameters:
      result: dictionary containing the keys:
          - "Model": model name/configuration
          - "Confusion_Matrix": confusion matrix (list of lists or array)
          - "Classification_Report": classification report (dictionary, obtained with output_dict=True)
    """
    # Extract information from the dictionary
    cm = result["Confusion_Matrix"]
    clf_report = result["Classification_Report"]
    model_name = result["Model"]
    
    # Create a DataFrame for the classification report
    clf_report_df = pd.DataFrame(clf_report).transpose()

    # Create the figure with two subplots (side by side)
    fig, ax = plt.subplots(ncols=2, figsize=(12, 5))
    
    # -- Plot the Confusion Matrix --
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', ax=ax[0])
    ax[0].set_title(f'Confusion Matrix\n{model_name}')
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('Actual')
    
    # -- Display the Classification Report as a Table --
    ax[1].axis('off')  # Turn off the axis, as we will only use the area for the table
    table = ax[1].table(cellText=clf_report_df.round(2).values,
                        rowLabels=clf_report_df.index,
                        colLabels=clf_report_df.columns,
                        cellLoc="center",
                        loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    ax[1].set_title(f'Classification Report\n{model_name}')
    
    plt.tight_layout()
    plt.savefig(f'plot_{result["Model"]}_{counter}.png')

def report_severity_metrics(y_true, y_pred):
    """
    Calcula e retorna as métricas para cada nível de severidade da precipitação.
    
    Severidade:
      - sem_chuva_leve: 0 <= x < 5
      - moderada: 5 <= x < 25
      - forte: 25 <= x < 50
      - tempestade: x >= 50
    """
    severity_levels = {
        "sem_chuva_leve": (0, 5),
        "moderada": (5, 25),
        "forte": (25, 50),
        "tempestade": (50, np.inf)
    }
    report = {}
    for level, (low, high) in severity_levels.items():
        mask = (y_true >= low) & (y_true < high)
        count = np.sum(mask)
        if count > 0:
            mse = mean_squared_error(y_true[mask], y_pred[mask])
            mae = mean_absolute_error(y_true[mask], y_pred[mask])
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true[mask], y_pred[mask])
            report[level] = {"count": int(count), "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}
        else:
            report[level] = {"count": 0, "MAE": None, "MSE": None, "RMSE": None, "R2": None}
    return report

# Start time measurement
start_time = time.time()
current_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'A652.pickle')

f = open(current_dir, 'rb')
(X_train, y_train, X_val, y_val, X_test, y_test) = pickle.load(f) 

print(f"Shapes: {X_train.shape}, {X_test.shape}, {X_val.shape}")

# Transform continuous values into binary labels
y_train = np.where(y_train == 0, 0, 1)
y_val = np.where(y_val == 0, 0, 1)
y_test = np.where(y_test == 0, 0, 1)

# Define configuration variables
target = 'target'
scoring_type = 'accuracy'  # or 'f1', 'precision', 'recall'

# Convert arrays to DataFrames and Series, and create a single DataFrame for each set
df_train = pd.concat([pd.DataFrame(X_train), pd.Series(y_train.ravel(), name=target)], axis=1).reset_index(drop=True)
df_val = pd.concat([pd.DataFrame(X_val), pd.Series(y_val.ravel(), name=target)], axis=1).reset_index(drop=True)
df_test = pd.concat([pd.DataFrame(X_test), pd.Series(y_test.ravel(), name=target)], axis=1).reset_index(drop=True)

df_global_train = pd.concat([df_train, df_val], axis=0, ignore_index=True)

# Separate features (X) and labels (y) for each set
X_train_data = df_train.drop(columns=[target])
y_train_data = df_train[target].values

X_val_data = df_val.drop(columns=[target])

X_test_data = df_test.drop(columns=[target])

# Preprocessing and scaling configuration
scaler_config = {'scaler_name': 'StandardScaler'}
feature_columns = X_train_data.columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('scaler', StandardScaler(), feature_columns)
    ]
)

# Definir as configurações dos modelos
model_classifiers = [
    {
        'class_name': "GradientBoostingClassifier",
        'model_class': GradientBoostingClassifier,
        'tuning_parameter': 'learning_rate',
        'random_state': 42,
        'parameter_range': [0.1, 0.01, 0.001],
    }
]

#Spling the data into train and test sets
sampling_configs = [
    {"sampling_name": "Threshold", 'sampling_class': np.arange(0.1, 1.0, 0.1)},
]

results = []

# ----------------
# Main loop: For each model, for each sampling technique, and for each tuning parameter value…
for model_config in model_classifiers:
    model_class_name = model_config['class_name']
    tuning_param = model_config['tuning_parameter']
    
    for sampling_config in sampling_configs:
        sampling_name = sampling_config['sampling_name']
        
        for param in model_config['parameter_range']:
            print(f"\nRunning: Model={model_config['class_name']}, {tuning_param}={param}, Sampling={sampling_name}")
            set_name = f"{model_class_name}&{tuning_param}&{param}&{sampling_name}"
            
            # No sampling applied, train on the standard pipeline and then adjust the threshold
            model = model_config['model_class'](**{tuning_param: param}, random_state=model_config['random_state'])
            pipeline_model = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            pipeline_model.fit(X_train_data, y_train_data)
            # Get probabilities on the validation set
            y_proba = pipeline_model.predict_proba(X_val_data)[:, 1]
            
            for threshold in sampling_config['sampling_class']:
                y_pred = (y_proba >= threshold).astype(int)
                cm = confusion_matrix(df_val[target], y_pred).tolist()
                acc = accuracy_score(df_val[target], y_pred)
                print(f"  Threshold={threshold:.1f} -> Accuracy: {acc:.4f}")
                class_report = classification_report(df_val[target], y_pred, output_dict=True)
                results.append({
                    "Model": set_name,
                    "Accuracy": acc,
                    'sampling': sampling_name,
                    'scaler': scaler_config['scaler_name'],
                    "threshold": threshold,
                    "Confusion_Matrix": cm,
                    "Tp": cm[0][0],
                    "Fp": cm[0][1],
                    "Fn": cm[1][0],
                    "Tn": cm[1][1],
                    "Classification_Report": class_report,
                    "MAE": mean_absolute_error(df_val[target], y_pred),
                    "MSE": mean_squared_error(df_val[target], y_pred),
                    "RMSE": np.sqrt(mean_squared_error(df_val[target], y_pred)),
                    "Y": df_val[target].tolist(),
                    "YPred": y_pred.tolist()
                })
           
            print("\n")
            
best_result = max(results, key=lambda x: x["Classification_Report"].get("1", {}).get("f1-score", 0))
print("------------------------------------------------------")
print("Best configuration obtained:")
print(best_result["Model"])
print("F1 score for class 1:", best_result["Classification_Report"].get("1", {}).get("f1-score", 0))
print("Threshold:", best_result["threshold"])

# Get the components of the string identifying the configuration
set_name = best_result["Model"]   # Ex.: "GradientBoostingClassifier&learning_rate&0.1&SMOTE"
components = set_name.split("&")
model_class_str = components[0]
tuning_param = components[1]
tuning_value = float(components[2])
sampling_name = components[3]

# Model mapping (here we only have GradientBoostingClassifier, but if there are more, add them)
model_class = model_classifiers[0]['model_class']

best_threshold = best_result["threshold"]
model_C = model_class(**{tuning_param: tuning_value}, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('scaler', StandardScaler(), feature_columns)
    ]
)

pipeline_C = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', model)
])

X_global_train = df_global_train.drop(columns=[target])
y_global_train = df_global_train[target]

pipeline_C.fit(X_global_train, y_global_train)
y_train_pred = pipeline_C.predict(X_global_train)
y_proba = pipeline_model.predict_proba(X_global_train)[:, 1]
y_pred = (y_proba >= best_threshold).astype(int)

# Extração dos exemplos positivos no conjunto de treino
mask = (y_pred == 1)
X_train_1 = X_global_train[mask]
y_train_1 = y_global_train[mask]

# --- Parte 2: Treinamento do Regressor (R) – Modelo Híbrido ---
# Definir configurações dos modelos de regressão
model_regression = [
    { 
        'class_name': "LinearRegression", 
        'model_class': LinearRegression,
        'params': {}
    },
    { 
        'class_name': "DecisionTreeRegressor", 
        'model_class': DecisionTreeRegressor,
        'params': {'random_state': 42}
    },
    { 
        'class_name': "RandomForestRegressor", 
        'model_class': RandomForestRegressor,
        'params': {'n_estimators': 100, 'random_state': 42}
    },
]

final_results = []
for model_config in model_regression:
    model_class_name = model_config['class_name']
    Param_reg = model_config.get('tuning_parameter', None)
    
    for sampling_config in sampling_configs:
        sampling_name = sampling_config['sampling_name']
        # Se não houver iteração de parâmetros, usamos [None]
        for param_reg in model_config.get('parameter_range', [None]):
            print(f"\nRodando: Modelo={model_class_name}, Parâmetro={model_config.get('tuning_parameter','')}"
                  f"={param_reg} Sampling={sampling_name}")
            
            # Para relatório, definimos um identificador:
            set_name = f"{model_class_name}&{model_config.get('tuning_parameter','')}&{param_reg}&{sampling_name}"
            
            # Preparar os parâmetros do modelo de regressão
            model_params_R = model_config['params'].copy()
            if Param_reg is not None and param_reg is not None:
                model_params_R[model_config['tuning_parameter']] = param_reg
            
            # Instanciar e treinar o modelo de regressão em (X_train_1, y_train_1)
            model_R = model_config['model_class'](**model_params_R)
            pipeline_R = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', model_R)
            ])
            pipeline_R.fit(X_train_1, y_train_1)
            print("Modelo de regressão (R) treinado.")
            
            # Obter as predições do regressor para os dados de teste
            # X_test (não processado) será utilizado; o pré-processador fará o escalonamento
            y_pred_reg = pipeline_R.predict(X_test)
            # Obter as predições do classificador C para os mesmos dados de teste
            y_pred_class = pipeline_C.predict(X_test)
            # Combinar os resultados: se o classificador prever 0, a predição final é 0; senão, usa a predição do regressor.
            y_pred_final = np.where(y_pred_class == 0, 0, y_pred_reg)
            
            # Para avaliação, compare y_pred_final com os valores contínuos reais do teste (y_test)
            mse = mean_squared_error(y_test, y_pred_final)
            mae = mean_absolute_error(y_test, y_pred_final)
            rmse = np.sqrt(mse)
            r2   = r2_score(y_test, y_pred_final)
            
            # Criar um relatório de métricas por severidade
            severity_report = report_severity_metrics(y_test, y_pred_final)
            
            final_results.append({
                "Model": set_name,
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "R2": r2,
                "Severity_Report": severity_report
            })
            print(f"Modelo: {set_name}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
  
# Converter os resultados finais para DataFrame e exibir
df_final_results = pd.DataFrame(final_results)
print("\nResultados finais dos modelos de regressão (modelo híbrido):")
print(df_final_results)

# =============================================================================
# Seleção do melhor modelo (por exemplo, o com menor RMSE)
# =============================================================================
best_model_result = df_final_results.loc[df_final_results['RMSE'].idxmin()]
print("\nMelhor modelo selecionado:")
print(best_model_result)

# Você pode também salvar os resultados, se necessário:
results_json_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'final_results.json')
df_final_results.to_json(results_json_path, orient='records', lines=True)

end_time = time.time()
print(f"\nTempo total de execução: {end_time - start_time:.2f} segundos")

