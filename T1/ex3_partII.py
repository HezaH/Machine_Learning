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
import seaborn as sns

def report_severity_metrics(y_true, y_pred):
    """
    Calcula e retorna as métricas para cada nível de severidade da precipitação.
    
    Severidade:
      - sem_chuva_leve: 0 <= x < 5
      - moderada: 5 <= x < 25
      - forte: 25 <= x < 50
      - tempestade: x >= 50
    """
    # Converter as entradas para arrays 1D (unidimensionais)
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
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

def plot_severity_metrics(severity_report, model_name):
    """Plota métricas por nível de severidade"""
    levels = list(severity_report.keys())
    metrics = ['MAE', 'RMSE', 'R2']
    
    plt.figure(figsize=(15, 5))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        values = [severity_report[level][metric] for level in levels]
        sns.barplot(x=levels, y=values, palette='viridis')
        plt.title(metric)
        plt.xticks(rotation=45)
        plt.ylabel(metric)
    
    plt.suptitle(f'Métricas por Severidade - {model_name}')
    plt.tight_layout()
    plt.savefig(f'severity_metrics_{model_name}.png')
    plt.close()

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
        # 'tuning_parameter': 'learning_rate',
        # 'random_state': 42,
        # 'parameter_range': [0.1, 0.01, 0.001],
    }
]

#Spling the data into train and test sets
sampling_configs = [
    {"sampling_name": "Threshold", 'sampling_class': np.arange(0.1, 1.0, 0.05)},
]

results = []

# ----------------
# Main loop: For each model, for each sampling technique, and for each tuning parameter value…
for model_config in model_classifiers:
    model_class_name = model_config['class_name']
    # tuning_param = model_config['tuning_parameter']
    
    for sampling_config in sampling_configs:
        sampling_name = sampling_config['sampling_name']
        
        # for param in model_config['parameter_range']:
        print(f"\nRunning: Model={model_config['class_name']}, Sampling={sampling_name}")
        set_name = f"{model_class_name}&{sampling_name}"
        
        # No sampling applied, train on the standard pipeline and then adjust the threshold
        model = model_config['model_class']()
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


sampling_name = components[1]

# Model mapping (here we only have GradientBoostingClassifier, but if there are more, add them)
model_class = model_classifiers[0]['model_class']

best_threshold = best_result["threshold"]
model_C = model_class()

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
#O método .predict() gera os rótulos preditos (por exemplo, 0 ou 1 em um problema de classificação) 
# para cada exemplo contido em X_global_train e armazena esses valores na variável y_train_pred.

# Obter as predições e probabilidades do classificador "C"
y_proba = pipeline_model.predict_proba(X_global_train)[:, 1] 
#Ao usar [:, 1], você está extraindo a segunda coluna da matriz, 
# ou seja, as probabilidades relativas à classe 1 para cada exemplo.
#  Esses valores são então armazenados na variável y_proba.

# Aqui, inicialmente usamos um threshold fixo (por exemplo, 0.5)
default_threshold = 0.5  
y_pred = (y_proba >= default_threshold).astype(int)

# Agora, extraia os exemplos classificados como 1
mask = (y_pred == 1)
X_train_1 = X_global_train[mask]
y_train_1 = y_global_train[mask]

print(f"Número de exemplos classificados como 1 após ajuste: {mask.sum()}")


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
    
    for sampling_config in sampling_configs:
        sampling_name = sampling_config['sampling_name']
        # Se não houver iteração de parâmetros, usamos [None]
        for param_reg in model_config.get('parameter_range', [None]):
            print(f"\nRodando: Modelo={model_class_name}"
                  f"={param_reg} Sampling={sampling_name}")
            
            # Para relatório, definimos um identificador:
            # set_name = f"{model_class_name}&{model_config.get('tuning_parameter','')}&{param_reg}&{sampling_name}"
            set_name = f"{model_class_name}&{param_reg}&{sampling_name}"
            
            # Preparar os parâmetros do modelo de regressão
            model_params_R = model_config['params'].copy()

            
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
            plot_severity_metrics(severity_report, set_name )
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

# import os
# import numpy as np
# import pandas as pd
# import pickle
# import time
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import (
#     accuracy_score, classification_report, confusion_matrix,
#     mean_absolute_error, mean_squared_error, r2_score
# )
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import (
#     RandomForestRegressor, GradientBoostingClassifier
# )

# # =============================================
# # Funções de Visualização
# # =============================================

# def plot_classifier_metrics(results):
#     """Plota a relação entre threshold e F1-Score"""
#     thresholds = [res['threshold'] for res in results]
#     f1_scores = [res['Classification_Report']['1']['f1-score'] for res in results]
    
#     plt.figure(figsize=(10, 5))
#     plt.plot(thresholds, f1_scores, marker='o', linestyle='--', color='darkorange')
#     plt.title('Desempenho do Classificador por Threshold')
#     plt.xlabel('Threshold')
#     plt.ylabel('F1-Score (Classe 1)')
#     plt.grid(True)
#     plt.savefig('plots/classifier_threshold_analysis.png')
#     plt.close()

# def plot_confusion_matrix(cm, classes, model_name):
#     """Plota matriz de confusão"""
#     plt.figure(figsize=(6, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#                 xticklabels=classes, 
#                 yticklabels=classes)
#     plt.title(f'Matriz de Confusão - {model_name}')
#     plt.ylabel('Verdadeiro')
#     plt.xlabel('Previsto')
#     plt.savefig(f'plots/confusion_matrix_{model_name}.png')
#     plt.close()

# def plot_regression_performance(y_true, y_pred, model_name):
#     """Plota gráficos de avaliação de regressão"""
#     plt.figure(figsize=(12, 5))
    
#     # Gráfico 1: Valores Reais vs. Previstos
#     plt.subplot(1, 2, 1)
#     sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, color='royalblue')
#     plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
#              '--', color='tomato', lw=2)
#     plt.title(f'Real vs. Previsto\n{model_name}')
#     plt.xlabel('Valor Real')
#     plt.ylabel('Valor Previsto')
    
#     # Gráfico 2: Distribuição dos Resíduos
#     plt.subplot(1, 2, 2)
#     residuals = y_true - y_pred
#     sns.histplot(residuals, kde=True, color='seagreen')
#     plt.title(f'Distribuição dos Resíduos\n{model_name}')
#     plt.xlabel('Resíduo (Real - Previsto)')
    
#     plt.tight_layout()
#     plt.savefig(f'plots/regression_performance_{model_name}.png')
#     plt.close()



# def plot_feature_importance(model, feature_names, model_name):
#     """Plota importância das features para modelos que suportam"""
#     try:
#         if hasattr(model, 'feature_importances_'):
#             importances = model.feature_importances_
#         elif hasattr(model, 'coef_'):
#             importances = model.coef_
#         else:
#             return
            
#         indices = np.argsort(importances)[::-1]
#         plt.figure(figsize=(10, 6))
#         plt.barh(range(len(indices)), importances[indices], align='center', color='darkcyan')
#         plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
#         plt.gca().invert_yaxis()
#         plt.title(f'Importância das Features - {model_name}')
#         plt.xlabel('Importância Relativa')
#         plt.tight_layout()
#         plt.savefig(f'plots/feature_importance_{model_name}.png')
#         plt.close()
#     except Exception as e:
#         print(f"Não foi possível plotar importância de features para {model_name}: {str(e)}")

# # =============================================
# # Funções Auxiliares
# # =============================================

# def report_severity_metrics(y_true, y_pred):
#     """Calcula métricas por nível de severidade da precipitação"""
#     y_true = np.asarray(y_true).ravel()
#     y_pred = np.asarray(y_pred).ravel()
    
#     severity_levels = {
#         "sem_chuva_leve": (0, 5),
#         "moderada": (5, 25),
#         "forte": (25, 50),
#         "tempestade": (50, np.inf)
#     }
    
#     report = {}
#     for level, (low, high) in severity_levels.items():
#         mask = (y_true >= low) & (y_true < high)
#         count = np.sum(mask)
#         if count > 0:
#             report[level] = {
#                 "count": int(count),
#                 "MAE": mean_absolute_error(y_true[mask], y_pred[mask]),
#                 "MSE": mean_squared_error(y_true[mask], y_pred[mask]),
#                 "RMSE": np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])),
#                 "R2": r2_score(y_true[mask], y_pred[mask])
#             }
#         else:
#             report[level] = {
#                 "count": 0,
#                 "MAE": None,
#                 "MSE": None,
#                 "RMSE": None,
#                 "R2": None
#             }
#     return report

# # =============================================
# # Fluxo Principal
# # =============================================

# if __name__ == "__main__":
#     # Configuração inicial
#     start_time = time.time()
#     os.makedirs('plots', exist_ok=True)
    
#     # Carregar dados
#     current_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'A652.pickle')
#     with open(current_dir, 'rb') as f:
#         X_train, y_train, X_val, y_val, X_test, y_test = pickle.load(f)
    
#     # Pré-processamento
#     y_train = np.where(y_train == 0, 0, 1)
#     y_val = np.where(y_val == 0, 0, 1)
#     y_test = np.where(y_test == 0, 0, 1)
    
#     feature_columns = X_train.columns.tolist()
#     preprocessor = ColumnTransformer([
#         ('scaler', StandardScaler(), feature_columns)
#     ])
    
#     # ==================================================================
#     # Parte 1: Treinamento do Classificador
#     # ==================================================================
#     print("\n" + "="*50)
#     print("Treinamento do Classificador")
#     print("="*50)
    
#     model_classifier = GradientBoostingClassifier()
#     pipeline_classifier = Pipeline([
#         ('preprocessor', preprocessor),
#         ('classifier', model_classifier)
#     ])
    
#     # Treinar e encontrar melhor threshold
#     pipeline_classifier.fit(X_train, y_train)
#     y_proba = pipeline_classifier.predict_proba(X_val)[:, 1]
    
#     results = []
#     for threshold in np.arange(0.1, 1.0, 0.05):
#         y_pred = (y_proba >= threshold).astype(int)
#         cm = confusion_matrix(y_val, y_pred)
#         results.append({
#             "threshold": threshold,
#             "Classification_Report": classification_report(y_val, y_pred, output_dict=True),
#             "Confusion_Matrix": cm.tolist()
#         })
    
#     # Plotar análise do classificador
#     plot_classifier_metrics(results)
#     best_result = max(results, key=lambda x: x["Classification_Report"]["1"]["f1-score"])
#     plot_confusion_matrix(
#         np.array(best_result["Confusion_Matrix"]), 
#         ['0', '1'], 
#         'GradientBoostingClassifier'
#     )
#     plot_feature_importance(
#         pipeline_classifier.named_steps['classifier'],
#         feature_columns,
#         'GradientBoostingClassifier'
#     )
    
#     # ==================================================================
#     # Parte 2: Modelo Híbrido (Classificador + Regressores)
#     # ==================================================================
#     print("\n" + "="*50)
#     print("Treinamento dos Modelos Híbridos")
#     print("="*50)
    
#     # Configurações dos regressores
#     regression_models = [
#         {'name': 'LinearRegression', 'model': LinearRegression()},
#         {'name': 'DecisionTreeRegressor', 'model': DecisionTreeRegressor(random_state=42)},
#         {'name': 'RandomForestRegressor', 'model': RandomForestRegressor(n_estimators=100, random_state=42)}
#     ]
    
#     final_results = []
#     for reg_config in regression_models:
#         model_name = reg_config['name']
#         print(f"\nProcessando: {model_name}")
        
#         # Treinar regressor
#         pipeline_regressor = Pipeline([
#             ('preprocessor', preprocessor),
#             ('regressor', reg_config['model'])
#         ])
#         pipeline_regressor.fit(X_train, y_train)
        
#         # Fazer previsões híbridas
#         y_pred_class = pipeline_classifier.predict(X_test)
#         y_pred_reg = pipeline_regressor.predict(X_test)
#         y_pred_final = np.where(y_pred_class == 0, 0, y_pred_reg)
        
#         # Calcular métricas
#         metrics = {
#             "MAE": mean_absolute_error(y_test, y_pred_final),
#             "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_final)),
#             "R2": r2_score(y_test, y_pred_final)
#         }
#         severity_report = report_severity_metrics(y_test, y_pred_final)
        
#         # Armazenar resultados
#         final_results.append({
#             "Model": model_name,
#             **metrics,
#             "Severity_Report": severity_report
#         })
        
#         # Gerar visualizações
#         plot_regression_performance(y_test, y_pred_final, model_name)
#         plot_severity_metrics(severity_report, model_name)
#         plot_feature_importance(
#             pipeline_regressor.named_steps['regressor'],
#             feature_columns,
#             model_name
#         )
    
#     # Resultados finais
#     df_results = pd.DataFrame(final_results)
#     print("\nResultados Finais:")
#     print(df_results[['Model', 'MAE', 'RMSE', 'R2']])
    
#     # Tempo total
#     print(f"\nTempo Total de Execução: {time.time() - start_time:.2f} segundos")