import pandas as pd
import numpy as np
from math import sqrt
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Importação dos modelos
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
import time
import matplotlib.pyplot as plt

def model_evaluation_regression(model_name, y_test, y_pred):
    """
    Avalia o desempenho de um modelo de regressão:
      - Exibe um gráfico com: 
            (1) Previsões vs. Valores reais
            (2) Gráfico dos resíduos
      - Exibe num texto as métricas MSE e R²

    Parameters:
      y_test : array-like, valores reais
      y_pred : array-like, valores previstos pelo modelo    
    """
    # Calcula as métricas
    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    residuals = y_test - y_pred

    # Cria um figure com 2 subplots lado a lado
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Predicted vs Actual
    axs[0].scatter(y_test, y_pred, color='blue', alpha=0.6)
    axs[0].plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()],
                'r--', lw=2)
    axs[0].set_xlabel("Actual")
    axs[0].set_ylabel("Predicted")
    axs[0].set_title("Predicted vs Actual")
    axs[0].grid(True)

    # 2. Residuals Plot
    axs[1].scatter(y_pred, residuals, color='green', alpha=0.6)
    axs[1].axhline(0, color='red', linestyle='--', lw=2)
    axs[1].set_xlabel("Predicted")
    axs[1].set_ylabel("Residuals")
    axs[1].set_title("Residual Plot")
    axs[1].grid(True)

    # Adiciona as métricas no figure (por exemplo, na parte inferior central)
    metrics_text = f"MSE: {mse:.2f}\nR²: {r2:.2f}"
    plt.figtext(0.5, 0.01, metrics_text, wrap=True, horizontalalignment='center', fontsize=12)

    # Ajusta o layout para não sobrepor a área das métricas
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f'{model_name}.png')

# Inicia a medição do tempo
start_time = time.time()

try:
    df_diamonds = pd.read_csv('diamonds.csv')
except FileNotFoundError:
    current_dir = os.path.dirname(os.path.realpath(__file__))
    df_diamonds = pd.read_csv(os.path.join(current_dir,'diamonds.csv'))

df_diamonds.head()

# Engenharia de features: Adiciona volume
df_diamonds['volume'] = df_diamonds['x'] * df_diamonds['y'] * df_diamonds['z']

columns = ['cut', 'color', 'clarity']
dict_unique = {}
for c in columns:
    dict_unique[c] = df_diamonds[c].unique()
print(dict_unique)

target_column = 'price'
X = df_diamonds.drop(target_column, axis=1)
y = df_diamonds[target_column]

# Divide os dados em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""Cria o pré-processador com codificação ordinal"""
numeric_features = ['carat', 'depth', 'table', 'x', 'y', 'z', 'volume']
categorical_features = ['cut', 'color', 'clarity']
# Define ordens das categorias
cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_order = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OrdinalEncoder(categories=[cut_order, color_order, clarity_order]), 
            categorical_features)
    ])

# Define model configurations; include tuning keys for models that require parameter tuning.
model_configurations = [
    {
        'class_name': "KNeighborsRegressor",
        'model_class': KNeighborsRegressor,
        'tuning_parameter': 'n_neighbors',
        'parameter_range': [5]
        # 'parameter_range': list(range(5, 10, 5))
    },
    {
        'class_name': "LogisticRegression",
        'model_class': LogisticRegression
    },
    {
        'class_name': "GradientBoostingRegressor",
        'model_class': GradientBoostingRegressor,
        'tuning_parameter': 'n_estimators',
        'parameter_range': [50],
        # 'parameter_range': list(range(50, 100, 50)),
        'random_state': 42
    }
]
# Define model configurations; include tuning keys for models that require parameter tuning.
model_configurations = [
    {
        'class_name': "KNeighborsRegressor",
        'model_class': KNeighborsRegressor,
        'tuning_parameter': 'n_neighbors',
        'parameter_range': [5]
        # 'parameter_range': list(range(5, 10, 5))
    },
    {
        'class_name': "LogisticRegression",
        'model_class': LogisticRegression
    },
    {
        'class_name': "GradientBoostingRegressor",
        'model_class': GradientBoostingRegressor,
        'tuning_parameter': 'n_estimators',
        'parameter_range': [50],
        # 'parameter_range': list(range(50, 100, 50)),
        'random_state': 42
    }
]

results = []

# Loop nos modelos definidos
for config in model_configurations:
    model_name = config['class_name']
    print(f"Processando modelo: {model_name}")
    
    # Verifica se há parâmetro de tuning na configuração do modelo
    if 'tuning_parameter' in config:
        tuning_param = config['tuning_parameter']
        param_range = config.get('parameter_range', [])
    else:
        tuning_param = None
        param_range = [None]
    
    # Para cada valor no range do parâmetro (se aplicável)
    for param_value in param_range:
        # Se houver parâmetro, instancie-o; caso contrário, use o default
        if tuning_param is not None and param_value is not None:
            # Se houver argumentos adicionais (e.g., random_state), inclua-os
            model_kwargs = {tuning_param: param_value}
            if 'random_state' in config:
                model_kwargs['random_state'] = config['random_state']
            model_instance = config['model_class'](**model_kwargs)
        else:
            model_instance = config['model_class']()
        
        # Cria o pipeline com o pré-processamento e o modelo
        pipeline_model = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model_instance)
        ])
        
        # Treina o pipeline usando os dados de treino
        pipeline_model.fit(X_train, y_train)
        
        # Faz a predição no conjunto de teste
        y_pred = pipeline_model.predict(X_test)
        
        # Calcula as métricas: RMSE e R²
        rmse = sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        result = {
            'model': model_name,
            'tuning_parameter': tuning_param,
            'param_value': param_value,
            'rmse': rmse,
            'r2': r2
        }
        results.append(result)
        model_final = model_name + "_" + str(tuning_param) + "_" + str(param_value)
        print(f"Modelo: {model_name}, {tuning_param}: {param_value}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        model_evaluation_regression(model_final, y_test, y_pred)

# Após executar o loop que preenche a variável 'results',
# convertemos os resultados para um DataFrame:
df_results = pd.DataFrame(results)
print(df_results)

# Definição do melhor modelo:
# Uma estratégia simples é selecionar o modelo que tenha o menor RMSE.
# (Alternativamente, você pode criar um critério composto que combine RMSE e R².)
best_model = df_results.loc[df_results['rmse'].idxmin()]
print("------------------------------------------------------")
print("Melhor modelo identificado:")
calib_str = (f"Calibração: {best_model['tuning_parameter']} = {best_model['param_value']}" 
             if best_model['tuning_parameter'] is not None else "Sem parâmetro de tuning")
print(f"Modelo: {best_model['model']} / {calib_str}")
print(f"RMSE: {best_model['rmse']:.4f}")
print(f"R²: {best_model['r2']:.4f}")

# --- Plot para auxiliar na seleção do melhor modelo ---
# Por exemplo, podemos criar um scatter plot onde o eixo X é o RMSE e o eixo Y é o R² para cada configuração,
# e destacar o melhor modelo.

plt.figure(figsize=(8, 6))
plt.scatter(df_results['rmse'], df_results['r2'], s=100, c='blue', alpha=0.7)
for i, row in df_results.iterrows():
    label = f"{row['model']}\n{row['tuning_parameter']}={row['param_value']}"
    plt.text(row['rmse']+0.01, row['r2']+0.005, label, fontsize=9)

# Destaque (por exemplo, com um círculo em vermelho) o melhor modelo
plt.scatter(best_model['rmse'], best_model['r2'], s=150, c='red', marker='*', label='Melhor Modelo')

plt.xlabel("RMSE (quanto menor melhor)")
plt.ylabel("R² (quanto maior melhor)")
plt.title("Comparação dos Modelos")
plt.legend()
plt.grid(True)
plt.savefig(f'best_model.png')

# Finaliza a medição do tempo
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTempo total de processamento: {elapsed_time:.2f} segundos")


