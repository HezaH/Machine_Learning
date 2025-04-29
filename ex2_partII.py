import pandas as pd
import numpy as np
from math import sqrt
import os, time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Importação dos modelos candidatos (apenas os que trabalham com regressão)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Função de avaliação do modelo de regressão
def model_evaluation_regression(model_name, y_test, y_pred):
    """
    Avalia o desempenho de um modelo de regressão:
      - Exibe um gráfico com: 
            (1) Previsões vs. Valores reais
            (2) Gráfico dos resíduos
      - Exibe num texto as métricas MSE e R²
    """
    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    residuals = y_test - y_pred

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Predicted vs Actual
    axs[0].scatter(y_test, y_pred, color='blue', alpha=0.6)
    axs[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
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
    
    metrics_text = f"MSE: {mse:.2f}\nR²: {r2:.2f}"
    plt.figtext(0.5, 0.01, metrics_text, wrap=True, horizontalalignment='center', fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f'{model_name}.png')
    #plt.show()


# Inicia a medição do tempo
start_time = time.time()

current_dir = os.path.dirname(os.path.realpath(__file__))
df_diamonds = pd.read_csv(os.path.join(current_dir, 'diamonds.csv'))

# Visualiza as categorias das colunas
columns = ['cut', 'color', 'clarity']
dict_unique = {c: df_diamonds[c].unique() for c in columns}
print("Categorias nas colunas:", dict_unique)

# Define o target 'price' e as features
target_column = 'price'
X = df_diamonds.drop(target_column, axis=1)
y = df_diamonds[target_column]

# Divide os dados em treino (80%) e teste (20%)
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define as colunas numéricas e categóricas
numeric_features = ['carat', 'depth', 'table', 'x', 'y', 'z']
categorical_features = ['cut', 'color', 'clarity']

# Cria o pré-processador: StandardScaler para numéricas e OneHotEncoder para categóricas.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Escolha dois modelos de aprendizado para regressão e defina os espaços de hiperparâmetros.
# Note que os hiperparâmetros do modelo são referenciados como "model__param" (pois estão dentro do pipeline).
model_configurations = [
    {
        'class_name': "KNeighborsRegressor",
        'model_class': KNeighborsRegressor,
        'param_grid': { 'model__n_neighbors': [3, 5, 7, 9] }
    },
    {
        'class_name': "GradientBoostingRegressor",
        'model_class': GradientBoostingRegressor,
        'param_grid': { 
            'model__n_estimators': [50, 100, 150],
            'model__learning_rate': [0.01, 0.1, 0.2]
        },
        'random_state': 42
    }
]

# Configuração dos CV: outer CV com 5 folds e inner CV com 5 folds
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

nested_results = []

# Para cada candidato (modelo e seu espaço de hiperparâmetros)
for config in model_configurations:
    model_name = config['class_name']
    print(f"\nProcessando candidato: {model_name}")
    
    # Se o modelo requer random_state, passa-o; caso contrário, instância sem parâmetro
    if 'random_state' in config:
        base_model = config['model_class'](random_state=config['random_state'])
    else:
        base_model = config['model_class']()
    
    # Cria o pipeline que integra o pré-processamento e o modelo
    pipeline_candidate = Pipeline([
        ('preprocessor', preprocessor),
        ('model', base_model)
    ])
    
    # Configura a GridSearch para o inner CV
    grid_search = GridSearchCV(estimator=pipeline_candidate,
                               param_grid=config['param_grid'],
                               cv=inner_cv,
                               scoring='neg_mean_squared_error',
                               n_jobs=-1)
    
    # Utiliza cross_val_score com o grid_search como estimador para realizar a outer CV.
    scores = cross_val_score(grid_search, X_train_full, y_train_full,
                             cv=outer_cv, scoring='neg_mean_squared_error', n_jobs=-1)
    avg_rmse = sqrt(-np.mean(scores))
    print(f"Média RMSE (Nested CV) para {model_name}: {avg_rmse:.4f}")
    
    # Ajusta o grid_search em todo o conjunto de treino para determinar os melhores hiperparâmetros.
    grid_search.fit(X_train_full, y_train_full)
    best_params = grid_search.best_params_
    best_inner_rmse = sqrt(-grid_search.best_score_)
    
    nested_results.append({
        'model': model_name,
        'avg_rmse_nested': avg_rmse,
        'best_params': best_params,
        'best_inner_rmse': best_inner_rmse,
        'best_estimator': grid_search.best_estimator_
    })

# Converte os resultados da nested CV para um DataFrame e exibe
df_nested = pd.DataFrame(nested_results)
print("\nResultados da Nested Cross-Validation:")
print(df_nested)

# Seleciona o melhor candidato com base no menor RMSE médio na outer CV
best_candidate = df_nested.loc[df_nested['avg_rmse_nested'].idxmin()]
print("\nMelhor modelo selecionado via Nested CV:")
print(f"Modelo: {best_candidate['model']}")
print(f"RMSE médio (Nested): {best_candidate['avg_rmse_nested']:.4f}")
print(f"Melhor hiperparâmetro: {best_candidate['best_params']}")

# Agora, com o melhor estimador (re-ajustado no conjunto de treino completo), obtenha as predições no conjunto de teste
best_model_pipeline = best_candidate['best_estimator']
y_pred_test = best_model_pipeline.predict(X_test)
test_rmse = sqrt(mean_squared_error(y_test, y_pred_test))
test_r2 = r2_score(y_test, y_pred_test)

print("\nAvaliação no conjunto de teste:")
print(f"RMSE: {test_rmse:.4f}")
print(f"R²: {test_r2:.4f}")

# Gera os plots de avaliação (Predicted vs Actual, Residual Plot e métricas) para o modelo final
model_final_name = best_candidate['model'] + "_final"
model_evaluation_regression(model_final_name, y_test, y_pred_test)

# Análise dos resultados: podemos plotar um scatter com a média de RMSE dos candidatos (obtidos na nested CV)
plt.figure(figsize=(8, 6))
plt.scatter(df_nested['avg_rmse_nested'], df_nested['best_inner_rmse'], 
            s=150, c='blue', alpha=0.7)
for i, row in df_nested.iterrows():
    plt.text(row['avg_rmse_nested']+0.01, row['best_inner_rmse']+0.01, 
             f"{row['model']}\n{row['best_params']}", fontsize=9)
plt.xlabel("RMSE Médio Outer CV (quanto menor melhor)")
plt.ylabel("RMSE Melhor Inner CV (quanto menor melhor)")
plt.title("Comparação dos Candidatos via Nested CV")
plt.grid(True)
plt.savefig('nested_cv_comparison.png')
#plt.show()

# Tempo total de processamento
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTempo total de processamento: {elapsed_time:.2f} segundos")
