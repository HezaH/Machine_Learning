import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error, 
                             median_absolute_error, mean_squared_log_error, explained_variance_score)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Inicia a medição do tempo
start_time = time.time()

# Carrega o dataset
try:
    df_diamonds = pd.read_csv('diamonds.csv')
except FileNotFoundError:
    current_dir = os.path.dirname(os.path.realpath(__file__))
    df_diamonds = pd.read_csv(os.path.join(current_dir, 'diamonds.csv'))

# Para treinar um modelo que prediz z, selecione apenas os registros com z > 0 (valores válidos)
df_diamonds = df_diamonds[~df_diamonds.index.isin(df_diamonds[df_diamonds[['x', 'y', 'z']].eq(0).sum(axis=1) >= 2].index)].copy()

df_valid = df_diamonds[df_diamonds['z'] > 0].copy()
df_invalid = df_diamonds[df_diamonds['z'] == 0].copy()

# Defina as features originais que queremos usar. Atenção: não use features derivadas que contenham z (como volume).
# Neste exemplo, utilizaremos as medidas originais e informações categóricas.
features = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'price', 'x', 'y']
target = 'z'

X = df_valid[features]
y = df_valid[target]

# Divide os dados em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Defina os nomes das colunas numéricas e categóricas
numeric_cols = ['carat', 'depth', 'table', 'price', 'x', 'y']
categorical_preproc = ['cut', 'color', 'clarity']

# Cria o pré-processador: escalona os numéricos e codifica as categorias (usando OneHotEncoder para evitar assumir ordens pré-definidas)
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(), categorical_preproc )
])

# Instancia o modelo. Aqui usamos regressão linear simples; você pode testar outros modelos, como RandomForestRegressor.
model = LinearRegression()

# Monta o pipeline de pré-processamento e modelo
pipeline_model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

# Treina o pipeline
pipeline_model.fit(X_train, y_train)

# Faz a predição no conjunto de teste e avalia o desempenho
y_pred = pipeline_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE para predição de z: {rmse:.4f}")

# Opcional: visualize a relação entre os valores reais e preditos
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Valores Reais de z")
plt.ylabel("Valores Preditos de z")
plt.title("Comparação entre Valores Reais e Preditos")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.savefig("compare_z.png", dpi=300, bbox_inches='tight')
plt.close()  # Fecha o gráfico atual

# Se houver registros com z igual a 0, prevê os valores de z para esses registros.
if not df_invalid.empty:
    X_invalid = df_invalid[features]
    z_pred = pipeline_model.predict(X_invalid)
    
    # Atualiza a coluna z dos registros com z = 0 no DataFrame original
    df_diamonds.loc[df_diamonds['z'] == 0, 'z'] = abs(z_pred)
    print("Valores de z imputados para registros com z == 0.")

# Visualiza as primeiras linhas do DataFrame atualizado
print(df_diamonds.head())

# Engenharia de features: Adiciona volume e outras derivadas
df_diamonds['volume'] = df_diamonds['x'] * df_diamonds['y'] * df_diamonds['z']
df_diamonds["price_per_carat"] = df_diamonds["price"] / df_diamonds["carat"]
df_diamonds["price_per_volume"] = df_diamonds["price"] / df_diamonds["volume"]

df_diamonds["ratio_x_y"] = df_diamonds["x"] / df_diamonds["y"]
df_diamonds["ratio_x_z"] = df_diamonds["x"] / df_diamonds["z"]
df_diamonds["ratio_y_z"] = df_diamonds["y"] / df_diamonds["z"]

df_diamonds["log_price"] = np.log(df_diamonds["price"])

# Remove a coluna indesejada (se existir) e reseta o índice
if 'Unnamed: 0' in df_diamonds.columns:
    df_diamonds = df_diamonds.drop(columns='Unnamed: 0').reset_index(drop=True)

# Gráfico 1: Relação entre Volume e Preço
fig, ax = plt.subplots(figsize=(10, 8))
sns.regplot(x='volume', y='price', data=df_diamonds, scatter_kws={'alpha': 0.4}, ax=ax)
ax.set_title("Relação entre Volume e Preço do Diamante")
ax.set_xlabel("Volume (x * y * z)")
ax.set_ylabel("Preço")

# Cria um inset (subgráfico) com zoom na região desejada
axins = inset_axes(ax, width="40%", height="40%", loc='lower right',
                   bbox_to_anchor=(0, 0.2, 1, 1),
                   bbox_transform=ax.transAxes)
sns.regplot(x='volume', y='price', data=df_diamonds, scatter_kws={'alpha': 0.4}, ax=axins)
axins.set_xlim(0, 800)         # Limita o eixo x do inset
axins.set_ylim(0, 25000)       # Limita o eixo y do inset
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.4")

plt.savefig("value_price.png", dpi=300, bbox_inches='tight')
plt.close()  # Fecha o gráfico atual

# Gráfico 2: Heatmap de Correlação entre as features numéricas
cols = list(df_diamonds.columns)
# Definir as features categóricas e a variável alvo

target_column = 'price'
# Para o heatmap, iremos usar apenas as colunas numéricas (excluindo as categóricas e a target)
numeric_features = [col for col in cols if col not in categorical_preproc + [target_column]]
corr_matrix = df_diamonds[numeric_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Heatmap de Correlação entre Variáveis")
plt.savefig("Heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

# Separando as variáveis preditoras e a variável alvo para modelagem
X = df_diamonds.drop(target_column, axis=1)
y = df_diamonds[target_column]

# Divide os dados em treino (60%), teste (20%) e calibração (20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Cria o pré-processador com codificação ordinal
# Define as ordens para as variáveis categóricas
cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_order = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

# Como X não possui a coluna target, definimos as features numéricas para o pré-processador:
numeric_preproc = [col for col in X.columns if col not in categorical_preproc]

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_preproc),
    ('cat', OrdinalEncoder(categories=[cut_order, color_order, clarity_order]), categorical_preproc)
])

# Instancia o modelo KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=5)

# Monta o pipeline (pré-processamento + modelo)
pipeline_model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

# Treina o pipeline usando os dados de treino
pipeline_model.fit(X_train, y_train)

# Faz a predição no conjunto de teste
y_calib_pred = pipeline_model.predict(X_calib)
residuals = np.abs(y_calib - y_calib_pred)

for alpha in [0.15, 0.1, 0.05, 0.01]:
    # 4. Define quantile threshold for 90% prediction interval
    q = np.quantile(residuals, 1 - alpha)
    
    # 5. Predict on test set with prediction intervals
    y_test_pred = pipeline_model.predict(X_test)
    y_lower = y_test_pred - q
    y_upper = y_test_pred + q

    # Calcula as métricas de avaliação
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2 = r2_score(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    median_ae = median_absolute_error(y_test, y_test_pred)
    msle = mean_squared_log_error(y_test, y_test_pred)
    evs = explained_variance_score(y_test, y_test_pred)
    n = len(y_test)
    p = X_test.shape[1]
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

    print("KNeighborsRegressor (n_neighbors=5)")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Median Absolute Error: {median_ae:.4f}")
    print(f"MSLE: {msle:.4f}")
    print(f"Explained Variance Score: {evs:.4f}")
    print(f"Adjusted R²: {adjusted_r2:.4f}")
    print(f"MAPE: {mape:.2f}%")

    # 6. Plot results
    preprocessor = pipeline_model.named_steps["preprocessor"]

    # Obtenha os nomes das features após a transformação
    feature_names = preprocessor.get_feature_names_out()

    # Transforme os dados de X_test com o pré-processador
    X_transformed = preprocessor.transform(X_test)

    # Converta o resultado em um DataFrame para facilitar a plotagem
    df_features = pd.DataFrame(X_transformed, columns=feature_names)
    # Agora, para cada coluna, crie um gráfico (por exemplo, um histograma)
    for col in df_features.columns:
        plt.figure(figsize=(8, 6))
        plt.scatter(df_features[col], y_test, color='blue', label='True')
        plt.plot(df_features[col], y_test_pred, color='black', label='Prediction')
        plt.fill_between(df_features[col], y_lower, y_upper, color='orange', alpha=0.3, label='90% Prediction Interval')
        plt.title("Conformal Prediction Interval (Regression)")
        plt.xlabel("Índice do exemplo")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"figures/conformal_prediction_{alpha}_{col}.png", dpi=300, bbox_inches='tight')
        plt.close()

# Finaliza a medição do tempo
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTempo total de processamento: {elapsed_time:.2f} segundos")



