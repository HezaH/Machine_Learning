import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             median_absolute_error, mean_squared_log_error, explained_variance_score)

# Inicia a medição do tempo
start_time = time.time()
out_dir = os.path.dirname(os.path.realpath(__file__))
# Carrega o dataset
try:
    df_diamonds = pd.read_csv('diamonds.csv')
except FileNotFoundError:
    current_dir = out_dir
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
plt.savefig(os.path.join(out_dir, "figures", "Heatmap.png"), dpi=300, bbox_inches='tight')
plt.close()

# Gráfico 1: Relação entre Volume e Preço
fig, ax = plt.subplots(figsize=(10, 8))
sns.scatterplot(x=y_test, y=y_pred)
ax.set_title("Comparação entre Valores Reais e Preditos")
ax.set_xlabel("Valores Reais de z")
ax.set_ylabel("Valores Preditos de z")
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Linha de Igualdade')
# Cria um inset (subgráfico) com zoom na região desejada
axins = inset_axes(ax, width="40%", height="40%", loc='lower right',
                   bbox_to_anchor=(0, 0.2, 1, 1),
                   bbox_transform=ax.transAxes)

sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha': 0.4}, ax=axins)
axins.set_xlim(min(y_test.min(), y_pred.min()) - 1, min(y_test.max(), y_pred.max()) + 1)         # Limita o eixo x do inset
axins.set_ylim(min(y_test.min(), y_pred.min()) - 1, min(y_test.max(), y_pred.max()) + 1)       # Limita o eixo y do inset
axins.set_title("Zoom na Região de Interesse")
axins.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Linha de Igualdade')
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.4")

plt.savefig(os.path.join(out_dir, "figures", "value_price.png"), dpi=300, bbox_inches='tight')
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

## Calcula quartis
Q1 = df_diamonds['volume'].quantile(0.25)  # Primeiro quartil (25%)
Q3 = df_diamonds['volume'].quantile(0.75)  # Terceiro quartil (75%)
IQR = Q3 - Q1  # Intervalo interquartil

# Define limites
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

# Filtra valores dentro do intervalo aceitável
df_diamonds = df_diamonds[(df_diamonds['volume'] >= limite_inferior) & (df_diamonds['volume'] <= limite_superior)]

df_diamonds["price_per_carat"] = df_diamonds["price"] / df_diamonds["carat"]
df_diamonds["price_per_volume"] = df_diamonds["price"] / df_diamonds["volume"]

df_diamonds["ratio_x_y"] = df_diamonds["x"] / df_diamonds["y"]
df_diamonds["ratio_x_z"] = df_diamonds["x"] / df_diamonds["z"]
df_diamonds["ratio_y_z"] = df_diamonds["y"] / df_diamonds["z"]

df_diamonds["log_price"] = np.log(df_diamonds["price"])

# Remove a coluna indesejada (se existir) e reseta o índice
if 'Unnamed: 0' in df_diamonds.columns:
    df_diamonds = df_diamonds.drop(columns='Unnamed: 0').reset_index(drop=True)

# Criando o gráfico interativo
fig = px.scatter(df_diamonds, x="volume", y="price", color="cut",
                 title="Relação entre Volume e Preço do Diamante")

# Exportando para HTML
fig.write_html(os.path.join(out_dir, "figures", "diamantes_interativo.html"))

# -----------------------------------------------------------
# 1. PREÇO  ×  PRICE-PER-CARAT   (cores = CUT)
# -----------------------------------------------------------
fig = px.scatter(
    df_diamonds,
    x="price_per_carat",
    y="price",
    color="cut",
    hover_data=["carat", "color", "clarity"],
    title="Preço vs. Preço por Quilate (CUT)",
    labels={
        "price_per_carat": "Preço / Quilate",
        "price": "price",
        "cut": "Cut"
    },
)
fig.update_traces(marker=dict(opacity=0.65, line=dict(width=0.3, color="white")))
fig.write_html(os.path.join(out_dir, "figures", "price_vs_price_per_carat_cut.html"), include_plotlyjs="cdn")

# -----------------------------------------------------------
# 2. PREÇO  ×  PRICE-PER-VOLUME  (cores = CLARITY)
# -----------------------------------------------------------
fig_volume = px.scatter(
    df_diamonds,
    x="price_per_volume",
    y="price",
    color="clarity",
    hover_data=["volume", "color", "cut"],
    title="Preço vs. Preço por Volume (CLARITY)",
    labels={
        "price_per_volume": "Preço / Volume",
        "price": "price",
        "clarity": "Clarity"
    },
)
fig_volume.update_traces(marker=dict(opacity=0.65, line=dict(width=0.3, color="white")))
fig_volume.write_html(os.path.join(out_dir, "figures", "price_vs_price_per_volume_clarity.html"), include_plotlyjs="cdn")

# -----------------------------------------------------------
# 3. EXTRA (opcional): RELAÇÃO ENTRE RÁCIOS DE LADOS E PREÇO
# -----------------------------------------------------------
fig_ratio = px.scatter(
    df_diamonds,
    x="ratio_x_y",
    y="price",
    color="color",
    hover_data=["ratio_x_z", "ratio_y_z", "cut", "clarity"],
    title="Preço vs. Proporção x/y (COLOR)",
    labels={
        "ratio_x_y": "x / y",
        "price": "price",
        "color": "Color"
    },
)
fig_ratio.update_traces(marker=dict(opacity=0.65, line=dict(width=0.3, color="white")))
fig_ratio.write_html(os.path.join(out_dir, "figures", "price_vs_ratio_xy_color.html"), include_plotlyjs="cdn")

# Separando as variáveis preditoras e a variável alvo para modelagem
X = df_diamonds.drop(target_column, axis=1)
y = df_diamonds[target_column]

# Divide os dados em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
y_pred = pipeline_model.predict(X_test)

# Calcula as métricas de avaliação
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
median_ae = median_absolute_error(y_test, y_pred)
msle = mean_squared_log_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
n = len(y_test)
p = X_test.shape[1]
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("KNeighborsRegressor (n_neighbors=5)")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"Median Absolute Error: {median_ae:.4f}")
print(f"MSLE: {msle:.4f}")
print(f"Explained Variance Score: {evs:.4f}")
print(f"Adjusted R²: {adjusted_r2:.4f}")
print(f"MAPE: {mape:.2f}%")

# Finaliza a medição do tempo
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTempo total de processamento: {elapsed_time:.2f} segundos")



