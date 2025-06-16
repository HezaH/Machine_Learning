import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error, 
                             median_absolute_error, mean_squared_log_error, explained_variance_score)
from mapie.regression import MapieRegressor, MapieQuantileRegressor
from sklearn.base         import clone 
from mapie.metrics import regression_coverage_score
from mapie.metrics import regression_mean_width_score

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

target_column = 'price'

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
##############################################################################
# BLOCO AUTÔNOMO – CONFORMAL PREDICTION COM:
#   1) MapieRegressor (“split-conformal plus”)
#   2) MapieQuantileRegressor (CQR)
# Corrige: • strings “Premium” passam pelo pré-processador
#          • cv="split" (nada de X_calib, y_calib no .fit)
##############################################################################
from sklearn.compose    import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline   import Pipeline
from sklearn.ensemble   import GradientBoostingRegressor
from mapie.regression   import MapieRegressor, MapieQuantileRegressor
import numpy as np, pandas as pd, matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 0. PRÉ-PROCESSADOR  (numéricas + categóricas ordenadas)
# ─────────────────────────────────────────────────────────────────────────────
preprocessor = ColumnTransformer([
    ("num", StandardScaler(),  numeric_preproc),
    ("cat", OrdinalEncoder(categories=[cut_order,
                                       color_order,
                                       clarity_order]), categorical_preproc)
])

# helper: constrói um pipeline GBR-quantile para qualquer α
def make_quantile_pipe(alpha):
    gbr = GradientBoostingRegressor(
        loss="quantile", alpha=alpha,
        n_estimators=500, max_depth=3,
        learning_rate=0.05, random_state=42
    )
    return Pipeline([("prep", preprocessor), ("gbr", gbr)])

##############################################################################
# 1. SPLIT-CONFORMAL  (MapieRegressor, método "plus")
##############################################################################
split_pipe = make_quantile_pipe(alpha=0.5)        # mediana como base
split_cp   = MapieRegressor(
                 estimator = split_pipe,
                 method    = "plus",
                 cv        = "split")             # split interno
split_cp.fit(X_train, y_train)

alpha_int     = 0.10                              # 90 % intervalo
y_hat_split,  int_split = split_cp.predict(X_test, alpha=[alpha_int])

if int_split.shape[2] == 1:          # única-caudal
    lower_s = int_split[:, 0, 0]
    upper_s = np.nan * lower_s       # ou recompute o outro lado
else:                                # bi-caudal
    lower_s = int_split[:, 0, 0]
    upper_s = int_split[:, 0, 1]
##############################################################################
# 2. CQR  (MapieQuantileRegressor) – cv="split"
##############################################################################
cqr_pipe = make_quantile_pipe(alpha=alpha_int)    # quantil 0.10 ou 0.90
cqr = MapieQuantileRegressor(
        estimator = cqr_pipe,
        alpha     = alpha_int,
        cv        = "split")
cqr.fit(X_train, y_train)
_, int_cqr = cqr.predict(X_test)                  # (n, 2)
lower_c, upper_c = int_cqr[:, 0], int_cqr[:, 1]

##############################################################################
# 3. FUNÇÃO PICP / MPIW  (cobertura e largura)
##############################################################################
def picp_mpiw(y_true, low, up):
    """
    Parameters
    ----------
    y_true : array-like, shape (n,)
    low    : array-like, shape (n,)
    up     : array-like, shape (n,)

    Returns
    -------
    PICP : float   – Percentual de pontos cobertos
    MPIW : float   – Largura média do intervalo
    """
    y_true = np.asarray(y_true).ravel()
    low    = np.asarray(low).ravel()
    up     = np.asarray(up).ravel()

    if not (len(y_true) == len(low) == len(up)):
        raise ValueError("y_true, low, up devem ter o mesmo comprimento.")

    inside = (y_true >= low) & (y_true <= up)
    return inside.mean(), np.mean(up - low)

picp_s, mpiw_s = picp_mpiw(y_test, lower_s, upper_s)
picp_c, mpiw_c = picp_mpiw(y_test, lower_c, upper_c)

print(f"Split-Conformal 90 % → PICP={picp_s:.3f}  MPIW={mpiw_s:8.1f}")
print(f"CQR             90 % → PICP={picp_c:.3f}  MPIW={mpiw_c:8.1f}")

##############################################################################
# 4. MAIOR e MENOR INTERVALO (Split-Conformal 90 %)
##############################################################################
width = upper_s - lower_s
idx_max, idx_min = width.argmax(), width.argmin()

def show_case(idx, tag):
    row = X_test.iloc[idx].copy()
    row["y_real"] = y_test.iloc[idx]
    row["y_pred"] = y_hat_split[idx]
    row["lower"]  = lower_s[idx]
    row["upper"]  = upper_s[idx]
    row["width"]  = width[idx]
    print(f"\n── {tag} intervalo (idx={idx}) ──────────────────────────")
    print(row)

show_case(idx_max, "MAIOR")
show_case(idx_min, "MENOR")

##############################################################################
# 5. GRÁFICO – comparação visual dos dois métodos
##############################################################################
order = np.argsort(y_test.values)
plt.figure(figsize=(10,5))
plt.plot(y_test.values[order], ".", color="black", label="Preço real")
plt.fill_between(np.arange(len(order)), lower_s[order], upper_s[order],
                 color="steelblue", alpha=.25, label="Split 90 %")
plt.fill_between(np.arange(len(order)), lower_c[order], upper_c[order],
                 color="crimson", alpha=.20, label="CQR  90 %")
plt.ylabel("Preço (USD)")
plt.xlabel("Exemplos de teste (ordenados)")
plt.title("Intervalos 90 % – Split-Conformal vs CQR")
plt.legend(); plt.tight_layout(); plt.show()
