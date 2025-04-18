import os
import numpy as np
import pandas as pd
import math
import pickle

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss, ClusterCentroids, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek


from utils import (
    calculate_metrics, find_optimal_cost_threshold, cross_validate_models)


current_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'A652.pickle')

f = open( current_dir , 'rb')
( X_train , y_train , X_test , y_test , X_test , y_test ) = pickle.load(f) 

print(f"Shapes: {X_train.shape}, {X_test.shape}, {X_test.shape}")

# Transformar os valores contínuos em rótulos binários
y_train = np.where(y_train == 0, 0, 1)
y_test   = np.where(y_test == 0, 0, 1)
y_test  = np.where(y_test == 0, 0, 1)

# Define model configurations; include tuning keys for models that require parameter tuning.
model_configurations = [
    {
        'class_name': "GradientBoostingClassifier",
        'model_class': GradientBoostingClassifier,
        'tuning_parameter': 'learning_rate',
        # "learning_rate": 0.1,   # Taxa de aprendizado
        # "max_depth": 3,         # Profundidade das árvores
        "random_state":42,
        'parameter_range': [0.1, 0.01, 0.001],  # Valores de taxa de aprendizado a serem testados
        # 'parameter_range': list(range(5, 10, 5))
        }
]

# Define scaler configurations
scaler_configs = [
    {"scaler_name": "MinMaxScaler", 'scaler_class': MinMaxScaler},
    # {"scaler_name": "StandardScaler", 'scaler_class': StandardScaler},
    # {"scaler_name": "RobustScaler", 'scaler_class': RobustScaler}
]

#Spling the data into train and test sets
spling_configs = [
    {"spling_name": "SMOTE", 'spling_class': SMOTE(random_state=42)},
    {"spling_name": "ADASYN", 'spling_class': ADASYN(random_state=42)},
    {"spling_name": "RandomUnderSampler", 'spling_class': RandomUnderSampler(random_state=42)},
    {"spling_name": "NearMiss", 'spling_class': NearMiss()},
    {"spling_name": "ClusterCentroids", 'spling_class': ClusterCentroids(random_state=42)},
    {"spling_name": "TomekLinks", 'spling_class': TomekLinks()},
    {"spling_name": "SMOTEENN", 'spling_class': SMOTEENN(random_state=42)},
    {"spling_name": "SMOTETomek", 'spling_class': SMOTETomek(random_state=42)},
    {"spling_name": "Threshold", 'spling_class': np.arange(0.1, 1.1, 0.1)},
    {"spling_name": "WithOut", 'spling_class': None},
]

# Converter arrays para DataFrame e Series
target = 'target'
scoring_type = 'accuracy'  # Or 'f1', 'precision', 'recall'

X_train_df = pd.DataFrame(X_train)
y_train_df = pd.Series(y_train.ravel(), name=target)

X_test_df = pd.DataFrame(X_test)
y_test_df = pd.Series(y_test.ravel(), name=target)

# Criar DataFrames finais
df_train = pd.concat([X_train_df, y_train_df], axis=1).reset_index(drop=True)
df_test  = pd.concat([X_test_df, y_test_df], axis=1).reset_index(drop=True)

# Concatene horizontalmente
data_set_train = df_train

#Split the data set into n_partes
n_partes = math.ceil(len(X_train)/len(X_test))
tamanho = len(data_set_train) // n_partes

# data_set_train["grupo"] = 0

for i in range(n_partes - 1):
    data_set_train.iloc[i*tamanho:(i+1)*tamanho, -1] = i  # última coluna é 'grupo'
data_set_train.iloc[(n_partes - 1)*tamanho:, -1] = n_partes - 1

data_set_train_splited = data_set_train

