import os
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from utils import (
    calculate_metrics, find_optimal_cost_threshold, cross_validate_models)


current_dir = os.path.dirname(os.path.realpath(__file__))
df_diamonds = pd.read_csv(os.path.join(current_dir,'diamonds.csv'))


df_diamonds.head()
columns = ['cut', 'color', 'clarity']
dict_unique = {}
for c in columns:
    dict_unique[c] = df_diamonds[c].unique()
print(dict_unique)


#To colum cut I will assing 0 at 5 by each value on the column
df_diamonds['cut'] = df_diamonds['cut'].map({'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4})

#Other way could be to use LabelEncoder from sklearn
# le = LabelEncoder()
# df_diamonds['cut_modify']  = le.fit_transform(df_diamonds['cut'] )

#To others columns with objects values, to not introduced the articial values
# df_base = pd.DataFrame()
for col in columns[1:]:
    # Starting the OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)

    # Applying OneHotEncoder
    encoded_data = encoder.fit_transform(df_diamonds[[col]])

    # Converting the encoded data to a DataFrame
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([col]))

    # Concatenating the encoded DataFrame with the original DataFrame
    df_diamonds = pd.concat([df_diamonds, encoded_df], axis=1)

df_diamonds_done = df_diamonds.drop(columns=columns[1:], axis=1)

target = 'price'
X = df_diamonds_done.drop(target, axis=1)
y = df_diamonds_done[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


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

# Define scaler configurations
scaler_configs = [
    {"scaler_name": "MinMaxScaler", 'scaler_class': MinMaxScaler},
    # {"scaler_name": "StandardScaler", 'scaler_class': StandardScaler},
    # {"scaler_name": "RobustScaler", 'scaler_class': RobustScaler}
]

#Spling the data into train and test sets
spling_configs = [
    {"spling_name": "WithOut", 'spling_class': None},
]

# Get cross-validation scores with additional evaluation metrics
scoring_type = 'accuracy'  # Or 'f1', 'precision', 'recall'
# Certifique-se de que y_train seja um DataFrame com um nome de coluna
if isinstance(y_train, pd.Series):
    y_train = y_train.to_frame(name=target)

# Redefina os índices para garantir que sejam únicos e alinhados
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

# Concatene horizontalmente
data_set_train = pd.concat([X_train, y_train], axis=1)

#Split the data set into n_partes
n_partes = math.ceil(len(X_train)/len(X_test))
tamanho = len(data_set_train) // n_partes

data_set_train["grupo"] = 0

for i in range(n_partes - 1):
    data_set_train.iloc[i*tamanho:(i+1)*tamanho, -1] = i  # última coluna é 'grupo'
data_set_train.iloc[(n_partes - 1)*tamanho:, -1] = n_partes - 1

data_set_train_splited = data_set_train

df_scores = cross_validate_models(model_configurations, scaler_configs, spling_configs, data_set_train_splited, target, cv=2, fit_scaler=True, scoring_type=scoring_type)

# Aggregate scores by 'Model_Scaler'
score_summary = df_scores.groupby('Model_Scaler').agg({'Scoring': ['mean', 'std']}).reset_index()

# Print aggregated results
for _, row in score_summary.iterrows():
    model_scaler = row['Model_Scaler']
    mean_acc = row["Scoring"]['mean']
    std_acc = row["Scoring"]['std']
    print(f"Model Scaler: {model_scaler}")
    print(f"  Average Scoring: {mean_acc:.4f}")
    print(f"  Scoring Std Dev: {std_acc:.4f}")
    print("-" * 40)

# Optionally, determine and print the best performing configuration
best_config = score_summary.loc[score_summary[("Scoring", "mean")].idxmax()]
print("Best performing configuration:")
print(best_config)

x = 1