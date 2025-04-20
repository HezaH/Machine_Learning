import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import ( mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix
                             )
from utils import (
    get_model_scores, preprocess_data)
import os

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

target_column = 'price'
X = df_diamonds_done.drop(target_column, axis=1)
y = df_diamonds_done[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
df_train = pd.DataFrame(X_train, columns=X.columns)
df_train[target_column] = y_train

df_test = pd.DataFrame(X_test, columns=X.columns)
df_test[target_column] = y_test

X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.4, random_state=42)
df_train_split = pd.DataFrame(X_train_split, columns=X.columns)
df_train_split[target_column] = y_train_split

df_val = pd.DataFrame(X_val, columns=X.columns)
df_val[target_column] = y_val

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

results = []
# Loop principal
for model_config in model_configurations:
    model_class_name = model_config['class_name']
    model_class = model_config['model_class']

    # Listas para armazenar dados das curvas ROC
    roc_data = []

    for scaler_config in scaler_configs:
        scaler_name = scaler_config['scaler_name']
        scaler_class = scaler_config['scaler_class']
        set_name = f"{model_class_name}_{scaler_name}"
        
        # Extract base parameters (excluding meta keys)
        base_model_params = {k: v for k, v in model_config.items()
                                if k not in ['model_class', 'class_name', 'tuning_parameter', 'parameter_range']}
        
        print(f"\nRodando: Modelo={model_class_name}, Scaler={scaler_name}")

        # Instanciar scaler e modelo
        scaler = scaler_class()
        model = model_class(**base_model_params)
    
        # Obter scores e rótulos de validação
        y_val_pred, y_val_true = get_model_scores(model, scaler, df_train_split, df_val, target_column=target_column)

        #Pré-processar o conjunto de teste usando o scaler ajustado:
        # Supondo que df_train_split seja o conjunto de treino utilizado para ajustar o scaler
        _, X_test_processed, _, y_test_processed = preprocess_data(scaler, df_train, df_test, target_column, fit=False)
                
        # Gerar as predições para o conjunto de teste
        y_pred = model.predict(X_test_processed)

        if "Regressor" in model_class_name:
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
        else:
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            # Se utilizar predict_proba para obter probabilidades, pode calcular:
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            loss = log_loss(y_test, model.predict_proba(X_test)[:, 1])
            cm = confusion_matrix(y_test, y_pred)
            
            # 1. Predicted vs Actual
            plt.figure(figsize=(6, 4))
            plt.scatter(y_test_processed, y_pred)
            plt.plot([y_test_processed.min(), y_test_processed.max()], [y_test_processed.min(), y_test_processed.max()], 'r--')
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title("Predicted vs Actual")
            plt.grid(True)
            plt.show()

            # 2. Residuals Plot
            residuals = y_test_processed - y_pred
            plt.figure(figsize=(6, 4))
            plt.scatter(y_pred, residuals)
            plt.axhline(0, color='red', linestyle='--')
            plt.xlabel("Predicted")
            plt.ylabel("Residuals")
            plt.title("Residual Plot")
            plt.grid(True)
            plt.show()

        results.append({
            "Model": set_name,
            "accuracy": accuracy if "Regressor" not in model_class_name else None,
            "precision": precision if "Regressor" not in model_class_name else None,
            "recall": recall if "Regressor" not in model_class_name else None,
            "f1": f1 if "Regressor" not in model_class_name else None,
            "roc_auc": roc_auc if "Regressor" not in model_class_name else None,
            "loss": loss if "Regressor" not in model_class_name else None,
            "cm": cm if "Regressor" not in model_class_name else None,
            "MAE": mae if "Regressor" in model_class_name else None,
            "MSE": mse if "Regressor" in model_class_name else None,
            "RMSE": rmse if "Regressor" in model_class_name else None,
            "R2": r2 if "Regressor" in model_class_name else None,})
        

df_res = pd.DataFrame(results)



