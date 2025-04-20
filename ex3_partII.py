import os
import numpy as np
import pandas as pd
import json
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss, ClusterCentroids, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score)

from utils import (
     preprocess_data, plot_confusion_matrices, plot_confusion_matrix)


current_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'A652.pickle')

f = open( current_dir , 'rb')
( X_train , y_train , X_val , y_val , X_test , y_test ) = pickle.load(f) 

print(f"Shapes: {X_train.shape}, {X_test.shape}, {X_val.shape}")

# Transformar os valores contínuos em rótulos binários
y_train_bin = np.where(y_train == 0, 0, 1)
y_val_bin  = np.where(y_val == 0, 0, 1)
y_test_bin  = np.where(y_test == 0, 0, 1)
# Definir as configurações dos modelos
model_configurations = [
    { 
        'class_name': "LogisticRegression", 
        'model_class': LogisticRegression,
        'params': {'solver': 'lbfgs', 'max_iter': 1000, 'random_state': 42}
    },
    { 
        'class_name': "DecisionTreeClassifier", 
        'model_class': DecisionTreeClassifier,
        'params': {'random_state': 42}
    },
    { 
        'class_name': "RandomForestClassifier", 
        'model_class': RandomForestClassifier,
        'params': {'n_estimators': 100, 'random_state': 42}
    },
]

# Define scaler configurations
scaler_configs = [
    {"scaler_name": "MinMaxScaler", 'scaler_class': MinMaxScaler},
    {"scaler_name": "StandardScaler", 'scaler_class': StandardScaler},
    {"scaler_name": "RobustScaler", 'scaler_class': RobustScaler}
]

#Spling the data into train and test sets
spling_configs = [
    # {"spling_name": "SMOTE", 'spling_class': SMOTE(random_state=42)},
    # {"spling_name": "ADASYN", 'spling_class': ADASYN(random_state=42)},
    # {"spling_name": "RandomUnderSampler", 'spling_class': RandomUnderSampler(random_state=42)},
    # {"spling_name": "NearMiss", 'spling_class': NearMiss()},
    # {"spling_name": "ClusterCentroids", 'spling_class': ClusterCentroids(random_state=42)},
    # {"spling_name": "TomekLinks", 'spling_class': TomekLinks()},
    # {"spling_name": "SMOTEENN", 'spling_class': SMOTEENN(random_state=42)},
    # {"spling_name": "SMOTETomek", 'spling_class': SMOTETomek(random_state=42)},
    # {"spling_name": "Threshold", 'spling_class': np.arange(0.1, 1.0, 0.1)},
    {"spling_name": "WithOut", 'spling_class': "None"},
]

# Converter arrays para DataFrame e Series
target = 'target'
scoring_type = 'accuracy'  # Or 'f1', 'precision', 'recall'

target = 'target'
X_train_df = pd.DataFrame(X_train)
y_train_bin_df = pd.Series(y_train_bin.ravel(), name=target)
# OBSERVAÇÃO: Mantenha também y_train_cont, que é o y_train original com valores contínuos.
y_train_cont_df = pd.Series(y_train.ravel(), name='precipitation')

X_test_df = pd.DataFrame(X_test)
y_test_bin_df = pd.Series(y_test_bin.ravel(), name=target)

X_val_df = pd.DataFrame(X_val)
y_val_df = pd.Series(y_val_bin.ravel(), name=target)

# Criar DataFrames finais
df_train = pd.concat([X_train_df, y_train_bin_df], axis=1).reset_index(drop=True)
df_test  = pd.concat([X_test_df, y_test_bin_df], axis=1).reset_index(drop=True)
df_val   = pd.concat([X_val_df, y_val_df], axis=1).reset_index(drop=True)

def predict_precipitation_final(classifier, regressor, x):
    x_formatted = np.array(x).reshape(1, -1)
    class_pred = classifier.predict(x_formatted)[0]
    if class_pred == 0:
        return 0
    else:
        return regressor.predict(x_formatted)[0]
    

results = []

# Loop principal
for model_config in model_configurations:
    model_class_name = model_config['class_name']
    # Exemplo de parâmetro de tuning se houver
    Param = model_config.get('tuning_parameter', None)  

    for scaler_config in scaler_configs:
        scaler_name = scaler_config['scaler_name']

        for spling_config in spling_configs:
            spling_name = spling_config['spling_name']
            
            for param in model_config.get('parameter_range', [None]):
                print(f"\nRodando: Modelo={model_config['class_name']}, "
                      f"Parâmetro={model_config.get('tuning_parameter','')}"
                      f"={param}, "
                      f"Scaler={scaler_config['scaler_name']}, "
                      f"Spling={spling_config['spling_name']}")
                set_name = f"{model_class_name}&{model_config.get('tuning_parameter','')}&{param}&{scaler_name}&{spling_name}"
                
                # Instanciar o scaler
                scaler = scaler_config['scaler_class']()

                # Pré-processamento: aqui ajustando o scaler (fit=True) com os dados de treino e transformando o conjunto de validação
                X_tr, X_val_scaled, y_tr, y_val = preprocess_data(
                    scaler, df_train, df_val, target_column='target', fit=True
                )
                
                # Instanciar o modelo com o parâmetro de tuning (se aplicável)
                # Use 'random_state' se presente nas configurações
                model_params = {}
                if Param is not None and param is not None:
                    model_params[model_config['tuning_parameter']] = param
                
                if 'random_state' in model_config:
                    model_params['random_state'] = model_config['random_state']
                
                model = model_config['model_class'](**model_params)
                
                # Aqui você pode aplicar o método de reamostragem (spling) se não for "WithOut"
                # Por simplicidade, neste exemplo vamos utilizar os dados pré-processados
                model.fit(X_tr, y_tr)
                
                # Obter as predições para o conjunto de validação
                y_pred = model.predict(X_val_scaled)
                
                # Verifica se o modelo é de classificação ou regressão.
                # Uma forma prática é testar a existência do método predict_proba.
                # Métricas de classificação
                acc = accuracy_score(y_val, y_pred)
                report = classification_report(y_val, y_pred, output_dict=True)
                cm = confusion_matrix(y_val, y_pred).tolist()
                
                # Armazena ou imprime as métricas
                print(f"Set: {set_name}")
                print(f"Acurácia: {acc:.4f}")
                print("Matriz de Confusão:", cm)
                print("Relatório de Classificação:", classification_report(y_val, y_pred))
                
                # Armazenar o resultado (exemplo de dicionário)
                results.append({
                    "Model": set_name,
                    "Accuracy": acc,
                    "Confusion_Matrix": cm,
                    "Tp": cm[0][0],
                    "Fp": cm[0][1],
                    "Fn": cm[1][0],
                    "Tn": cm[1][1],
                    "Classification_Report": report,
                    "Y": y_val,
                    "YPred": y_pred
                })
                
                print("\n")

df_res = pd.DataFrame(results)
df_res = df_res[(df_res["Tp"] > 0) & (df_res["Fp"]>0) & (df_res["Tn"] > 0) & (df_res["Fn"]>0)]  # Filtrar apenas os resultados com True Positives
best_idx = df_res['Accuracy'].idxmax()
best_cfg = df_res.loc[best_idx]

model_selected =  [mod for mod in model_configurations if mod['class_name'] == best_cfg["Model"].split("&")[0]][0]
scaler_selected = [sca for sca in scaler_configs if sca['scaler_name'] == best_cfg["Model"].split("&")[3]][0]

# Instanciar o scaler
scaler = scaler_selected['scaler_class']()

# Pré-processamento: aqui ajustando o scaler (fit=True) com os dados de treino e transformando o conjunto de validação
X_tr, X_val_scaled, y_tr, y_val = preprocess_data(
    scaler, df_train, df_val, target_column='target', fit=True
)

# Instanciar o modelo com o parâmetro de tuning (se aplicável)
# Use 'random_state' se presente nas configurações
model_params = {}
if Param is not None and param is not None:
    model_params[model_selected['tuning_parameter']] = param

if 'random_state' in model_selected:
    model_params['random_state'] = model_selected['random_state']

model_C = model_selected['model_class'](**model_params)

# Aqui você pode aplicar o método de reamostragem (spling) se não for "WithOut"
# Por simplicidade, neste exemplo vamos utilizar os dados pré-processados
model_C.fit(X_tr, y_tr)

# Obter as predições para o conjunto de validação
y_pred = model_C.predict(X_val_scaled)

# Após treinar o modelo C com X_tr e y_tr:
model_C.fit(X_tr, y_tr)

# Obter ay_train_preds predições no próprio conjunto de treino
y_train_pred = model_C.predict(X_tr)

# Cria uma máscara para selecionar os exemplos classificados como 1
mask_positivos = (y_train_pred == 1)

# Definir X_train_1 e y_train_1 com base na máscara:
X_train_1 = X_tr[mask_positivos]
y_train_1 = y_tr[mask_positivos]

X_train_1_df = pd.DataFrame(X_train_1)
y_train_1_df = pd.Series(y_train_1.ravel(), name=target)

# Criar DataFrames finais
df_train_1 = pd.concat([X_train_1_df, y_train_1_df], axis=1).reset_index(drop=True)

# Definir as configurações dos modelos de regressão
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
# Loop principal
for model_config in model_regression:
    model_class_name = model_config['class_name']
    # Exemplo de parâmetro de tuning se houver
    Param = model_config.get('tuning_parameter', None)  

    for scaler_config in scaler_configs:
        scaler_name = scaler_config['scaler_name']

        for spling_config in spling_configs:
            spling_name = spling_config['spling_name']
            
            for param in model_config.get('parameter_range', [None]):
                print(f"\nRodando: Modelo={model_config['class_name']}, "
                      f"Parâmetro={model_config.get('tuning_parameter','')}"
                      f"={param}, "
                      f"Scaler={scaler_config['scaler_name']}, "
                      f"Spling={spling_config['spling_name']}")
                
                set_name = f"{model_class_name}&{model_config.get('tuning_parameter','')}&{param}&{scaler_name}&{spling_name}"
                
                # Instanciar o scaler
                scaler = scaler_config['scaler_class']()

                # Pré-processamento: aqui ajustando o scaler (fit=True) com os dados de treino e transformando o conjunto de validação
                X_tr_1, X_t, y_tr_1, y_t = preprocess_data(
                    scaler, df_train_1, df_test, target_column='target', fit=True
                )
                
                # Instanciar o modelo com o parâmetro de tuning (se aplicável)
                # Use 'random_state' se presente nas configurações
                model_params = {}
                if Param is not None and param is not None:
                    model_params[model_config['tuning_parameter']] = param
                
                if 'random_state' in model_config:
                    model_params['random_state'] = model_config['random_state']
                
                model_R = model_config['model_class'](**model_params)

                                # Aqui você pode aplicar o método de reamostragem (spling) se não for "WithOut"
                # Por simplicidade, neste exemplo vamos utilizar os dados pré-processados
                model_R.fit(X_tr_1, y_tr_1)
                
                # Obter as predições para o conjunto de validação
                y_pred_reg = model_R.predict(X_t)
                # Prever a ocorrência de precipitação
                y_pred_class = model_C.predict(X_t)
                
                y_pred_final = np.where(y_pred_class == 0, 0, y_pred_reg)

                # Avaliar o desempenho
                mse = mean_squared_error(y_pred_reg, y_pred_final)
                mse = mean_squared_error(y_pred_reg, y_pred_final)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_pred_reg, y_pred_final)

                final_results.append({
                    "Model": set_name,
                    "MSE":  mse,
                    "RMSE": rmse,
                    "r2":   r2
                })


df_final = pd.DataFrame(final_results)


 


