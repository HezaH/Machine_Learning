import os
import numpy as np
import pandas as pd
import json
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss, ClusterCentroids, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.metrics import (confusion_matrix, classification_report,
                             mean_absolute_error, mean_squared_error, f1_score,
                             accuracy_score, classification_report)

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

# Define model configurations; include tuning keys for models that require parameter tuning.
model_configurations = [
    {
        'class_name': "KNeighborsRegressor",
        'model_class': KNeighborsRegressor,
        'tuning_parameter': 'n_neighbors',
        # 'parameter_range': [5]
        'parameter_range': list(range(5, 40, 5))
    },
    {
        'class_name': "LogisticRegression",
        'model_class': LogisticRegression
    },
    {
        'class_name': "GradientBoostingRegressor",
        'model_class': GradientBoostingRegressor,
        'tuning_parameter': 'n_estimators',
        # 'parameter_range': [50],
        'parameter_range': list(range(50, 100, 50)),
        'random_state': 42
    }
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
    {"spling_name": "WithOut", 'spling_class': None},
]

# Converter arrays para DataFrame e Series
target = 'target'
scoring_type = 'accuracy'  # Or 'f1', 'precision', 'recall'

X_train_df = pd.DataFrame(X_train)
y_train_df = pd.Series(y_train_bin.ravel(), name=target)

X_test_df = pd.DataFrame(X_test)
y_test_df = pd.Series(y_test_bin.ravel(), name=target)

X_val_df = pd.DataFrame(X_val)
y_val_df = pd.Series(y_val_bin.ravel(), name=target)

# Criar DataFrames finais
df_train = pd.concat([X_train_df, y_train_df], axis=1).reset_index(drop=True)
df_test  = pd.concat([X_test_df, y_test_df], axis=1).reset_index(drop=True)
df_val   = pd.concat([X_val_df, y_val_df], axis=1).reset_index(drop=True)

results = []
#Step1
y_train_bin = (y_train > 0).astype(int)

#Step 2

#Step 3

#Step 4

#Step 5

# Loop principal
for model_config in model_configurations:
    model_class_name = model_config['class_name']
    Param = model_config['tuning_parameter']

    for scaler_config in scaler_configs:
        scaler_name = scaler_config['scaler_name']

        for spling_config in spling_configs:
            spling_name = spling_config['spling_name']
            
            for param in model_config['parameter_range']:
                print(f"\n Rodando: Modelo={model_config['class_name']}, "
                      f"Parâmetro={model_config['tuning_parameter']}={param}, "
                      f"Scaler={scaler_config['scaler_name']}, "
                      f"Spling={spling_config['spling_name']}")
                
                set_name = f"{model_class_name}_{model_config['tuning_parameter']}_{param}_{scaler_name}_{spling_name}"
                # Instanciar scaler
                scaler = scaler_config['scaler_class']()

                # Pré-processamento
                X_tr, X_val_scaled, y_tr, y_val = preprocess_data(
                    scaler, df_train, df_val, target_column='target', fit=True
                )


                # Aplicar spling (se aplicável)
                if spling_config['spling_name'] == "Threshold":
                    for threshold in spling_config['spling_class']:
                        model_R = model_config['model_class'](
                            **{model_config['tuning_parameter']: param},
                            random_state=model_config['random_state']
                        )
                        model_R.fit(X_tr, y_tr)
                        y_proba = model_R.predict_proba(X_val_scaled)[:, 1]
                        y_pred = (y_proba >= threshold).astype(int)

                        acc = accuracy_score(y_val, y_pred)
                        print(f"  Threshold={threshold:.1f} -> Accuracy: {acc:.4f}")

                        cm = confusion_matrix(y_val, y_pred).tolist()
                        results.append({
                            "Model": set_name,
                            "Accuracy": acc,
                            'spling': spling_config['spling_name'],
                            'scaling': scaler_config['scaler_name'],
                            "threshold": threshold,
                            "Confusion_Matrix": cm,
                            "Classification_Report": classification_report(y_val, y_pred, output_dict=True),
                            "MAE": mean_absolute_error(y_val, y_pred),
                            "MSE": mean_squared_error(y_val, y_pred),
                            "RMSE": np.sqrt(mean_squared_error(y_val, y_pred))
                        })

                        plot_confusion_matrix(
                            y_true=y_val,
                            y_pred=y_pred,
                            model_name=set_name,
                            labels=["Negative", "Positive"],
                            normalize=False,
                            cm = cm)

                elif spling_config['spling_name'] != "WithOut":
                    X_resampled, y_resampled = spling_config['spling_class'].fit_resample(X_tr, y_tr)

                    model_R = model_config['model_class'](
                        **{model_config['tuning_parameter']: param},
                        random_state=model_config['random_state']
                    )
                    model_R.fit(X_resampled, y_resampled)
                    y_pred = model_R.predict(X_val_scaled) 
                    cm = confusion_matrix(y_val, y_pred).tolist()

                    results.append({
                        "Model": set_name,
                        "Accuracy": accuracy_score(y_val, y_pred),
                        'spling': spling_config['spling_name'],
                        'scaler': scaler_config['scaler_name'],
                        "threshold": None,
                        "Confusion_Matrix": cm,
                        "Classification_Report": classification_report(y_val, y_pred, output_dict=True),
                        "MAE": mean_absolute_error(y_val, y_pred),
                        "MSE": mean_squared_error(y_val, y_pred),
                        "RMSE": np.sqrt(mean_squared_error(y_val, y_pred))
                    })

                    plot_confusion_matrix(
                        y_true=y_val,
                        y_pred=y_pred,
                        model_name=set_name,
                        labels=["Negative", "Positive"],
                        normalize=False,
                        cm = cm)

                else:
                    # Caso sem balanceamento
                    model_R = model_config['model_class'](
                        **{model_config['tuning_parameter']: param},
                        random_state=model_config['random_state']
                    )
                    model_R.fit(X_tr, y_train_bin)
                    y_pred = model_R.predict(X_tr)
                    mask_pos = (y_tr == 1)
                    X_train_1 = X_tr[mask_pos]
                    y_train_1 = y_train[mask_pos]

                    cm = confusion_matrix(y_val, y_pred).tolist()
                    results.append({
                        "Model": set_name,
                        "Accuracy": accuracy_score(y_val, y_pred),
                        'spling': spling_config['spling_name'],
                        'scaler': scaler_config['scaler_name'],
                        "threshold": None,
                        "Confusion_Matrix": cm,
                        "Classification_Report": classification_report(y_val, y_pred, output_dict=True),
                        "MAE": mean_absolute_error(y_val, y_pred),
                        "MSE": mean_squared_error(y_val, y_pred),
                        "RMSE": np.sqrt(mean_squared_error(y_val, y_pred))
                    })

                    plot_confusion_matrix(
                        y_true=y_val,
                        y_pred=y_pred,
                        model_name=set_name,
                        labels=["Negative", "Positive"],
                        normalize=False,
                        cm = cm,
                    )
                print("\n")
