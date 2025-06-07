import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (confusion_matrix, classification_report,
                             mean_absolute_error, mean_squared_error, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split, KFold


#função para criar a matriz de confusão e relatório de classificação
def MatConf(verdadeiros, previstos, titulo, rotulos_x = "AxisX", rotulos_y = "AxisY"):
  conf_matrix = confusion_matrix(verdadeiros, previstos,  normalize="true")
  s = sns.heatmap(conf_matrix, annot=True, cmap="Greens", 
              xticklabels=rotulos_x, yticklabels=rotulos_y)
  s.set(xlabel = "Rótulo Previsto", ylabel="Rótulo Verdadeiro", title=titulo)
#   plt.show()
  print(classification_report(verdadeiros, previstos))


def cross_validate_models(model_configurations, scaler_configs, dataset, target_column, cv=5, 
                          fit_scaler=True, scoring_type="accuracy"):
    """
    Performs cross-validation on a dataset, evaluating a set of machine learning models 
    with different scaling/preprocessing methods. Additionally, computes evaluation metrics 
    including a configurable scoring value, confusion matrix, classification report, MAE, MSE, and RMSE.
    
    Parameters:
        model_configurations (list of dict): List of dictionaries containing model classes 
            and their parameters. To tune a parameter, include:
            - 'tuning_parameter': The name of the parameter to tune (e.g., 'n_neighbors')
            - 'parameter_range': List or range of values to test.
            Example:
                {'class_name': "KNeighborsClassifier",
                 'model_class': KNeighborsClassifier,
                 'tuning_parameter': 'n_neighbors',
                 'parameter_range': list(range(1, 40))}
        scaler_configs (list of dict): List of dictionaries containing scaler classes and names.
            Example:
                {'scaler_name': "MinMaxScaler", 'scaler_class': MinMaxScaler}
        dataset (pd.DataFrame): The complete dataset for training (including target column).
        target_column (str): Name of the target column.
        cv (int): Number of folds for cross-validation.
        fit_scaler (bool): If True, fits the scaler on training data; otherwise, only transforms.
        scoring_type (str): Metric to be observed: "accuracy", "f1", "precision", or "recall". 
            Default is "accuracy".
            
    Returns:
        pd.DataFrame: DataFrame containing aggregated scores and evaluation metrics for each model/scaler combination.
    """
    
    def preprocess_data(scaler, df_train, df_test, target_column, fit=True):
        # Drop target column and scale features
        if fit:
            X_train = scaler.fit_transform(df_train.drop(target_column, axis=1))
            X_test  = scaler.transform(df_test.drop(target_column, axis=1))
        else:
            X_train = scaler.transform(df_train.drop(target_column, axis=1))
            X_test  = scaler.transform(df_test.drop(target_column, axis=1))
        y_train = df_train[target_column]
        y_test  = df_test[target_column]
        return X_train, X_test, y_train, y_test

    def compute_scoring(y_true, y_pred, scoring_type):
        """
        Compute the scoring metric based on scoring_type.
        """
        if scoring_type == "accuracy":
            return np.mean(y_true == y_pred)
        elif scoring_type == "f1":
            # Using weighted average to account for multiple classes
            return f1_score(y_true, y_pred, average='weighted')
        elif scoring_type == "precision":
            return precision_score(y_true, y_pred, average='weighted')
        elif scoring_type == "recall":
            return recall_score(y_true, y_pred, average='weighted')
        else:
            # Default to accuracy
            return np.mean(y_true == y_pred)
    
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []  # list to store scores across folds

    fold = 1
    for train_index, test_index in kf.split(dataset):
        df_train = dataset.iloc[train_index]
        df_test  = dataset.iloc[test_index]
        print(f"Processing fold {fold}...")
        
        for scaler_config in scaler_configs:
            scaler = scaler_config['scaler_class']()
            scaler_name = scaler_config['scaler_name']
            
            X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(scaler, df_train, df_test, target_column, fit=fit_scaler)
            
            for model_config in model_configurations:
                model_class = model_config['model_class']
                # Extract base parameters (excluding meta keys)
                base_model_params = {k: v for k, v in model_config.items() 
                                     if k not in ['model_class', 'class_name', 'tuning_parameter', 'parameter_range']}
                
                # Check if tuning is specified for this model.
                if 'tuning_parameter' in model_config and 'parameter_range' in model_config:
                    tuning_param = model_config['tuning_parameter']
                    best_score = -np.inf
                    best_model_params = None
                    best_evaluation = {}  # To store all evaluation metrics from the best parameter
                    # Loop over the range of parameter values
                    for param_value in model_config['parameter_range']:
                        current_params = base_model_params.copy()
                        current_params[tuning_param] = param_value
                        model = model_class(**current_params)
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        
                        score_value = compute_scoring(y_test, y_pred, scoring_type)
                        
                        # If this configuration gives a better score, save it
                        if score_value > best_score:
                            best_score = score_value
                            best_model_params = current_params.copy()
                            # Compute additional metrics for the best configuration
                            best_evaluation = {
                                "Confusion_Matrix": confusion_matrix(y_test, y_pred).tolist(),
                                "Classification_Report": classification_report(y_test, y_pred, output_dict=True),
                                "MAE": mean_absolute_error(y_test, y_pred),
                                "MSE": mean_squared_error(y_test, y_pred),
                                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
                            }
                            MatConf(y_test, y_pred, f"{model_config['class_name']}_{scaler_name}")

                    model_identifier = f"{model_config['class_name']}_{scaler_name}_{tuning_param}={param_value}"
                    score_record = {
                        "Model_Scaler": f"{model_config['class_name']}_{scaler_name}",
                        "Scoring": best_score,
                        "Parameters": best_model_params,
                        "Evaluation": best_evaluation
                    }
                    scores.append(score_record)
                else:
                    # No tuning; use base parameters
                    model = model_class(**base_model_params)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)

                    MatConf(y_test, y_pred, f"{model_config['class_name']}_{scaler_name}")

                    score_value = compute_scoring(y_test, y_pred, scoring_type)
                    evaluation_metrics = {
                        "Confusion_Matrix": confusion_matrix(y_test, y_pred).tolist(),
                        "Classification_Report": classification_report(y_test, y_pred, output_dict=True),
                        "MAE": mean_absolute_error(y_test, y_pred),
                        "MSE": mean_squared_error(y_test, y_pred),
                        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
                    }
                    
                    model_identifier = f"{model_config['class_name']}_{scaler_name}"
                    score_record = {
                        "Model_Scaler": model_identifier,
                        "Scoring": score_value,
                        "Parameters": base_model_params,
                        "Evaluation": evaluation_metrics
                    }
                    scores.append(score_record)
        fold += 1
        
    df_scores = pd.DataFrame(scores)
    return df_scores

# --- Example usage ---

# Set column names for the dataset
column_names = ["ESCT", "NDEP", "RENDA", "TIPOR", "VBEM", "NPARC", 
                "VPARC", "TEL", "IDADE", "RESMS", "ENTRADA", "CLASSE"]

# Load datasets (assuming files are in the same directory as this script)
current_dir = os.path.dirname(os.path.realpath(__file__))
df_test = pd.read_csv(os.path.join(current_dir, 'credtest.txt'), sep='\t', header=None)
df_test.columns = column_names

df_train = pd.read_csv(os.path.join(current_dir, 'credtrain.txt'), sep='\t', header=None)
df_train.columns = column_names

# Define model configurations; include tuning keys for models that require parameter tuning.
model_configurations = [
    {
        'class_name': "KNeighborsClassifier", 
        'model_class': KNeighborsClassifier, 
        'tuning_parameter': 'n_neighbors', 
        'parameter_range': list(range(1, 20, 2))
    },
    {
        'class_name': "LogisticRegression", 
        'model_class': LogisticRegression
    },
    {
        'class_name': "GradientBoostingClassifier", 
        'model_class': GradientBoostingClassifier, 
        'tuning_parameter': 'n_estimators',
        'parameter_range': list(range(50, 151, 10)),
        'random_state': 42
    }
]

# Define scaler configurations
scaler_configs = [
    {"scaler_name": "MinMaxScaler", 'scaler_class': MinMaxScaler},
    # {"scaler_name": "StandardScaler", 'scaler_class': StandardScaler},
    # {"scaler_name": "RobustScaler", 'scaler_class': RobustScaler}
]

target = 'CLASSE'
dataset_train = df_train.copy()
scoring_type = 'accuracy'  # Or 'f1', 'precision', 'recall'

# Get cross-validation scores with additional evaluation metrics
df_scores = cross_validate_models(model_configurations, scaler_configs, dataset_train, target, cv=5, fit_scaler=True, scoring_type=scoring_type)

# Aggregate scores by 'Model_Scaler'
score_summary = df_scores.groupby('Model_Scaler').agg({'Scoring': ['mean', 'std']}).reset_index()

# Print aggregated results
for _, row in score_summary.iterrows():
    model_scaler = row['Model_Scaler']
    mean_score = row["Scoring"]['mean']
    std_score = row["Scoring"]['std']
    print(f"Model Scaler: {model_scaler}")
    print(f"  Average {scoring_type}: {mean_score:.4f}")
    print(f"  Std Dev: {std_score:.4f}")
    print("-" * 40)

# (Optional) Determine and print the best performing configuration
best_config = score_summary.loc[score_summary[("Scoring", "mean")].idxmax()]
print("Best performing configuration:")
print(best_config)


set_model = score_summary[score_summary["Scoring"]['mean'] == score_summary["Scoring"]['mean'].max()]["Model_Scaler"].to_list()[0]

model_configurations_name = set_model.split("_")[0]
scaler_configs_name = set_model.split("_")[1]

set_class = [model for model in model_configurations if model['class_name'] == model_configurations_name]
set_estimators = [model for model in scaler_configs if model["scaler_name"] == scaler_configs_name]

x = 1