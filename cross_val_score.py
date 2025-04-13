import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, KFold


def cross_validate_models(model_configurations, scaler_configs, dataset, target_column, cv=5, fit_scaler=True, scoring_type="accuracy"):
    """
    Performs cross-validation on a dataset, evaluating a set of machine learning models 
    with different scaling/preprocessing methods.
    
    Parameters:
        model_configurations (list of dict): List of dictionaries containing model classes 
            and their optional parameters. Example:
            {'class_name': "LogisticRegression", 'model_class': LogisticRegression}
        scaler_configs (list of dict): List of dictionaries containing scaler classes 
            and names. Example:
            {'scaler_name': "MinMaxScaler", 'scaler_class': MinMaxScaler}
        dataset (pd.DataFrame): The complete dataset for training (including target column).
        target_column (str): Name of the target column.
        cv (int): Number of folds for cross-validation.
        fit_scaler (bool): If True, fit the scaler on training data; else, only transform.
        
    Returns:
        pd.DataFrame: DataFrame with aggregated scores for each model/scaler combination.
    """
    
    def preprocess_data(scaler, df_train, df_test, target_column, fit=True):
        """
        Preprocesses the training and testing data using the given scaler.
        """
        # Drop target column and scale features
        if fit:
            X_train = scaler.fit_transform(df_train.drop(target_column, axis=1))
            X_test = scaler.transform(df_test.drop(target_column, axis=1))
        else:
            X_train = scaler.transform(df_train.drop(target_column, axis=1))
            X_test = scaler.transform(df_test.drop(target_column, axis=1))
        
        y_train = df_train[target_column]
        y_test = df_test[target_column]
        return X_train, X_test, y_train, y_test
    
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []  # List to store scores across folds
    
    fold = 1
    for train_index, test_index in kf.split(dataset):
        df_train = dataset.iloc[train_index]
        df_test = dataset.iloc[test_index]
        print(f"Processing fold {fold}...")
        
        # Loop through each scaling configuration
        for scaler_config in scaler_configs:
            # Instantiate the scaler
            scaler = scaler_config['scaler_class']()
            scaler_name = scaler_config['scaler_name']
            
            # Preprocess the data using the current scaler
            X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(scaler, df_train, df_test, target_column, fit=fit_scaler)
            
            # Loop through each model configuration
            for model_config in model_configurations:
                model_class = model_config['model_class']
                # Extract optional parameters (those keys not reserved for meta-information)
                model_params = {k: v for k, v in model_config.items() 
                                if k not in ['model_class', 'class_name', 'param_grid', 'parameter_range']}
                if 'param_grid' in model_config and 'parameter_range' in model_config:
                    tuning_param = model_config['param_grid']
                    best_accuracy = -np.inf
                    best_model_params = None
                    
                    # Loop over the range of values for tuning
                    for param_value in model_config['parameter_range']:
                        current_params = model_params.copy()
                        
                        # Set the tuning parameter to the current value
                        current_params[tuning_param] = param_value
                        model = model_class(**current_params)
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        accuracy = np.mean(y_pred == y_test)
                        
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_model_params = current_params.copy()
                    
                    # Define the model identifier that includes the tuned parameter value
                    model_identifier = f"{model_config['class_name']}_{scaler_name}"
                    # Append results from the best tuned parameter for this fold
                    score_record = {
                        "Model_Scaler": model_identifier,
                        "Scoring": best_accuracy,
                        "Parameters": best_model_params
                    }
                    scores.append(score_record)
                
                else:
                    # No tuning; use base parameters
                    model = model_class(**model_params)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    
                    accuracy = np.mean(y_pred == y_test)
                    model_identifier = f"{model_config['class_name']}_{scaler_name}"
                    score_record = {
                        "Model_Scaler": model_identifier,
                        "Scoring": accuracy,
                        "Parameters": model_params
                    }
                    scores.append(score_record)
        fold += 1
        
    df_scores = pd.DataFrame(scores)
    return df_scores


# Setting column names for the dataset
column_names = ["ESCT", "NDEP", "RENDA", "TIPOR", "VBEM", "NPARC", "VPARC", "TEL", "IDADE", "RESMS", "ENTRADA", "CLASSE"]

# Loading datasets (assuming files are in the same directory do script)
current_dir = os.path.dirname(os.path.realpath(__file__))
df_test = pd.read_csv(os.path.join(current_dir, 'credtest.txt'), sep='\t', header=None)
df_test.columns = column_names

df_train = pd.read_csv(os.path.join(current_dir, 'credtrain.txt'), sep='\t', header=None)
df_train.columns = column_names

# Define model configurations
# Define model configurations; for models that require parameter tuning, include tuning keys:
model_configurations = [
    {
        'class_name': "KNeighborsClassifier", 
        'model_class': KNeighborsClassifier, 
        'param_grid': 'n_neighbors', 
        'parameter_range': list(range(1, 40 , 2))
    },
    {
        'class_name': "LogisticRegression", 
        'model_class': LogisticRegression
    },
    {
        'class_name': "GradientBoostingClassifier", 
        'model_class': GradientBoostingClassifier, 
        'param_grid': 'n_estimators',
        'parameter_range': list(range(50, 151, 10)),
        'random_state': 42
    }
]

# Define scaler configurations
scaler_configs = [
    {"scaler_name": "MinMaxScaler", 'scaler_class': MinMaxScaler},
    {"scaler_name": "StandardScaler", 'scaler_class': StandardScaler},
    {"scaler_name": "RobustScaler", 'scaler_class': RobustScaler}
]

# Use df_train as the training dataset
train_dataset = df_train.copy()
target = 'CLASSE'
scoring_type = 'accuracy'

# Get cross-validation scores
df_scores = cross_validate_models(model_configurations, scaler_configs, train_dataset, target, cv=5, fit_scaler=True, scoring_type=scoring_type)

# Aggregate scores by 'Model_Scaler'
score_summary = df_scores.groupby('Model_Scaler').agg({'Scoring': ['mean', 'std']}).reset_index()

# Display aggregated scores
for _, row in score_summary.iterrows():
    model_scaler = row['Model_Scaler']
    mean_acc = row["Scoring"]['mean']
    std_acc = row["Scoring"]['std']
    print(f"Model Scaler: {model_scaler}")
    print(f"  Average Scoring: {mean_acc:.4f}")
    print(f"  Scoring Std Dev: {std_acc:.4f}")
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