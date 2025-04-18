import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (confusion_matrix, classification_report,
                             mean_absolute_error, mean_squared_error, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split, KFold


#função para criar a matriz de confusão e relatório de classificação
def MatConf(verdadeiros, previstos, titulo, rotulos_x = "AxisX", rotulos_y = "AxisY"):
  conf_matrix =  confusion_matrix(verdadeiros, previstos,  normalize="true")
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
                model_class_name = model_config['class_name']
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
                            if "Regressor" in model_class_name:
                                # Applying discretization for regression
                                bins = np.linspace(y_test.min(), y_test.max(), num=5)  # Adjust the number of bins as needed
                                y_test_cat = np.digitize(y_test, bins=bins, right=False)
                                y_pred_cat = np.digitize(y_pred, bins=bins, right=False)

                                # For regression models, we can use MAE, MSE, RMSE
                                best_evaluation = {
                                    "Confusion_Matrix": confusion_matrix(y_test_cat, y_pred_cat).tolist(),
                                    "MAE": mean_absolute_error(y_test, y_pred),
                                    "MSE": mean_squared_error(y_test, y_pred),
                                    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
                                }
                                MatConf(y_test_cat, y_pred_cat, f"{model_class_name}_{scaler_name}")
                            else:
                                best_evaluation = {
                                    "Confusion_Matrix": confusion_matrix(y_test, y_pred).tolist(),
                                    "Classification_Report": classification_report(y_test, y_pred, output_dict=True),
                                    "MAE": mean_absolute_error(y_test, y_pred),
                                    "MSE": mean_squared_error(y_test, y_pred),
                                    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
                                }
                                MatConf(y_test, y_pred, f"{model_class_name}_{scaler_name}")

                    model_identifier = f"{model_class_name}_{scaler_name}_{tuning_param}={param_value}"
                    score_record = {
                        "Model_Scaler": f"{model_class_name}_{scaler_name}",
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

                    MatConf(y_test, y_pred, f"{model_class_name}_{scaler_name}")

                    score_value = compute_scoring(y_test, y_pred, scoring_type)
                    evaluation_metrics = {
                        "Confusion_Matrix": confusion_matrix(y_test, y_pred).tolist(),
                        "Classification_Report": classification_report(y_test, y_pred, output_dict=True),
                        "MAE": mean_absolute_error(y_test, y_pred),
                        "MSE": mean_squared_error(y_test, y_pred),
                        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
                    }

                    model_identifier = f"{model_class_name}_{scaler_name}"
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
        'parameter_range': list(range(5, 10, 5))
    },
    {
        'class_name': "LogisticRegression",
        'model_class': LogisticRegression
    },
    {
        'class_name': "GradientBoostingRegressor",
        'model_class': GradientBoostingRegressor,
        'tuning_parameter': 'n_estimators',
        'parameter_range': list(range(50, 100, 50)),
        'random_state': 42
    }
]

# Define scaler configurations
scaler_configs = [
    {"scaler_name": "MinMaxScaler", 'scaler_class': MinMaxScaler},
    # {"scaler_name": "StandardScaler", 'scaler_class': StandardScaler},
    # {"scaler_name": "RobustScaler", 'scaler_class': RobustScaler}
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
df_scores = cross_validate_models(model_configurations, scaler_configs, data_set_train, target, cv=5, fit_scaler=True, scoring_type=scoring_type)

x = 1