import os
import numpy as np
import pandas as pd
import seaborn as sns
import json
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImPipeline
from sklearn.metrics import (confusion_matrix, accuracy_score, classification_report, 
                             mean_absolute_error, mean_squared_error)
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss, ClusterCentroids, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek
import time

# Start time measurement
start_time = time.time()
current_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'A652.pickle')

f = open(current_dir, 'rb')
(X_train, y_train, X_val, y_val, X_test, y_test) = pickle.load(f) 

print(f"Shapes: {X_train.shape}, {X_test.shape}, {X_val.shape}")

# Transform continuous values into binary labels
y_train = np.where(y_train == 0, 0, 1)
y_val = np.where(y_val == 0, 0, 1)
y_test = np.where(y_test == 0, 0, 1)

# Define configuration variables
target = 'target'
scoring_type = 'accuracy'  # or 'f1', 'precision', 'recall'

# Convert arrays to DataFrames and Series, and create a single DataFrame for each set
df_train = pd.concat([pd.DataFrame(X_train), pd.Series(y_train.ravel(), name=target)], axis=1).reset_index(drop=True)
df_val = pd.concat([pd.DataFrame(X_val), pd.Series(y_val.ravel(), name=target)], axis=1).reset_index(drop=True)
df_test = pd.concat([pd.DataFrame(X_test), pd.Series(y_test.ravel(), name=target)], axis=1).reset_index(drop=True)

# Separate features (X) and labels (y) for each set
X_train_data = df_train.drop(columns=[target])
y_train_data = df_train[target].values

X_val_data = df_val.drop(columns=[target])
X_test_data = df_test.drop(columns=[target])

# Preprocessing and scaling configuration
scaler_config = {'scaler_name': 'StandardScaler'}
feature_columns = X_train_data.columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('scaler', StandardScaler(), feature_columns)
    ]
)

# Define model configurations; include tuning keys for models that require parameter tuning
model_configurations = [
    {
        'class_name': "GradientBoostingClassifier",
        'model_class': GradientBoostingClassifier,
        'tuning_parameter': 'learning_rate',
        'random_state': 42,
        'parameter_range': [0.1, 0.01, 0.001],
    }
]

# Sampling configurations
sampling_configs = [
    {"sampling_name": "SMOTE", 'sampling_class': SMOTE(random_state=42)},
    {"sampling_name": "ADASYN", 'sampling_class': ADASYN(random_state=42)},
    {"sampling_name": "RandomUnderSampler", 'sampling_class': RandomUnderSampler(random_state=42)},
    {"sampling_name": "NearMiss", 'sampling_class': NearMiss()},
    {"sampling_name": "ClusterCentroids", 'sampling_class': ClusterCentroids(random_state=42)},
    {"sampling_name": "TomekLinks", 'sampling_class': TomekLinks()},
    {"sampling_name": "SMOTEENN", 'sampling_class': SMOTEENN(random_state=42)},
    {"sampling_name": "SMOTETomek", 'sampling_class': SMOTETomek(random_state=42)},
    {"sampling_name": "Threshold", 'sampling_class': np.arange(0.1, 1.0, 0.1)},
    {"sampling_name": "Without", 'sampling_class': None},
]

results = []

# ----------------
# Main loop: For each model, for each sampling technique, and for each tuning parameter value…
for model_config in model_configurations:
    model_class_name = model_config['class_name']
    tuning_param = model_config['tuning_parameter']
    
    for sampling_config in sampling_configs:
        sampling_name = sampling_config['sampling_name']
        
        for param in model_config['parameter_range']:
            print(f"\nRunning: Model={model_config['class_name']}, {tuning_param}={param}, Sampling={sampling_name}")
            set_name = f"{model_class_name}&{tuning_param}&{param}&{sampling_name}"
            
            if sampling_name == "Threshold":
                # No sampling applied, train on the standard pipeline and then adjust the threshold
                model = model_config['model_class'](**{tuning_param: param}, random_state=model_config['random_state'])
                pipeline_model = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', model)
                ])
                pipeline_model.fit(X_train_data, y_train_data)
                # Get probabilities on the validation set
                y_proba = pipeline_model.predict_proba(X_val_data)[:, 1]
                
                for threshold in sampling_config['sampling_class']:
                    y_pred = (y_proba >= threshold).astype(int)
                    cm = confusion_matrix(df_val[target], y_pred).tolist()
                    acc = accuracy_score(df_val[target], y_pred)
                    print(f"  Threshold={threshold:.1f} -> Accuracy: {acc:.4f}")
                    class_report = classification_report(df_val[target], y_pred, output_dict=True)
                    results.append({
                        "Model": set_name,
                        "Accuracy": acc,
                        'sampling': sampling_name,
                        'scaler': scaler_config['scaler_name'],
                        "threshold": threshold,
                        "Confusion_Matrix": cm,
                        "Tp": cm[0][0],
                        "Fp": cm[0][1],
                        "Fn": cm[1][0],
                        "Tn": cm[1][1],
                        "Classification_Report": class_report,
                        "MAE": mean_absolute_error(df_val[target], y_pred),
                        "MSE": mean_squared_error(df_val[target], y_pred),
                        "RMSE": np.sqrt(mean_squared_error(df_val[target], y_pred)),
                        "Y": df_val[target].tolist(),
                        "YPred": y_pred.tolist()
                    })
            else:       
                if sampling_name != "Without":
                    # For balancing techniques (except Threshold), use an imblearn pipeline
                    sampler = sampling_config['sampling_class']
                    model = model_config['model_class'](**{tuning_param: param}, random_state=model_config['random_state'])
                    pipeline_model = ImPipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('sampler', sampler),
                        ('classifier', model)
                    ])
                else:
                    # Without any balancing (technique 'Without')
                    model = model_config['model_class'](**{tuning_param: param}, random_state=model_config['random_state'])
                    pipeline_model = Pipeline([
                        ('preprocessor', preprocessor),
                        ('classifier', model)
                    ])

                pipeline_model.fit(X_train_data, y_train_data)
                y_pred = pipeline_model.predict(X_val_data)
                cm = confusion_matrix(df_val[target], y_pred).tolist()
                acc = accuracy_score(df_val[target], y_pred)
                class_report = classification_report(df_val[target], y_pred, output_dict=True)
                
                results.append({
                    "Model": set_name,
                    "Accuracy": acc,
                    'sampling': sampling_name,
                    'scaler': scaler_config['scaler_name'],
                    "threshold": None,
                    "Confusion_Matrix": cm,
                    "Tp": cm[0][0],
                    "Fp": cm[0][1],
                    "Fn": cm[1][0],
                    "Tn": cm[1][1],
                    "Classification_Report": class_report,
                    "MAE": mean_absolute_error(df_val[target], y_pred),
                    "MSE": mean_squared_error(df_val[target], y_pred),
                    "RMSE": np.sqrt(mean_squared_error(df_val[target], y_pred)),
                    "Y": df_val[target].tolist(),
                    "YPred": y_pred.tolist()
                })    
            print("\n")

# Results DataFrame and best configuration selection
df_results = pd.DataFrame(results)
results_json_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results_ex3.json')
df_results.to_json(results_json_path, orient='records', lines=True)

# Select the best configuration among all tested (here we use Accuracy; if AUC is available, it can be changed)
best_result = max(results, key=lambda x: x["Classification_Report"].get("1", {}).get("f1-score", 0))
print("------------------------------------------------------")
print("Best configuration obtained:")
print(best_result["Model"])
print("F1 score for class 1:", best_result["Classification_Report"].get("1", {}).get("f1-score", 0))
print("Threshold:", best_result["threshold"])

final_result_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'best_result.json')
with open(final_result_path, 'w', encoding='utf-8') as f:
    json.dump(best_result, f, ensure_ascii=False, indent=4)

# For final training, merge X_train_data and X_val_data:
df_train_merged = pd.concat([df_train, df_val]).reset_index(drop=True)
X_train_merged = df_train_merged.drop(columns=[target])
y_train_merged = df_train_merged[target].values

# Get the components of the string identifying the configuration
set_name = best_result["Model"]   # Ex.: "GradientBoostingClassifier&learning_rate&0.1&SMOTE"
components = set_name.split("&")
model_class_str = components[0]
tuning_param = components[1]
tuning_value = float(components[2])
sampling_name = components[3]

# Model mapping (here we only have GradientBoostingClassifier, but if there are more, add them)
model_class = model_configurations[0]['model_class']

if sampling_name == "Threshold":
    best_threshold = best_result["threshold"]
    model = model_class(**{tuning_param: tuning_value}, random_state=42)
    pipeline_best = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipeline_best.fit(X_train_merged, y_train_merged)
    y_proba_test = pipeline_best.predict_proba(X_test_data)[:, 1]
    y_pred_test = (y_proba_test >= best_threshold).astype(int)
elif sampling_name != "Without":
    sampler_mapping = {
        "SMOTE": SMOTE(random_state=42),
        "ADASYN": ADASYN(random_state=42),
        "RandomUnderSampler": RandomUnderSampler(random_state=42),
        "NearMiss": NearMiss(),
        "ClusterCentroids": ClusterCentroids(random_state=42),
        "TomekLinks": TomekLinks(),
        "SMOTEENN": SMOTEENN(random_state=42),
        "SMOTETomek": SMOTETomek(random_state=42)
    }
    sampler = sampler_mapping[sampling_name]
    model = model_class(**{tuning_param: tuning_value}, random_state=42)
    pipeline_best = ImPipeline(steps=[
        ('preprocessor', preprocessor),
        ('sampler', sampler),
        ('classifier', model)
    ])
    pipeline_best.fit(X_train_merged, y_train_merged)
    y_pred_test = pipeline_best.predict(X_test_data)
else:
    model = model_class(**{tuning_param: tuning_value}, random_state=42)
    pipeline_best = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipeline_best.fit(X_train_merged, y_train_merged)
    y_pred_test = pipeline_best.predict(X_test_data)

cm_final = confusion_matrix(df_test[target], y_pred_test).tolist()
acc_final = accuracy_score(df_test[target], y_pred_test)
class_report_final = classification_report(df_test[target], y_pred_test, output_dict=True)

final_result = {
    "Model": set_name,
    "Accuracy": acc_final,
    "threshold": best_result["threshold"] if sampling_name == "Threshold" else None,
    "Confusion_Matrix": cm_final,
    "Classification_Report": class_report_final,
}

print("\nFinal evaluation on the test set:")
print("Accuracy:", acc_final)
print("Confusion Matrix:", cm_final)
print("Classification Report:", class_report_final)

plot_results("final", final_result)

# End time measurement
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTotal processing time: {elapsed_time:.2f} seconds")
