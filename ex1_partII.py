import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (roc_curve, roc_auc_score, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, f1_score,
                             brier_score_loss)
import seaborn as sns

def plot_calibration_curve(y_test, y_prob, plot_title, method_label="", n_bins=15):
    disp = CalibrationDisplay.from_predictions(y_test, 
                                               y_prob, 
                                               n_bins=n_bins,
                                               color='red',  
                                               marker='o',  
                                               markersize=4,  
                                               markeredgecolor='red',  
                                               markerfacecolor='red', 
                                               label=method_label,  
                                               )

    disp.ax_.set_title(plot_title)

    plt.grid(True)
    plt.show()

def add_calibration_curve(ax, y_true, y_prob, n_bins=15, label=""):
    # Generate calibration display from predictions and true labels
    disp = CalibrationDisplay.from_predictions(y_true, y_prob, n_bins=n_bins, ax=ax, label=label)
    return disp


def compute_brier_score(y_true, y_prob):
    """Calculate the Brier Score."""
    return brier_score_loss(y_true, y_prob)

def compute_ece(y_true, y_prob, n_bins=10):
    """Calculate the Expected Calibration Error (ECE)."""
    y_true = np.asarray(y_true)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    total = len(y_prob)
    ece = 0.0
    for i in range(n_bins):
        indices = np.where(binids == i)[0]
        if len(indices) == 0:
            continue
        avg_prob = np.mean(y_prob[indices])
        avg_true = np.mean(y_true[indices])
        weight = len(indices) / total
        ece += weight * np.abs(avg_prob - avg_true)
    return ece

def compute_mce(y_true, y_prob, n_bins=10):
    """Calculate the Maximum Calibration Error (MCE)."""
    y_true = np.asarray(y_true)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    mce = 0.0
    for i in range(n_bins):
        indices = np.where(binids == i)[0]
        if len(indices) == 0:
            continue
        avg_prob = np.mean(y_prob[indices])
        avg_true = np.mean(y_true[indices])
        error = np.abs(avg_prob - avg_true)
        if error > mce:
            mce = error
    return mce

def evaluate_calibration(y_true, y_prob, n_bins=10):
    """
    Calculate calibration metrics: Brier Score, ECE, and MCE.
    Returns a dictionary with these metrics.
    """
    brier = compute_brier_score(y_true, y_prob)
    ece   = compute_ece(y_true, y_prob, n_bins=n_bins)
    mce   = compute_mce(y_true, y_prob, n_bins=n_bins)
    return {"Brier Score": brier, "ECE": ece, "MCE": mce}

def select_best_calibration(result_list):
    """
    Select the best calibration configuration based on
    (Brier Score + ECE). Returns the corresponding dictionary.
    """
    best_config = None
    best_score = float('inf')
    for res in result_list:
        score = res["Brier Score"] + res["ECE"]
        if score < best_score:
            best_score = score
            best_config = res
    return best_config


def plot_confusion_matrix_from_values(tp, tn, fp, fn, labels=['Negativo', 'Positivo'], title='Matriz de Confusão'):
    """
    Plota a matriz de confusão a partir dos valores TP, TN, FP, FN.

    Parâmetros:
    - tp: Verdadeiros Positivos
    - tn: Verdadeiros Negativos
    - fp: Falsos Positivos
    - fn: Falsos Negativos
    - labels: Lista com os rótulos das classes [Negativo, Positivo]
    - title: Título do gráfico
    """
    cm = np.array([[tn, fp],
                   [fn, tp]])

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Classe Predita')
    plt.ylabel('Classe Real')
    plt.title(title)
    plt.show()

# Start timing
start_time = time.time()

# ===========================
# DATA LOADING AND PREPROCESSING
# ===========================
# Define column names
column_names = ["Education", "Dependents", "Income", "PropertyType", "AssetValue", "Installments", "InstallmentValue", "Phone", "Age", "ResidenceMonths", "DownPayment", "Target"]
categorical_columns = ["Education", "Dependents", "PropertyType", "Phone"]  
numerical_columns = ["Income", "AssetValue", "Installments", "InstallmentValue", "Age", "ResidenceMonths", "DownPayment"]
target_column = "Target"  

# Load training and testing data
base_path = os.path.dirname(os.path.realpath(__file__))
df_train = pd.read_csv(os.path.join(base_path, 'credtrain.txt'), sep='\t', header=None)
df_test = pd.read_csv(os.path.join(base_path, 'credtest.txt'), sep='\t', header=None)

df_train.columns = column_names
df_test.columns = column_names

# Separate features and target
X_test, y_test = df_test.drop(target_column, axis=1), df_test[target_column]
X_train, y_train = df_train.drop(target_column, axis=1), df_train[target_column]

# Additional split: training and validation
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, train_size=.5)
df_train_split = pd.DataFrame(X_train_split, columns=column_names[:-1])
df_train_split[target_column] = y_train_split

df_val = pd.DataFrame(X_val, columns=column_names[:-1])
df_val[target_column] = y_val

# Create preprocessor using ColumnTransformer:
# - OneHotEncoder for categorical columns.
# - StandardScaler for numerical columns.
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_columns),
        ('num', StandardScaler(), numerical_columns)
    ]
)

# Fit the preprocessor with training data (features only) and apply it to partitions
X_train_split_transformed = preprocessor.fit_transform(X_train_split)
X_val_transformed = preprocessor.transform(X_val)
X_test_transformed = preprocessor.transform(X_test)

# Define model configurations; include tuning keys for models that require parameter tuning.
model_configurations = [
    {
        'class_name': "KNeighborsClassifier",
        'model_class': KNeighborsClassifier
    },
    {
        'class_name': "LogisticRegression",
        'model_class': LogisticRegression
    },
    {
        'class_name': "GradientBoostingClassifier",
        'model_class': GradientBoostingClassifier,
    }
]

calibration_methods = [
    {
        'calibration_name': "CalibClassCV_Prefit",
        'calibration_class': CalibratedClassifierCV,
        'params': {'cv': 'prefit'}
    },
    {
        'calibration_name': "CalibClassCV_5Fold",
        'calibration_class': CalibratedClassifierCV,
        'params': {'cv': 5}
    },
    {
        'calibration_name': "CalibClassCV_Prefit_Isotonic",
        'calibration_class': CalibratedClassifierCV,
        'params': {'cv': 'prefit',  'method': 'isotonic'}
    },
    {
        'calibration_name': "CalibClassCV_Prefit_Sigmoid",
        'calibration_class': CalibratedClassifierCV,
        'params': {'cv': 'prefit', 'method': 'sigmoid'}
    },
    {
        'calibration_name': "None",
        'calibration_class': None,
    },
]

results = []
# Lists to accumulate calibration curves for joint plotting (validation and test)
calibration_curves_val = []
calibration_curves_test = []

# Iterate through all models defined in model_configurations
for config in model_configurations:
    model_class_name = config['class_name']
    print(f"Processing model: {model_class_name}")
    
    # Create classifier pipeline with preprocessor
    clf = config['model_class']()
    pipeline_clf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    # Train the pipeline using training data (split)
    pipeline_clf.fit(X_train_split, y_train_split)
    
    # List to store calibration results for this model
    model_calibration_results = []
    
    # For each calibration configuration, fit or use the model without calibration
    for calib in calibration_methods:
        calib_name = calib['calibration_name']
        if calib['calibration_class'] is not None:
            # Instantiate the calibrator with the 'estimator' parameter
            calibrator = calib["calibration_class"](estimator=pipeline_clf, **calib["params"])
            # Calibrate using validation data
            calibrator.fit(X_val, y_val)
            used_model = calibrator
        else:
            used_model = pipeline_clf

        # Get predictions and probabilities for validation and test
        y_pred_val   = used_model.predict(X_val)
        y_proba_val  = used_model.predict_proba(X_val)[:, 1]
        y_pred_test  = used_model.predict(X_test)
        y_proba_test = used_model.predict_proba(X_test)[:, 1]
        
        # Calculate ROC curves and ROC AUC
        fpr_val, tpr_val, thresholds_val = roc_curve(y_val, y_proba_val)
        roc_auc_val = roc_auc_score(y_val, y_proba_val)
        
        closest_val = np.sqrt(fpr_val**2 + (1 - tpr_val)**2)
        gmeans_val = np.sqrt(tpr_val * (1 - fpr_val))
        young_val = tpr_val + (1 - fpr_val) - 1

        fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_proba_test)
        roc_auc_test = roc_auc_score(y_test, y_proba_test)
        
        closest_test = np.sqrt(fpr_test**2 + (1 - tpr_test)**2)
        gmeans_test = np.sqrt(tpr_test * (1 - fpr_test))
        young_test = tpr_test + (1 - fpr_test) - 1
        
        # Determine the best threshold on the youg
        ix_val = np.argmax(young_val)
        best_threshold_val = thresholds_val[ix_val]
        print(f"Best threshold (Val) for {model_class_name} with calibration '{calib_name}': {best_threshold_val:.4f} | Young: {young_test[ix_val]:.4f}")
        
        ix_test = np.argmax(young_test)
        best_threshold_test = thresholds_test[ix_test]
        print(f"Best threshold (Test) for {model_class_name} with calibration '{calib_name}': {best_threshold_test:.4f} | Young: {young_test[ix_test]:.4f}")
        
        # Plot confusion matrix for the test set (fixed threshold 0.5)
        cm = confusion_matrix(y_test, (y_proba_test >= 0.5).astype(int))
        tn, fp, fn, tp = cm.ravel()
        plot_confusion_matrix_from_values(tp, tn, fp, fn, title=f"{model_class_name} - Confusion Matrix (Test) - {calib_name}")
        
        # Accumulate data (labels and probabilities) for calibration curve plotting
        calibration_curves_val.append((y_val, y_proba_val, calib_name))
        calibration_curves_test.append((y_test, y_proba_test, calib_name))
        
        # Calculate performance metrics on the validation set
        accuracy  = accuracy_score(y_val, y_pred_val)
        precision = precision_score(y_val, y_pred_val, zero_division=0)
        recall    = recall_score(y_val, y_pred_val, zero_division=0)
        f1        = f1_score(y_val, y_pred_val, zero_division=0)
        
        # Calculate calibration metrics
        calibration_metrics = evaluate_calibration(y_val, y_proba_val, n_bins=15)
        
        result_item = {
            'class_name': model_class_name,
            'calibration': calib_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc_val': roc_auc_val,
            'roc_auc_test': roc_auc_test,
            'best_threshold': best_threshold_test,
        }
        # Merge calibration metrics into the result dictionary
        result_item.update(calibration_metrics)
        results.append(result_item)
        model_calibration_results.append(result_item)
    
    # Select the best calibration configuration for this model
    best_calibration_for_model = select_best_calibration(model_calibration_results)
    print(f"Best calibration for {model_class_name} is: {best_calibration_for_model['calibration']} with Brier: {best_calibration_for_model['Brier Score']:.4f}, ECE: {best_calibration_for_model['ECE']:.4f}")
    
    # --- Joint plotting of calibration curves for this model ---
    plt.figure(figsize=(8, 6))
    ax_val = plt.gca()
    
    for (y_true_val, y_prob_val, label_cal) in calibration_curves_val:
        disp = CalibrationDisplay.from_predictions(y_true_val, y_prob_val, n_bins=15, ax=ax_val, label=label_cal)
    ax_val.set_title(f"Calibration Curve (Validation) - {model_class_name}")
    ax_val.legend(loc="best")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    ax_test = plt.gca()
    
    for (y_true_test, y_prob_test, label_cal) in calibration_curves_test:
        disp = CalibrationDisplay.from_predictions(y_true_test, y_prob_test, n_bins=15, ax=ax_test, label=label_cal)
    ax_test.set_title(f"Calibration Curve (Test) - {model_class_name}")
    ax_test.legend(loc="best")
    plt.grid(True)
    plt.show()
    
    # Plot ROC curves for validation and test
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Baseline')
    plt.plot(fpr_val, tpr_val, label=f"Validation (AUC = {roc_auc_val:.2f})")
    plt.plot(fpr_test, tpr_test, label=f"Test (AUC = {roc_auc_test:.2f})")
    plt.scatter(fpr_val[ix_val], tpr_val[ix_val], marker='o', color='black', label=f"Best threshold (Val): {best_threshold_val:.2f}")
    plt.scatter(fpr_test[ix_test], tpr_test[ix_test], marker='x', color='black', label=f"Best threshold (test): {best_threshold_test:.2f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_class_name}")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # Clear lists for the next model
    calibration_curves_val.clear()
    calibration_curves_test.clear()

# Display overall results in a DataFrame
df_results = pd.DataFrame(results)
print(df_results)

# Select the best overall model based on test ROC AUC
best_model = max(results, key=lambda item: item['roc_auc_test'])
calibration_str = (f"Calibration: {best_model['calibration']}" if best_model['calibration'] != "None" else "No calibration")

print("------------------------------------------------------")
print("Best identified model:")
print(f"Model: {best_model['class_name']} / {calibration_str}")
print(f"Accuracy (Validation): {best_model['accuracy']:.4f}")
print(f"Precision (Validation): {best_model['precision']:.4f}")
print(f"Recall (Validation): {best_model['recall']:.4f}")
print(f"F1 Score (Validation): {best_model['f1']:.4f}")
print(f"ROC AUC (Validation): {best_model['roc_auc_val']:.4f}")
print(f"ROC AUC (Test): {best_model['roc_auc_test']:.4f}")
print(f"Brier Score: {best_model['Brier Score']:.4f}")
print(f"ECE: {best_model['ECE']:.4f}")
print(f"MCE: {best_model['MCE']:.4f}")

# End timing
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTotal processing time: {elapsed_time:.2f} seconds")
