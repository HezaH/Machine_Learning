import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, 
    classification_report, mean_absolute_error, mean_squared_error
)
import os

def calculate_metrics(tn, fp, fn, tp):
    # Calculate common metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        'Sensitivity (TPR)': sensitivity,
        'Specificity (TNR)': specificity,
        'Precision': precision,
        'F1 Score': f1,
        'Accuracy': accuracy,
        'TP': tp, 
        'FP': fp, 
        'TN': tn, 
        'FN': fn
    }

# Function to find the optimal threshold based on cost ratio
def find_optimal_cost_threshold(y_true, y_scores, cost_ratio, thresholds=None):
    """
    Find the optimal threshold that minimizes the total cost.

    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_scores : array-like
        Target scores (probability estimates of the positive class)
    cost_ratio : float
        Cost ratio of false negatives to false positives (CFN/CFP)
        e.g., cost_ratio=2 means false negatives are twice as costly as false positives
    thresholds : array-like, optional
        Thresholds to evaluate, if None, 100 evenly spaced thresholds will be used

    Returns:
    --------
    optimal_threshold : float
        The threshold that minimizes the total cost
    total_costs : array-like
        Total cost at each threshold
    evaluated_thresholds : array-like
        The thresholds that were evaluated
    """
    if thresholds is None:
        # Create an array of thresholds to evaluate
        evaluated_thresholds = np.linspace(0.01, 0.99, 100)
    else:
        evaluated_thresholds = thresholds

    total_costs = []

    n_samples = len(y_true)
    n_pos = np.sum(y_true)
    n_neg = n_samples - n_pos

    for threshold in evaluated_thresholds:
        # Make predictions using the threshold
        y_pred = (y_scores >= threshold).astype(int)

        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        confusionMatrix = np.array([[tn, fp], [fn, tp]])

        # Calculate costs
        # Cost of false positives (normalized by the number of negatives)
        cost_fp = fp / n_neg if n_neg > 0 else 0
        # Cost of false negatives (normalized by the number of positives)
        cost_fn = fn / n_pos if n_pos > 0 else 0

        # Calculate total cost
        # FN costs are weighted by the cost ratio
        total_cost = cost_fp + cost_ratio * cost_fn

        total_costs.append(total_cost)

    # Find the threshold that minimizes the total cost
    optimal_idx = np.argmin(total_costs)
    optimal_threshold = evaluated_thresholds[optimal_idx]

    return optimal_threshold, np.array(total_costs), evaluated_thresholds, confusionMatrix

# Find threshold indices on the ROC curve for visualization
def find_closest_threshold_idx(thresholds, target_threshold):
    return np.abs(thresholds - target_threshold).argmin()

# Function to evaluate and plot confusion matrices
def plot_confusion_matrices(calculateMetrics, cost_scenarios, optimal_thresholds):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (name, cost_ratio) in enumerate(cost_scenarios.items()):
        if i >= len(axes):
            break

        threshold = optimal_thresholds[name]["threshold"]

        cm = calculateMetrics

        # Compute normalized confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Display confusion matrix
        im = axes[i].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[i].set_title(f"{name}\nThreshold = {threshold:.3f}")

        # Add class labels
        axes[i].set_xticks([0, 1])
        axes[i].set_yticks([0, 1])
        axes[i].set_xticklabels(['Negative', 'Positive'])
        axes[i].set_yticklabels(['Negative', 'Positive'])
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')

        # Add text annotations
        for r in range(2):
            for c in range(2):
                axes[i].text(c, r, f"{cm[r, c]}\n({cm_norm[r, c]:.2%})",
                           ha="center", va="center",
                           color="white" if cm[r, c] > cm.max()/2 else "black")

    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    return fig

#Setting the names of the columns
name_columns = ["ESCT","NDEP","RENDA","TIPOR","VBEM","NPARC","VPARC","TEL","IDADE","RESMS","ENTRADA", "CLASSE"]
target_column = "CLASSE"

#Loading the datasets
df_test = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'credtest.txt'), sep='\t', header=None)
# df_test = pd.read_csv('credtest.txt', sep='\t', header=None)

df_train = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'credtrain.txt'), sep='\t', header=None)
# df_train = pd.read_csv('credtrain.txt', sep='\t', header=None)

df_train.columns, df_test.columns = name_columns, name_columns

X_test  = df_test.drop(target_column, axis=1)
y_test  = df_test[target_column]
X_train = df_train.drop(target_column, axis=1)
y_train = df_train[target_column]

list_data = [set(df_train[i].to_list()) for i in df_train.columns]

for i in list_data:
    print(f"Column: {df_train.columns[list_data.index(i)]}, Min: {min(i)}, Max: {max(i)} " )

dict_classification = {
    "NormalizedData": [],
    "StandardData": [],
    "RobustData": []
}
#fit transorming use on data train and transforming on data test because not happend the data leakage
#Standardization is not recommended for data with outliers
#1. Normalization: scales the data to a range of [0, 1]
scaler_norm = MinMaxScaler()
norm_data_Xtrain = scaler_norm.fit_transform(X_train)
norm_data_Xtest = scaler_norm.transform(X_test)

# #2. Standardization: scales the data to a mean of 0 and a standard deviation of 1   
scaler_standard = StandardScaler()
standard_data_Xtrain = scaler_standard.fit_transform(X_train)
standard_data_Xtest = scaler_standard.transform(X_test)

# #3. RobustScaler: scales the data using statistics that are robust to outliers
scaler_robuster = RobustScaler()
robust_data_Xtrain = scaler_robuster.fit_transform(X_train)
robust_data_Xtest = scaler_robuster.transform(X_test)

#1. Normalization: scales the data to a range of [0, 1]
# X_train_norm_data, X_test_norm_data, y_train_norm_data, y_test_norm_data = train_test_split(norm_data_Xtrain,y_train,
#                                                     test_size=1)
X_train_norm_data, y_train_norm_data = norm_data_Xtrain, y_train
X_test_norm_data, y_test_norm_data = norm_data_Xtest, y_test
list_norm = [X_train_norm_data, X_test_norm_data, y_train_norm_data, y_test_norm_data, "NormalizedData"]

#2. Standardization: scales the data to a mean of 0 and a standard deviation of 1   
# X_train_standard_data, X_test_standard_data, y_train_standard_data, y_test_standard_data_Xtrain = train_test_split(standard_data_Xtrain,y_train,
# #                                                     test_size=1)
X_train_standard_data, y_train_standard_data = standard_data_Xtrain, y_train
X_test_standard_data, y_test_standard_data = standard_data_Xtest, y_test
list_standard = [X_train_standard_data, X_test_standard_data, y_train_standard_data, y_test_standard_data, "StandardData"]

# #3. RobustScaler
# # X_train_robust_data, X_test_robust_data, y_train_robust_data, y_test_robust_data = train_test_split(robust_data_Xtrain,y_train,
# #                                                     test_size=1)
X_train_robust_data, y_train_robust_data = robust_data_Xtrain, y_train
X_test_robust_data, y_test_robust_data = robust_data_Xtest, y_test
list_robust = [X_train_robust_data, X_test_robust_data, y_train_robust_data, y_test_robust_data, "RobustData"]

# Logistic Regression for each preprocessing method
# Dictionary to store predictions and residuals for combined plots
predictions = {}
residuals = {}
# Create a table to compare metrics for different cost scenarios
results = {}
calculateMetrics = []
# Define different cost scenarios
cost_scenarios = {
    "Equal Costs (FN = FP)": 1.0,
    "FN twice as costly as FP": 1.0,
    "FN five times as costly as FP": 1.0,
    "FP twice as costly as FN": 1.0,
    "FP five times as costly as FN": 1.0,
}

# Create visualization
plt.figure(figsize=(15, 10))
# 2) ROC and Precision‑Recall Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Logistic Regression for each preprocessing method
# for list_data in [list_norm, list_standard, list_robust]:
for list_data in [list_norm]:
    
    lm = LogisticRegression()
    
    X_train_model, X_test_model, y_train_model, y_test_model, method = list_data
    print(f"Method: {method}")
    
    # Fit the model to the training data
    lm.fit(X_train_model, y_train_model)

    # Display coefficients
    print('Coefficients: \n', lm.coef_)
    
    # Make predictions
    pred = lm.predict(X_test_model)
    
    # Get probability scores for the positive class
    y_scores = lm.predict_proba(X_test)[:, 1]

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)

    # Calculate AUC (Area Under Curve)
    roc_auc = auc(fpr, tpr)
    
    ax1.plot(fpr, tpr, label=f"{method} (AUC={roc_auc:.2f})")
    
    # PR
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    ap = average_precision_score(y_test, y_scores)
    
    # Find optimal thresholds for each cost scenario
    optimal_cost_thresholds = {}
    for name, cost_ratio in cost_scenarios.items():
        opt_threshold, costs, eval_thresholds, confusionMatrix = find_optimal_cost_threshold(
            y_test, y_scores, cost_ratio, thresholds
        )
        optimal_cost_thresholds[name] = {
            "threshold": opt_threshold,
            "costs": costs,
            "eval_thresholds": eval_thresholds
        }

    threshold_indices = {}
    for name, result in optimal_cost_thresholds.items():
        threshold_indices[name] = find_closest_threshold_idx(thresholds, result["threshold"])

    # Plot ROC curve with optimal thresholds
    plt.subplot(2, 2, 1)
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'{method} - ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax2.plot(recall, precision, label=f"{method} (AP={ap:.2f})")

    # Colors for different cost scenarios
    colors = ['red', 'green', 'purple', 'orange', 'brown']

    # Mark optimal points for different cost scenarios
    for i, (name, idx) in enumerate(threshold_indices.items()):
        plt.scatter(fpr[idx], tpr[idx], marker='o', color=colors[i],
                label=f"{name} (t={optimal_cost_thresholds[name]['threshold']:.3f})")

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve with Cost-Based Optimal Thresholds')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    # Format PR
    ax2.set_title("Precision-Recall Curves")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(loc='lower left')
    ax2.grid(True)

    cm = confusion_matrix(y_test_model, pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Plot confusion matrices for different cost scenarios
    plot_confusion_matrices(cm, cost_scenarios, optimal_cost_thresholds)
    
    # Create a table to compare metrics for different cost scenarios
    results = {}
    for name, data in optimal_cost_thresholds.items():
        threshold = data["threshold"]
        metrics = calculate_metrics(tn, fp, fn, tp)
        results[name] = {
            'Threshold': threshold,
            **metrics
        }

    # Print comparison table
    print("\nComparison of Different Cost-Based Thresholds:")
    print("-" * 100)
    metrics_to_display = ['Threshold', 'Sensitivity (TPR)', 'Specificity (TNR)',
                        'Precision', 'F1 Score', 'Accuracy']

    # Print header
    print(f"{'Cost Scenario':<25}", end=" ")
    for metric in metrics_to_display:
        print(f"{metric:<15}", end=" ")
    print()
    print("-" * 100)

    # Print rows
    for name, metrics in results.items():
        print(f"{name:<25}", end="")
        for metric in metrics_to_display:
            value = metrics[metric]
            if metric == 'Threshold':
                print(f"{value:<15.3f}", end="")
            else:
                print(f"{value:<15.3f}", end="")
        print()

    predictions[method] = (y_test_model, pred)
    residuals[method] = y_test_model - pred

    print('MAE:', round(mean_absolute_error(y_test_model, pred), 3))
    print('MSE:', round(mean_squared_error(y_test_model, pred), 3))
    print('RMSE:', round(np.sqrt(mean_squared_error(y_test_model, pred)), 3))
    print('\n' + '-'*80 + '\n')

plt.tight_layout()
plt.show()

# Plot all residual histograms in one figure
# Quando você vê um pico muito alto em resíduos 0, isso significa que muitas previsões foram exatas. Ja os picos em +1 (Falso Negativo) e -1 (Falso Positivo) correspondem a previsões erradas.
plt.figure(figsize=(12, 6))

for method, resid in residuals.items():
    sns.histplot(resid, bins=50, kde=True, label=method, stat='density', element='step')

plt.title('Residuals Distribution - Logistic Regression')
plt.xlabel('Residuals')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

x = 1






