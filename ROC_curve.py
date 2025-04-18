import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Set random seed for reproducibility
np.random.seed(42)

# Generate a synthetic binary classification dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                          n_redundant=5, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Get probability scores for the positive class
y_scores = clf.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

# Calculate AUC (Area Under Curve)
roc_auc = auc(fpr, tpr)


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

    return optimal_threshold, np.array(total_costs), evaluated_thresholds

# Define different cost scenarios
cost_scenarios = {
    "Equal Costs (FN = FP)": 1.0,
    "FN twice as costly as FP": 2.0,
    "FN five times as costly as FP": 5.0,
    "FP twice as costly as FN": 0.5,
    "FP five times as costly as FN": 0.2
}

# Find optimal thresholds for each cost scenario
optimal_cost_thresholds = {}
for name, cost_ratio in cost_scenarios.items():
    opt_threshold, costs, eval_thresholds = find_optimal_cost_threshold(
        y_test, y_scores, cost_ratio, thresholds
    )
    optimal_cost_thresholds[name] = {
        "threshold": opt_threshold,
        "costs": costs,
        "eval_thresholds": eval_thresholds
    }

    
# Find threshold indices on the ROC curve for visualization
def find_closest_threshold_idx(thresholds, target_threshold):
    return np.abs(thresholds - target_threshold).argmin()

threshold_indices = {}
for name, result in optimal_cost_thresholds.items():
    threshold_indices[name] = find_closest_threshold_idx(thresholds, result["threshold"])

# Create visualization
plt.figure(figsize=(15, 10))

# Plot ROC curve with optimal thresholds
plt.subplot(2, 2, 1)
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

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

# Plot cost curves
plt.subplot(2, 2, 2)
for i, (name, result) in enumerate(optimal_cost_thresholds.items()):
    plt.plot(result["eval_thresholds"], result["costs"], color=colors[i], label=f"{name}")
    plt.axvline(x=result["threshold"], color=colors[i], linestyle='--', alpha=0.5)

plt.xlabel('Threshold')
plt.ylabel('Total Cost')
plt.title('Cost vs. Threshold for Different Cost Ratios')
plt.legend(loc="best")
plt.grid(alpha=0.3)

# Function to evaluate and plot confusion matrices
def plot_confusion_matrices(y_true, y_scores, cost_scenarios, optimal_thresholds):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (name, cost_ratio) in enumerate(cost_scenarios.items()):
        if i >= len(axes):
            break

        threshold = optimal_thresholds[name]["threshold"]
        y_pred = (y_scores >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)

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

# Plot confusion matrices for different cost scenarios
plot_confusion_matrices(y_test, y_scores, cost_scenarios, optimal_cost_thresholds)

def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

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
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn
    }

# Create a table to compare metrics for different cost scenarios
results = {}
for name, data in optimal_cost_thresholds.items():
    threshold = data["threshold"]
    metrics = calculate_metrics(y_test, y_scores, threshold)
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
print(f"{'Cost Scenario':<25}", end="")
for metric in metrics_to_display:
    print(f"{metric:<15}", end="")
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

plt.show()