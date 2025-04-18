import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_curve, auc, make_scorer, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, label_binarize
import os
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import loguniform, randint
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# Load data from Parquet file
input = '5'  # Hardcoded for testing
dfDirectory = os.path.normpath('Database\\files' + '\\' + input +'savesDatabase.parquet')
df = pd.read_parquet(dfDirectory)




# Setting up
# Define the target variable: potential ability range (intervals of width 10). If PA is 200, we need to make it 190, as I will allow that group to be bigger
df['PA_Intervals_Min'] = (df['PA'] // 10) * 10

# In case I keep forgetting. In the next line we create a boolean filter (true if PA=200). 
# We send it to loc so that it selects that row and the column specified after ('PA_Intervals_Min')
# We then change the value to 190
df.loc[df['PA_Intervals_Min'] == 200, 'PA_Intervals_Min'] = 190 




# Unimportant outlier elimination. We take out values that are very low where there are very few individuals as they are not of interest
df = df.groupby('PA_Intervals_Min').filter(lambda x: 
    not ((x.name < 120) and (len(x) < 30)))





# Select features and target
features = [col for col in df.columns if col not in ['UID', 'Name', 'PA', 'PA_Intervals_Min']]
X = df[features]
y = df['PA_Intervals_Min']







# Preprocessing
# Separate numeric and non-numeric columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

# Fill missing values
# For numeric columns, fill with a placeholder (e.g., -1)
X.loc[:, numeric_cols] = X[numeric_cols].fillna(-1)

# For non-numeric columns, fill with 'Missing'
X.loc[:, categorical_cols] = X[categorical_cols].fillna('Missing')

# 2. Encode categorical data
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore') 
encoded_categorical = encoder.fit_transform(X[categorical_cols])

# Get feature names for encoded categorical data6
encoded_feature_names = encoder.get_feature_names_out(categorical_cols)

# Convert encoded_categorical to a DataFrame
encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=encoded_feature_names) 

# Combine numeric and encoded categorical data
X_processed = pd.concat([
    encoded_categorical_df,
    X[numeric_cols].reset_index(drop=True)
], axis=1)
#Ensure all columns are float
X_processed = X_processed.astype(float)






# Feature engineering. Created a list of the 25 most influential features. Filter out any not included
columnsToKeep= ['AP', 'CA', 'Age', 'minPotential', 'maxPotential', 'Imp M', 'Sta', 'Det', 'Ldr', 'WR', 'Bra', 'Str', 'HR', 'Temp', 'Wor', 'Wage', 'Dirt', 'Inj Pr', 'AT Apps', 'CR', 'Prof', 'Fir', 'Amb', 'Min WD', 'Hea']
X_processed = X_processed.loc[:, columnsToKeep]






# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)






#Sampling techniques
print("Original training dataset shape:", y_train.value_counts())

# Calculate sampling strategy for undersampling
# Undersample majority classes (those with more than X samples)
# The strategy is to determine a threshold and reduce all classes above it
class_counts = Counter(y_train)
median_count = np.median(list(class_counts.values()))

# Undersample classes with more than twice the median count
undersample_strategy = {k: min(v, int(median_count*2)) for k, v in class_counts.items()}

# Apply undersampling
undersampler = RandomUnderSampler(sampling_strategy=undersample_strategy, random_state=42)
X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)

# Calculate oversampling strategy more conservatively for minority classes
min_sample_count = min(class_counts.values())
# For very small classes, don't oversample too aggressively
# Set a cap on how much we'll oversample (3x the original size at most)
oversample_strategy = {k: min(int(min_sample_count*3), int(median_count/3)) 
                      for k, v in Counter(y_train_under).items() 
                      if v < min_sample_count*2}

# Apply SMOTE for oversampling (only on classes that need it)
if oversample_strategy:
    oversampler = SMOTE(sampling_strategy=oversample_strategy, random_state=42)
    X_train, y_train = oversampler.fit_resample(X_train_under, y_train_under)
else:
    X_train, y_train = X_train_under, y_train_under

# Print the resampled class distribution
print("Balanced training dataset shape:", y_train.value_counts())






# Initialize and train the Random Forest model.We give it a seed to ensure reproducibility. Using class_weight='balanced' as db is imbalanced
model = RandomForestClassifier(bootstrap= False, max_depth= 23, max_features= 'log2', min_samples_leaf= 2, min_samples_split= 9, n_estimators= 532)   
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Model original accuracy:", accuracy)
f1_macro=f1_score(y_test, y_pred, average='macro')
print(f"Model original F1_Macro: {f1_macro:.4f}")


'''
#Hyperparameter tuning
param_dist = {
    # n_estimators: Number of trees from 20 to 1000, defines the number of different trees in the random forest. 
    # I want to try values at random but I use log_uniform as I don't want it to focus just on huge values
    # As it is made of floats, we sample 100 of them and turn them into integers   
    'n_estimators': [int(x) for x in loguniform(15, 1001).rvs(100)],
    'max_depth': randint(2, 51),                    # Depth from 2 to 50. Controls the maximum depth of each decision tree in the forest
    'min_samples_split': randint(2, 21),            # Min samples to split. Minimum number of samples required in a node for the algorithm to consider splitting it further. Low generates overfitting, high underfitting 
    'min_samples_leaf': randint(1, 21),             # Min samples per leaf. Sets the minimum of samples that can be in a leaf of a tree. It stops the creation of trivial leaves, reducing overfitting
    'max_features': ['sqrt', 'log2'],               # Still use categorical options here
    'bootstrap': [True, False]                      # Use or not use bootstrap samples.
}


rf = RandomForestClassifier(random_state=42, class_weight='balanced')

random_search = RandomizedSearchCV(
    estimator=rf,                                   # We are estimating a random forest
    param_distributions=param_dist,                 # Previously set parameters to check
    n_iter=1000,                                    # Number of random combinations to try
    cv=5,                                           # Cross-validation folds.Data will be split into 5 parts (folds). For each hyperparameter combo, the model trains on 4 folds and validates on the remaining one, repeating this 5 times.
    verbose=1,                                      # Level of logging output during fitting. 2 shows progress of every fold + combination. 1: Minimal info (one line per candidate)
    n_jobs=-1,                                      # Default value, means use every CPU core avaliable                            
    # Other scoring options:
    # - accuracy: Not that useful as database is really unbalanced (model can cheat by favoring more frequent classes)
    # - f1_micro: Better than accuracy but we don't use it as we care about outliers (e.g., 190+ players)
    # We use f1_macro as we want the model to perform well across all classes, even rare ones
    scoring='f1_macro',
    random_state=42                                 # Ensures reproducibility
)


random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_
print("Best Parameters:", random_search.best_params_)

y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1_macro=f1_score(y_test, y_pred, average='macro')
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Macro: {f1_macro:.4f}")
f1_micro=f1_score(y_test, y_pred, average='micro')
print(f"F1-Micro: {f1_micro:.4f}")


interval_labels = [f"{int(i)}-{int(i)+10}" for i in np.unique(y)]
'''


'''
# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix with interval labels
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=interval_labels, yticklabels=interval_labels)
plt.xlabel('Predicted Intervals')
plt.ylabel('Actual Intervals')
plt.title('Confusion Matrix with Interval Labels')
plt.show()
'''






#Statistics



#Force panda to show every column and row
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)




# 1. Confusion Matrix
interval_labels = [f"{int(i)}-{int(i)+10}" for i in np.unique(y)]

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix with interval labels
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=interval_labels, yticklabels=interval_labels)
plt.xlabel('Predicted Intervals')
plt.ylabel('Actual Intervals')
plt.title('Confusion Matrix with Interval Labels')
plt.show()





# 2. Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=np.nan))





# 3. Feature Importance
feature_importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X_processed.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("Feature Importance:")
print(feature_importance_df)

# Plot feature importance
plt.figure()
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15))
plt.title('Top 15 Feature Importances')
plt.show()





# 4. Cross-Validation Scores
scoring = {
    'accuracy': 'accuracy',
    'f1_micro': make_scorer(f1_score, average='micro', zero_division=np.nan),
    'f1_macro': make_scorer(f1_score, average='macro', zero_division=np.nan),
    'precision_macro': make_scorer(precision_score, average='macro', zero_division=np.nan),
    'recall_macro': make_scorer(recall_score, average='macro', zero_division=np.nan)
}


cv_results = cross_validate(model, X_train, y_train, cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=42), scoring=scoring) #Using statified k fold to ensure balance in between folds

print("\nCross validating scores")
for metric in scoring:
    scores = cv_results[f'test_{metric}']
    print(f'{metric}: {scores.mean():.2f} (± {scores.std():.2f})')






# 5. ROC validation
# Binarize the output for multi-class ROC
y_test_bin = label_binarize(y_test, classes=np.unique(y))
y_pred_proba = model.predict_proba(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = y_test_bin.shape[1]

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure(figsize=(10, 8))
colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of {interval_labels[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curves for PA Interval Prediction')
plt.legend(loc="lower right", bbox_to_anchor=(1.4, 0))
plt.grid(alpha=0.3)
plt.show()

# Compute micro-average ROC curve and AUC
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and AUC
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot micro and macro averages
plt.figure(figsize=(8, 6))
plt.plot(fpr["micro"], tpr["micro"],
         label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.2f})',
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label=f'Macro-average ROC (AUC = {roc_auc["macro"]:.2f})',
         color='navy', linestyle=':', linewidth=4)

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Micro and Macro-average ROC Curves')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

# Print AUC scores
print("\nROC AUC Scores:")
for i in range(n_classes):
    print(f"{interval_labels[i]}: {roc_auc[i]:.4f}")

print(f"\nMicro-average ROC AUC: {roc_auc['micro']:.4f}")
print(f"Macro-average ROC AUC: {roc_auc['macro']:.4f}")

