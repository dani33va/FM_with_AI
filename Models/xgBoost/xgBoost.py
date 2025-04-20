import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_curve, auc, make_scorer, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, label_binarize, LabelEncoder
import os
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import loguniform, randint
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from xgboost import XGBClassifier
import pickle
from sklearn.utils.class_weight import compute_sample_weight
import optuna
from optuna.samplers import TPESampler





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
columnsToKeep= ['minPotential', 'maxPotential', 'AP', 'Min Fee Rls Clubs In Cont Comp', 'CA', 'Age', 'Wage Contrib.', 'Opt Ext by Club', 'Prof', 'AT Apps', 'Min Fee Rls', 'Yth Apps', 'Pun', 'Wage', 'HR', 'WR', 'Det', 'Consist', 'CR', 'Amb', 'Pen', 'Agi', 'Min Fee Rls to Foreign Clubs', 'Str', 'Cor']
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




# Preprocess the y values. Convert PA intervals to 0-based indices
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)



# Initialize and train the Random Forest model.We give it a seed to ensure reproducibility. Using class_weight='balanced' as db is imbalanced


'''
#Original model
model = XGBClassifier(
    random_state=42,
    eval_metric='mlogloss'
)

model.fit( X_train,y_train_encoded)
'''

 #final 1 model
'''
model = XGBClassifier(
    random_state=42,
    eval_metric='mlogloss',
    max_depth= 17,
    learning_rate= 0.027549770509215588,
    gamma= 0.7272621853811047,
    min_child_weight= 14,
    subsample= 0.9739578448716864,
    colsample_bytree= 0.8431140769235174,
    reg_alpha= 0.6327837343185934,
    reg_lambda= 1.5422800679934323,
    early_stopping_rounds= 50,
    n_estimators= 10000)

'''

#final 2 model
model = XGBClassifier(
    random_state=42,
    eval_metric='mlogloss',
    max_depth= 15,
    learning_rate= 0.27777024407880774,
    gamma= 0.3247146487151806,
    min_child_weight= 18,
    subsample= 0.9377790750541092,
    colsample_bytree=  0.5422539078768078,
    reg_alpha= 1.3120773052421253,
    reg_lambda= 4.460358741734858,
    early_stopping_rounds= 50,
    n_estimators= 10000)


sample_weights = compute_sample_weight('balanced', y_train_encoded)

model.fit(
    X_train,
    y_train_encoded,
    sample_weight=sample_weights,
    eval_set=[(X_test, y_test_encoded)],
    verbose=0
)



# Make predictions
y_pred_encoded = model.predict(X_test)
y_pred = label_encoder.inverse_transform(y_pred_encoded)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Model original accuracy:", accuracy)
f1_macro=f1_score(y_test, y_pred, average='macro')
print(f"Model original F1_Macro: {f1_macro:.4f}")

'''
#Hyperparameter tuning. Here we are aplying a bayesian optimization. It is more effective as it uses probability to guide the search.

# Compute sample weights for imbalanced classes
sample_weights = compute_sample_weight('balanced', y_train_encoded)

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.5, log=True),
        'gamma': trial.suggest_float('gamma', 0, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 20),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 20),
        'tree_method': 'hist',
        'device': 'cpu',
        'random_state': 42,
        'eval_metric': 'mlogloss',
        'early_stopping_rounds': 50
    }
    
    model = XGBClassifier(n_estimators= 10000,**params)
    model.fit(
        X_train,
        y_train_encoded,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test_encoded)],
        verbose=0
    )
    
    y_pred = model.predict(X_test)
    return f1_score(y_test_encoded, y_pred, average='macro')

# Set up Optuna study with pruning
study = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=10)
)

# Run optimization
study.optimize(objective, n_trials=300, show_progress_bar=True)

# Print best results
print("\n=== Best Trial ===")
trial = study.best_trial
print(f"Best F1 (Macro): {trial.value:.4f}")
print("Best Params:")
for key, value in trial.params.items():
    print(f"  {key}: {value}")

# Train final model with best params + weights
best_params = trial.params
best_params.update({
    'tree_method': 'hist',
    'device': 'cpu',
    'random_state': 42,
    'eval_metric': 'mlogloss',
    'early_stopping_rounds': 50
})

final_model = XGBClassifier(**best_params)
final_model.fit(
    X_train,
    y_train_encoded,
    sample_weight=sample_weights,
    eval_set=[(X_test, y_test_encoded)],
    verbose=True
)

# Replace your original model with the optimized one
model = final_model


# Make predictions
y_pred_encoded = model.predict(X_test)
y_pred = label_encoder.inverse_transform(y_pred_encoded)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Model optimized accuracy:", accuracy)
f1_macro=f1_score(y_test, y_pred, average='macro')
print(f"Model optimized F1_Macro: {f1_macro:.4f}")

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




'''
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

'''

'''
# 4. Cross-Validation Scores
scoring = {
    'accuracy': 'accuracy',
    'f1_micro': make_scorer(f1_score, average='micro', zero_division=np.nan),
    'f1_macro': make_scorer(f1_score, average='macro', zero_division=np.nan),
    'precision_macro': make_scorer(precision_score, average='macro', zero_division=np.nan),
    'recall_macro': make_scorer(recall_score, average='macro', zero_division=np.nan)
}


cv_results = cross_validate(model, X_train, y_train_encoded, cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=42), scoring=scoring) #Using statified k fold to ensure balance in between folds

print("\nCross validating scores")
for metric in scoring:
    scores = cv_results[f'test_{metric}']
    print(f'{metric}: {scores.mean():.2f} (± {scores.std():.2f})')

'''




# 5. ROC validation
# Binarize the output for multi-class ROC
y_test_bin = label_binarize(y_test_encoded, classes=np.unique(y_test_encoded))
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



modelDirectory= os.path.normpath('Models\\xgBoost\\XGBoostFiles\\_final\\all\\xgboost_model.pkl')
with open(modelDirectory, 'wb') as f:
    pickle.dump(model, f)
