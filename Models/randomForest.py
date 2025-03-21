import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import os
import numpy as np
import time

# Load data from Parquet file
input = '2'  # Hardcoded for testing
dfDirectory = os.path.normpath(r'Database\files\2Export\data' + '\\' + input + 'saveFinal.parquet')
df = pd.read_parquet(dfDirectory)

# Define the target variable: potential ability range (intervals of width 5)
df['PA_Intervals_Min'] = (df['PA'] // 10) * 10

# Select features and target
features = [col for col in df.columns if col not in ['UID', 'Name', 'PA', 'PA_Intervals_Min']]
X = df[features]
y = df['PA_Intervals_Min']

# Preprocessing
# 1. Handle missing values
# Separate numeric and non-numeric columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

# Fill missing values
# For numeric columns, fill with a placeholder (e.g., -1)
X.loc[:, numeric_cols] = X[numeric_cols].fillna(-1)

# For non-numeric columns, fill with 'Missing'
X.loc[:, categorical_cols] = X[categorical_cols].fillna('Missing')

# 2. Encode categorical data
encoder = OneHotEncoder(handle_unknown='ignore')  # Removed `sparse=False`
encoded_categorical = encoder.fit_transform(X[categorical_cols])

# Get feature names for encoded categorical data
encoded_feature_names = encoder.get_feature_names_out(categorical_cols)

# Convert encoded_categorical to a DataFrame
encoded_categorical_df = pd.DataFrame(encoded_categorical.toarray(), columns=encoded_feature_names)

# Combine numeric and encoded categorical data
X_processed = pd.concat([
    encoded_categorical_df,
    X[numeric_cols].reset_index(drop=True)
], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Model original accuracy:", accuracy)


#Trying different parameters and seing their impact
'''
Should try again after adding more cases. Did not see a clear difference from 50 onwards, will keep it as 100 until I expand the database
n_Values=[50, 500, 5000]
for value in n_Values:
    start=time.time()
    model= RandomForestClassifier(n_estimators=value, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    elapsed= time.time()- start
    print("Model accuracy with", value,"as n:", accuracy, "in", elapsed)

'''

'''
# 1. Confusion Matrix
interval_labels = [f"{int(i)}-{int(i)+5}" for i in np.unique(y)]

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
print(classification_report(y_test, y_pred))

# 3. Feature Importance
feature_importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X_processed.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("Top 10 Most Important Features:")
print(feature_importance_df.head(30))

# Plot feature importance
plt.figure(figsize=(10, 7))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
plt.title('Top 10 Feature Importances')
plt.show()
'''



'''
# 4. Cross-Validation Scores
cv_scores = cross_val_score(model, X_processed, y, cv=5, scoring='accuracy')
print(f'Cross-Validation Accuracy: {cv_scores.mean():.2f} (± {cv_scores.std():.2f})')


# 5. ROC Curve and AUC (for binary classification only)
if len(y.unique()) == 2:  # Check if it's a binary classification problem
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
'''
print(X_processed.shape)