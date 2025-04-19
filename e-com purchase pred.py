# E-Commerce Customer Purchase Prediction Analysis
'''This project analyzes online shopping behavior and predicts whether visitors will
make a purchase'''

#import lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')


# Load the dataset
df = pd.read_csv('online_shoppers_intention.csv')

print(df.head())
print(df.isnull().sum())

# Target variable distribution
revenue_distribution = df['Revenue'].value_counts(normalize=True) * 100
print(revenue_distribution)


# Create visualizations directory
import os
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')
    
# Plotting target variable distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Revenue', data=df)
plt.title('Distribution of Revenue (Purchase vs No Purchase)')
plt.xlabel('Made Purchase')
plt.ylabel('Count')
plt.savefig('visualizations/revenue_distribution.png')


# Explore relationship between visitor type and purchases
plt.figure(figsize=(10, 6))
sns.countplot(x='VisitorType', hue='Revenue', data=df)
plt.title('Purchase Rate by Visitor Type')
plt.xlabel('Visitor Type')
plt.ylabel('Count')
plt.savefig('visualizations/purchase_by_visitor_type.png')


# Explore seasonal patterns
plt.figure(figsize=(12, 6))
monthly_purchase = pd.crosstab(df['Month'], df['Revenue'])
monthly_purchase_pct = monthly_purchase.div(monthly_purchase.sum(axis=1), axis=0) * 100
monthly_purchase_pct[True].sort_values().plot(kind='line', figsize=(12, 6))
monthly_purchase_pct[True].sort_values().plot(kind='bar', figsize=(12, 6))
plt.title('Purchase Rate by Month')
plt.xlabel('Month')
plt.ylabel('Purchase Rate (%)')
plt.savefig('visualizations/purchase_rate_by_month.png')

#Weekend vs. Weekday analysis
weekend_conversion = df.groupby('Weekend')['Revenue'].mean() * 100
print(weekend_conversion)

plt.figure(figsize=(10, 6))
sns.countplot(x='Weekend_conversion', hue='Revenue', data=df)
plt.title('Purchase Rate: Weekend vs. Weekday')
plt.xlabel('Weekend')
plt.ylabel('Count')
plt.savefig('visualizations/weekend_vs_weekday.png')
plt.show()

# Correlation analysis for numeric features
plt.figure(figsize=(14, 10))
numeric_df = df.select_dtypes(include=['float64', 'int64'])
correlation = numeric_df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numeric Features')
plt.tight_layout()
plt.savefig('visualizations/correlation_matrix.png')


#FEATURE ENGINEERING
# Create new features
df['TotalDuration'] = df['Administrative_Duration'] + df['Informational_Duration'] + df['ProductRelated_Duration']
df['TotalPages'] = df['Administrative'] + df['Informational'] + df['ProductRelated']
df['AvgDuration'] = df['TotalDuration'] / df['TotalPages'].replace(0, 1)  # Avoid division by zero
df['BounceExitRatio'] = df['BounceRates'] / df['ExitRates'].replace(0, 0.001)  # Avoid division by zero
df['PageValueBucket'] = pd.qcut(df['PageValues'], q=5, labels=False, duplicates='drop')

# Prepare data for modeling
X = df.drop('Revenue', axis=1)
y = df['Revenue']

# Identify categorical and numerical features
categorical_features = ['Month', 'VisitorType', 'Weekend']
numerical_features = [col for col in X.columns if col not in categorical_features]

#MODEL BUILDING

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Create and train model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
model.fit(X_train, y_train)

#MODEL EVALUATION

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
print(f"\nCross-Validation ROC-AUC Scores: {cv_scores}")
print(f"Mean ROC-AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Evaluate on test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('visualizations/confusion_matrix.png')

# ROC curve
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('visualizations/roc_curve.png')

#FEATURE IMPORTANCE ANALYSIS

# Get feature names from pipeline
preprocessor.fit(X)
feature_names = (
    numerical_features +
    list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
)

# Extract feature importances from the model
importances = model.named_steps['classifier'].feature_importances_

# Create DataFrame for feature importances
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15))
plt.title('Top 15 Features by Importance')
plt.tight_layout()
plt.savefig('visualizations/feature_importance.png')
print(feature_importance_df.head(15))

# 7. BUSINESS INSIGHTS AND RECOMMENDATIONS
print("\nSTEP 7: Generating business insights and recommendations...")

# Print insights
print("\nTop Business Insights:")
print("1. The most important predictors of purchase behavior are:", ', '.join(feature_importance_df['Feature'].head(5).tolist()))
print("2. Monthly purchase rates show seasonality with peak months being:", 
      monthly_purchase_pct[True].sort_values(ascending=False).index.tolist()[:3])
print("3. Returning visitor conversion rate vs. new visitor conversion rate:", 
      round(df[df['VisitorType'] == 'Returning_Visitor']['Revenue'].mean() * 100, 2), "% vs", 
      round(df[df['VisitorType'] == 'New_Visitor']['Revenue'].mean() * 100, 2), "%")
print("4. Weekend vs. weekday conversion rates:", 
      round(df[df['Weekend'] == True]['Revenue'].mean() * 100, 2), "% vs", 
      round(df[df['Weekend'] == False]['Revenue'].mean() * 100, 2), "%")

print("\nBusiness Recommendations:")
print("1. Focus optimization efforts on the top 5 most influential website features")
print("2. Adjust marketing campaigns to target peak conversion months")
print("3. Implement personalized strategies for different visitor types")
print("4. Optimize for the day-of-week patterns identified in the analysis")

print("\nPredictive Analytics Project completed successfully!")