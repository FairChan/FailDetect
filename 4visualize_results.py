import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set random seed for reproducibility (Must match train_catboost.py)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
# Paths
DATA_PATH = r'c:\Users\ssema\Desktop\FailDetect\dataProcessing\Final.csv'
CODEBOOK_PATH = r'c:\Users\ssema\Desktop\FailDetect\codebook\FinalCodeBook.csv'
MODEL_DIR = r'c:\Users\ssema\Desktop\FailDetect\models'
MODEL_PATH = os.path.join(MODEL_DIR, 'catboost_tuned.cbm')

print("="*50)
print("GENERATING VISUALIZATION ARTIFACTS")
print("="*50)

# 1. Load Data
print("Loading dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)

# 2. Identify Categorical vs Numerical Features based on Codebook
print("Parsing Codebook for feature types...")
codebook = pd.read_csv(CODEBOOK_PATH, encoding='latin-1')

cat_features = []
num_features = []
target = 'Fail'
exclude_cols = ['IDCNTRY', 'IDBOOK', 'IDSCHOOL', 'IDCLASS', 'IDSTUD', 
                'IDTEALIN', 'IDTEACH', 'IDLINK', 'IDPOP', 'IDGRADER', 
                'IDGRADE', 'IDSUBJ', 'MATWGT', 'JKREP', 'JKZONE', 
                'Math_Mean', 'Fail']

for col in df.columns:
    if col in exclude_cols:
        continue
    
    var_info = codebook[codebook.iloc[:, 0].astype(str).str.strip() == col]
    
    if not var_info.empty:
        level = var_info.iloc[0, 3]
        if level == 'Nominal':
            cat_features.append(col)
        elif level == 'Scale':
            num_features.append(col)
        else:
            if pd.api.types.is_numeric_dtype(df[col]):
                num_features.append(col)
            else:
                cat_features.append(col)
    else:
        if pd.api.types.is_numeric_dtype(df[col]):
            num_features.append(col)
        else:
            cat_features.append(col)

# 3. Handle Categorical Features
for col in cat_features:
    df[col] = df[col].astype(str).replace('nan', 'Missing')

# Handle Numerical Missing Values
for col in num_features:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())

# 4. Split Data (Must match train_catboost.py)
X = df[cat_features + num_features]
y = df[target]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
)

# 5. Load Model
print(f"Loading model from {MODEL_PATH}...")
model = CatBoostClassifier()
model.load_model(MODEL_PATH)

# 6. Generate Feature Importance
print("Generating Feature Importance Table...")
feature_importances = model.get_feature_importance(Pool(X_train, y_train, cat_features=cat_features))
feature_names = X_train.columns
fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
fi_df = fi_df.sort_values(by='Importance', ascending=False)

# Save Feature Importance CSV
fi_csv_path = os.path.join(MODEL_DIR, 'feature_importance.csv')
fi_df.to_csv(fi_csv_path, index=False)
print(f"Feature Importance CSV saved to: {fi_csv_path}")

# Plot Feature Importance (Top 20)
plt.figure(figsize=(10, 8))
sns.barplot(x="Importance", y="Feature", data=fi_df.head(20))
plt.title('Top 20 Important Features (CatBoost)')
plt.tight_layout()
fi_png_path = os.path.join(MODEL_DIR, 'feature_importance.png')
plt.savefig(fi_png_path)
print(f"Feature Importance Plot saved to: {fi_png_path}")

# 7. Generate Confusion Matrix
print("Generating Confusion Matrix...")
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Pass', 'Fail'], 
            yticklabels=['Pass', 'Fail'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix (Test Set)')
cm_png_path = os.path.join(MODEL_DIR, 'confusion_matrix.png')
plt.savefig(cm_png_path)
print(f"Confusion Matrix Plot saved to: {cm_png_path}")

print("="*50)
print("Artifact generation complete.")
