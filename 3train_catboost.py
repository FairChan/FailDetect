import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Paths
DATA_PATH = r'dataProcessing/Final.csv'
CODEBOOK_PATH = r'codebook/FinalCodeBook.csv'
MODEL_DIR = r'models'

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

print("="*50)
print("PHASE 1: DATA PREPARATION")
print("="*50)

# 1. Load Data
print("Loading dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Data loaded: {df.shape}")

# 2. Identify Categorical vs Numerical Features based on Codebook
# We will parse the Codebook to identify 'Nominal' as Categorical and 'Scale' as Numerical
print("Parsing Codebook for feature types...")
codebook = pd.read_csv(CODEBOOK_PATH, encoding='latin-1')

# Create dictionaries for feature types
cat_features = []
num_features = []

# Target variable
target = 'Fail'

# Features to exclude (IDs, weights, target, auxiliary)
exclude_cols = ['IDCNTRY', 'IDBOOK', 'IDSCHOOL', 'IDCLASS', 'IDSTUD', 
                'IDTEALIN', 'IDTEACH', 'IDLINK', 'IDPOP', 'IDGRADER', 
                'IDGRADE', 'IDSUBJ', 'MATWGT', 'JKREP', 'JKZONE', 
                'Math_Mean', 'Fail']

# Scan dataset columns and match with codebook
for col in df.columns:
    if col in exclude_cols:
        continue
    
    # Find variable in codebook
    # We strip whitespace just in case
    var_info = codebook[codebook.iloc[:, 0].astype(str).str.strip() == col]
    
    if not var_info.empty:
        # Use numerical index for Level (column 3 based on inspection)
        # Variable,Label,Question Location,Level,...
        level = var_info.iloc[0, 3]
        if level == 'Nominal':
            cat_features.append(col)
        elif level == 'Scale':
            num_features.append(col)
        else:
            # Fallback: check dtype
            if pd.api.types.is_numeric_dtype(df[col]):
                num_features.append(col)
            else:
                cat_features.append(col)
    else:
        # If not in codebook, guess based on dtype
        if pd.api.types.is_numeric_dtype(df[col]):
            num_features.append(col)
        else:
            cat_features.append(col)

print(f"Categorical features ({len(cat_features)}): {cat_features}")
print(f"Numerical features ({len(num_features)}): {num_features}")

# 3. Handle Categorical Features for CatBoost
# CatBoost requires categorical features to be strings or integers.
# We fill NaNs with a placeholder string to treat missingness as a category
print("Handling missing values for categorical features...")
for col in cat_features:
    df[col] = df[col].astype(str).replace('nan', 'Missing')

# Fill numerical missing values with median (simple imputation)
print("Handling missing values for numerical features...")
for col in num_features:
    # Ensure numeric type first, coercing errors to NaN
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())

# 4. Split Data (70% Train, 15% Validation, 15% Test)
print("Splitting data...")
X = df[cat_features + num_features]
y = df[target]

# First split: Train (70%) vs Temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
)

# Second split: Validation (15% of total -> 50% of Temp) vs Test (15% of total -> 50% of Temp)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
)

print(f"Training Set: {X_train.shape}")
print(f"Validation Set: {X_val.shape}")
print(f"Test Set: {X_test.shape}")

# Create CatBoost Pools
train_pool = Pool(X_train, y_train, cat_features=cat_features)
val_pool = Pool(X_val, y_val, cat_features=cat_features)
test_pool = Pool(X_test, y_test, cat_features=cat_features)

print("\n" + "="*50)
print("PHASE 2: TRAINING (BASELINE)")
print("="*50)

# 1. Baseline Model
print("Initializing Baseline CatBoostClassifier (GPU)...")
baseline_model = CatBoostClassifier(
    task_type="GPU",
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=RANDOM_SEED,
    verbose=100
)

# 2. Train with Early Stopping
print("Training Baseline Model...")
start_time = time.time()
baseline_model.fit(
    train_pool,
    eval_set=val_pool,
    early_stopping_rounds=50,
    use_best_model=True
)
end_time = time.time()
print(f"Baseline Training Time: {end_time - start_time:.2f} seconds")

# 3. Save Baseline
baseline_path = os.path.join(MODEL_DIR, 'catboost_baseline.cbm')
baseline_model.save_model(baseline_path)
print(f"Baseline model saved to {baseline_path}")

print("\n" + "="*50)
print("PHASE 3: HYPERPARAMETER TUNING")
print("="*50)

# 1. RandomizedSearchCV
print("Setting up RandomizedSearchCV...")

# Hyperparameter tuning
# Reduced search space for faster demonstration
param_dist = {
    'iterations': [500],
    'depth': [4, 6],
    'learning_rate': [0.03, 0.1],
    'l2_leaf_reg': [1, 3]
}

# Initialize a base model for tuning
# Note: For GridSearchCV/RandomizedSearchCV with CatBoost, it's often better to pass CPU task_type for the search controller 
# but the estimator can use GPU. However, sklearn cross-validation splits data which forces CatBoost to re-upload to GPU many times.
# We will use a simplified approach: use the built-in randomized_search or grid_search from CatBoost library if possible, 
# or use sklearn's RandomizedSearchCV with GPU enabled on the estimator.

# Using CatBoost's built-in randomized_search
print("Starting Hyperparameter Tuning using CatBoost's built-in randomized_search...")
start_time = time.time()

tuning_model = CatBoostClassifier(
    task_type="GPU",
    loss_function='Logloss',
    eval_metric='AUC',
    cat_features=cat_features,
    random_seed=RANDOM_SEED,
    verbose=0,
    early_stopping_rounds=50
)

# CatBoost expects params in a different format for randomized_search
# It returns a dictionary with 'params' key containing the best parameters
tuned_result = tuning_model.randomized_search(
    param_distributions=param_dist,
    X=X_train,
    y=y_train,
    cv=2,
    n_iter=2,
    partition_random_seed=RANDOM_SEED,
    calc_cv_statistics=True,
    search_by_train_test_split=False, # Use CV
    verbose=False,
    plot=False
)

end_time = time.time()
print(f"Hyperparameter Tuning Time: {end_time - start_time:.2f} seconds")

# 4. Best Parameters
best_params = tuned_result['params']
print(f"Best Parameters found: {best_params}")

# 5. Retrain with Best Parameters
print("Retraining Tuned Model with Best Parameters...")
tuned_model = CatBoostClassifier(
    task_type="GPU",
    iterations=best_params.get('iterations', 1000), # Default if not in result
    depth=best_params.get('depth', 6),
    learning_rate=best_params.get('learning_rate', 0.03),
    l2_leaf_reg=best_params.get('l2_leaf_reg', 3),
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=RANDOM_SEED,
    verbose=100
)

tuned_model.fit(
    train_pool,
    eval_set=val_pool,
    early_stopping_rounds=50,
    use_best_model=True
)

# 6. Save Tuned Model
tuned_path = os.path.join(MODEL_DIR, 'catboost_tuned.cbm')
tuned_model.save_model(tuned_path)
print(f"Tuned model saved to {tuned_path}")


print("\n" + "="*50)
print("PHASE 4: VALIDATION & EVALUATION")
print("="*50)

def evaluate_model(model, pool, name="Model"):
    y_pred = model.predict(pool)
    y_prob = model.predict_proba(pool)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"--- {name} Evaluation ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    return f1

print("Evaluating Baseline Model on Test Set...")
f1_baseline = evaluate_model(baseline_model, test_pool, "Baseline")

print("\nEvaluating Tuned Model on Test Set...")
f1_tuned = evaluate_model(tuned_model, test_pool, "Tuned")

print("\nPerformance Comparison:")
print(f"Baseline F1: {f1_baseline:.4f}")
print(f"Tuned F1:    {f1_tuned:.4f}")
if f1_tuned > f1_baseline:
    print("Result: Tuned model outperformed Baseline.")
else:
    print("Result: Tuned model did not outperform Baseline (might need more search iterations).")


print("\n" + "="*50)
print("PHASE 5: MODEL INTERPRETATION")
print("="*50)

# Feature Importance
feature_importances = tuned_model.get_feature_importance(train_pool)
feature_names = X_train.columns
fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
fi_df = fi_df.sort_values(by='Importance', ascending=False).head(10)

print("Top 10 Important Features:")
print(fi_df)

# Interpretation Text
top_feature = fi_df.iloc[0]['Feature']
print(f"\nInterpretation Analysis:")
print(f"The most influential feature for predicting student failure is '{top_feature}'.")
print("This suggests that this specific background factor plays a critical role in academic performance.")
print("Educators should focus on monitoring these high-impact variables to intervene early.")

print("\n" + "="*50)
print("EXECUTION COMPLETE")
print("="*50)
