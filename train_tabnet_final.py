import pandas as pd
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import os
import time
import json
import random
import itertools

# ==========================================
# Configuration & Setup
# ==========================================
SEED = 42
ENABLE_FEATURE_FUSION = True

def add_feature_fusion(df, numeric_cols, max_base_features=20):
    if not numeric_cols:
        return df, numeric_cols
    base_cols = numeric_cols[:max_base_features]
    new_numeric_cols = list(numeric_cols)
    for col in base_cols:
        col_numeric = pd.to_numeric(df[col], errors="coerce")
        new_col = f"{col}__sq"
        df[new_col] = col_numeric ** 2
        new_numeric_cols.append(new_col)
    for c1, c2 in itertools.combinations(base_cols, 2):
        c1_numeric = pd.to_numeric(df[c1], errors="coerce")
        c2_numeric = pd.to_numeric(df[c2], errors="coerce")
        new_col = f"{c1}__x__{c2}"
        df[new_col] = c1_numeric * c2_numeric
        new_numeric_cols.append(new_col)
    return df, new_numeric_cols
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

DATA_PATH = r'c:\Users\ssema\Desktop\FailDetect\dataProcessing\Final.csv'
CODEBOOK_PATH = r'c:\Users\ssema\Desktop\FailDetect\codebook\FinalCodeBook.csv'
OUTPUT_DIR = r'c:\Users\ssema\Desktop\FailDetect\models_tabnet'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# ==========================================
# 1. Data Preparation
# ==========================================
def load_and_preprocess_data():
    print("\n[Phase 1] Data Preparation...")
    
    # Load Data
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded data shape: {df.shape}")
    
    # Identify Target
    # User said "at_risk" (1=Risk). 
    # In our data, 'Fail' is 1 (Fail) / 0 (Pass).
    if 'Fail' in df.columns:
        df.rename(columns={'Fail': 'at_risk'}, inplace=True)
    elif 'at_risk' not in df.columns:
        # Fallback: assume last column is target if not named Fail/at_risk
        print("Warning: 'Fail' or 'at_risk' column not found. Using last column as target.")
        df.rename(columns={df.columns[-1]: 'at_risk'}, inplace=True)
    
    target_col = 'at_risk'
    
    # Force target to numeric and drop NaNs
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df.dropna(subset=[target_col], inplace=True)
    df[target_col] = df[target_col].astype(int)
    
    print(f"Target distribution:\n{df[target_col].value_counts(normalize=True)}")

    # Load Codebook to determine allowed features and types
    try:
        cb = pd.read_csv(CODEBOOK_PATH, encoding='latin-1')
        cb.columns = cb.columns.str.strip().str.replace('^ï»¿', '', regex=True)
        var_col = cb.columns[0] # Usually 'Variable'
        
        allowed_vars = set(cb[var_col].dropna().astype(str).str.strip().tolist())
        print(f"Allowed variables from codebook: {len(allowed_vars)}")
        
        # Filter DataFrame to keep only allowed vars + Target
        # Also ensure we don't drop the target if it's not in allowed_vars (though it should be)
        keep_cols = [c for c in df.columns if c in allowed_vars or c == target_col]
        df = df[keep_cols]
        print(f"Filtered data shape (codebook vars only): {df.shape}")
        
        type_map = dict(zip(cb[var_col], cb['Level']))
    except Exception as e:
        print(f"Codebook reading failed ({e}), using all columns.")
        allowed_vars = set(df.columns)
        type_map = {}

    # Drop ID columns and Weights if they exist (heuristics based on previous file knowledge)
    # Even after filtering, double check to remove IDs if they somehow got into codebook
    drop_cols = [c for c in df.columns if c.startswith('ID') or 'WGT' in c or c == target_col]
    feature_cols = [c for c in df.columns if c not in drop_cols and c != target_col]
    
    numeric_cols = []
    categorical_cols = []
        
    for col in feature_cols:
        level = type_map.get(col, 'Unknown')
        if level == 'Scale':
            numeric_cols.append(col)
        elif level == 'Nominal' or level == 'Ordinal':
            categorical_cols.append(col)
        else:
            # Fallback heuristic
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 10:
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

    if ENABLE_FEATURE_FUSION:
        df, numeric_cols = add_feature_fusion(df, numeric_cols)
        feature_cols = [c for c in df.columns if c not in drop_cols and c != target_col]

    print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Split first (70% Train, 30% Temp)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED
    )
    # Split Temp (50% Valid, 50% Test -> 15% Valid, 15% Test of total)
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
    )
    
    # Missing Value Imputation
    # Numeric -> Median (fit on Train)
    for col in numeric_cols:
        # Force to numeric (coerce errors to NaN)
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_valid[col] = pd.to_numeric(X_valid[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
        
        median_val = X_train[col].median()
        X_train[col].fillna(median_val, inplace=True)
        X_valid[col].fillna(median_val, inplace=True)
        X_test[col].fillna(median_val, inplace=True)
        
    # Categorical -> Mode or "Missing" (fit on Train)
    for col in categorical_cols:
        # If numeric-like categorical (int codes), fill with -1 or mode.
        # If string, fill "Missing".
        if pd.api.types.is_numeric_dtype(X_train[col]):
            fill_val = -1 # Common for int-encoded categories
        else:
            fill_val = "Missing"
        
        X_train[col].fillna(fill_val, inplace=True)
        X_valid[col].fillna(fill_val, inplace=True)
        X_test[col].fillna(fill_val, inplace=True)

    # Encoding
    # Numeric -> StandardScaler
    scaler = StandardScaler()
    if numeric_cols:
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_valid[numeric_cols] = scaler.transform(X_valid[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    # Categorical -> LabelEncoder
    # TabNet needs positive integers 0..N-1
    cat_idxs = []
    cat_dims = []
    
    # Map feature names to indices for TabNet
    feature_names = list(X_train.columns)
    
    for col in categorical_cols:
        le = LabelEncoder()
        # Fit on all possible values to avoid unknown class error, OR handle unknowns
        # Best practice: Fit on Train, handle unknown in Valid/Test
        le.fit(X_train[col].astype(str))
        
        # Helper to safely transform
        def safe_transform(series, encoder):
            classes = set(encoder.classes_)
            # Replace unknown with the first class (or a specific 'unknown' if we had one)
            # Here we map unknown to the most frequent (mode) which is usually class 0 after some sorting, 
            # or just use 0. Better: map to a special 'Unknown' if not present.
            # Simplified: Map unknown to class 0
            return series.astype(str).apply(lambda x: encoder.transform([x])[0] if x in classes else 0)

        X_train[col] = le.transform(X_train[col].astype(str))
        X_valid[col] = safe_transform(X_valid[col], le)
        X_test[col] = safe_transform(X_test[col], le)
        
        cat_idxs.append(feature_names.index(col))
        cat_dims.append(len(le.classes_))
    
    return {
        'X_train': X_train.values, 'y_train': y_train.values,
        'X_valid': X_valid.values, 'y_valid': y_valid.values,
        'X_test': X_test.values, 'y_test': y_test.values,
        'cat_idxs': cat_idxs, 'cat_dims': cat_dims,
        'feature_names': feature_names
    }

# ==========================================
# 2. Training (Baseline)
# ==========================================
def train_baseline(data):
    print("\n[Phase 2] Training Baseline TabNet...")
    
    clf = TabNetClassifier(
        seed=SEED,
        cat_idxs=data['cat_idxs'],
        cat_dims=data['cat_dims'],
        cat_emb_dim=1, # Default is 1, can be tuned
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size":10, "gamma":0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='sparsemax', # This is useful for interpretability
        verbose=1
    )
    
    start_time = time.time()
    clf.fit(
        X_train=data['X_train'], y_train=data['y_train'],
        eval_set=[(data['X_train'], data['y_train']), (data['X_valid'], data['y_valid'])],
        eval_name=['train', 'valid'],
        eval_metric=['auc', 'accuracy'],
        max_epochs=50, # Reduced for demo speed, typically 100+
        patience=10,
        batch_size=1024, 
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )
    print(f"Baseline Training Time: {time.time() - start_time:.2f}s")
    print(f"Best Valid Score: {clf.best_cost}")
    
    save_path = os.path.join(OUTPUT_DIR, 'tabnet_baseline.zip')
    clf.save_model(save_path)
    return clf

# ==========================================
# 3. Tuning (Systematic Search)
# ==========================================
def tune_model(data):
    print("\n[Phase 3] Tuning (Randomized Search Manual Loop)...")
    
    # Parameter Space
    param_grid = {
        'n_d': [8, 16, 24],
        'n_steps': [3, 5],
        'gamma': [1.0, 1.5],
        'lambda_sparse': [0, 1e-4],
        'learning_rate': [0.01, 0.02],
    }
    
    # Generate random combinations (e.g., 5 trials)
    import itertools
    keys, values = zip(*param_grid.items())
    all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    # Randomly sample 5
    search_space = random.sample(all_combinations, min(5, len(all_combinations)))
    
    best_score = -1
    best_params = None
    best_model = None
    
    results = []

    for i, params in enumerate(search_space):
        print(f"Trial {i+1}/{len(search_space)}: {params}")
        
        # Set n_a = n_d as per TabNet recommendation
        params['n_a'] = params['n_d']
        
        clf = TabNetClassifier(
            seed=SEED,
            n_d=params['n_d'],
            n_a=params['n_a'],
            n_steps=params['n_steps'],
            gamma=params['gamma'],
            lambda_sparse=params['lambda_sparse'],
            cat_idxs=data['cat_idxs'],
            cat_dims=data['cat_dims'],
            optimizer_params=dict(lr=params['learning_rate']),
            verbose=0
        )
        
        clf.fit(
            X_train=data['X_train'], y_train=data['y_train'],
            eval_set=[(data['X_valid'], data['y_valid'])],
            eval_metric=['auc'],
            max_epochs=30,
            patience=5,
            batch_size=1024,
            virtual_batch_size=128
        )
        
        val_auc = clf.best_cost
        results.append({'params': params, 'val_auc': val_auc})
        print(f"  -> Valid AUC: {val_auc}")
        
        if val_auc > best_score:
            best_score = val_auc
            best_params = params
            best_model = clf
            
    print(f"\nBest Params: {best_params}")
    print(f"Best Valid AUC: {best_score}")
    
    # Save Best Model
    save_path = os.path.join(OUTPUT_DIR, 'tabnet_tuned.zip')
    best_model.save_model(save_path)
    
    # Save Tuning Logs
    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, 'tuning_results.csv'), index=False)
    
    return best_model

# ==========================================
# 4. Validation & Evaluation
# ==========================================
def evaluate_model(model, data, model_name="Model"):
    print(f"\n[Phase 4] Evaluating {model_name}...")
    
    # Predict
    preds = model.predict(data['X_test'])
    # TabNet predict_proba returns [prob_0, prob_1]
    probs = model.predict_proba(data['X_test'])[:, 1]
    
    y_true = data['y_test']
    
    # Metrics
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds)
    rec = recall_score(y_true, preds)
    f1 = f1_score(y_true, preds)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f} (Crucial for At-Risk)")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, preds))
    
    # Save Metrics
    metrics = {
        'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1
    }
    with open(os.path.join(OUTPUT_DIR, f'{model_name}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Confusion Matrix Plot
    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{model_name} Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Not Risk', 'At Risk'])
    plt.yticks(tick_marks, ['Not Risk', 'At Risk'])
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(OUTPUT_DIR, f'{model_name}_cm.png'))
    plt.close()

# ==========================================
# 5. Interpretability
# ==========================================
def interpret_model(model, data):
    print("\n[Phase 5] Feature Importance...")
    
    feat_importances = model.feature_importances_
    indices = np.argsort(feat_importances)[::-1]
    
    print("All Feature Importances:")
    top_feats = []
    for i in range(len(indices)):
        feat_name = data['feature_names'][indices[i]]
        score = feat_importances[indices[i]]
        print(f"{i+1}. {feat_name}: {score:.4f}")
        top_feats.append({'Feature': feat_name, 'Importance': score})
        
    pd.DataFrame(top_feats).to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # 1. Prepare
    data = load_and_preprocess_data()
    
    # 2. Train Baseline
    baseline_model = train_baseline(data)
    
    # 3. Tune
    tuned_model = tune_model(data)
    
    # 4. Evaluate
    import itertools # Ensure imported for plotting
    evaluate_model(baseline_model, data, "Baseline")
    evaluate_model(tuned_model, data, "Tuned")
    
    # 5. Interpret
    interpret_model(tuned_model, data)
    
    print(f"\nAll artifacts saved to {OUTPUT_DIR}")
