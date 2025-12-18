import pandas as pd
import os

codebook_path = r'c:\Users\ssema\Desktop\FailDetect\codebook\FinalCodeBook.csv'
backup_path = r'c:\Users\ssema\Desktop\FailDetect\codebook\FinalCodeBook_backup.csv'

# List of variables to keep
keep_vars = [
    'BCBG14E', 'BCBG14F', 'BCBG14G', 'BCBG14H', 
    'BCBG16A', 'BCBG16B', 
    'BCBG14I', 'BCBG14J', 
    'BCBG15A', 'BCBG15B', 'BCBG15C', 'BCBG15D', 'BCBG15E', 'BCBG15F', 'BCBG15G', 'BCBG15H', 
    'BCBGEAS', 'BCDGEAS'
]

# Always keep 'Fail' as it is the target variable added previously
keep_vars.append('Fail')

print(f"Reading {codebook_path}...")
try:
    df = pd.read_csv(codebook_path, encoding='latin-1')
except:
    df = pd.read_csv(codebook_path, encoding='utf-8')

print(f"Original Codebook shape: {df.shape}")

# Backup
df.to_csv(backup_path, index=False)
print(f"Backup saved to {backup_path}")

# Filter
# The first column is usually 'Variable' but might have invisible chars/BOM if not handled by encoding
# We will use column index 0
filtered_df = df[df.iloc[:, 0].astype(str).str.strip().isin(keep_vars)]

print(f"Filtered Codebook shape: {filtered_df.shape}")
print("Variables kept:")
print(filtered_df.iloc[:, 0].tolist())

# Save back
filtered_df.to_csv(codebook_path, index=False)
print(f"Updated Codebook saved to {codebook_path}")
