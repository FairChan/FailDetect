import pandas as pd
import numpy as np

# Paths
processed_csv_path = r'dataProcessing/Processed_Original.csv'
result_csv_path = r'dataProcessing/OriginalResult.csv'
output_csv_path = r'dataProcessing/Final.csv'

print("Loading datasets...")
# Load processed data
df_processed = pd.read_csv(processed_csv_path, low_memory=False)
print(f"Processed data loaded: {len(df_processed)} rows.")

# Load result data
# We only need IDSTUD and the Math PVs to determine fail status
cols_to_use = ['IDCNTRY', 'IDSTUD', 'BSMMAT01', 'BSMMAT02', 'BSMMAT03', 'BSMMAT04', 'BSMMAT05']
df_result = pd.read_csv(result_csv_path, usecols=cols_to_use, low_memory=False)
print(f"Result data loaded: {len(df_result)} rows.")

# Deduplicate result data
# OriginalResult.csv contains multiple rows per student (e.g. linked to different teachers)
# We drop duplicates based on IDSTUD, keeping the first occurrence (scores are invariant for the student)
df_result_unique = df_result.drop_duplicates(subset=['IDCNTRY', 'IDSTUD'])
print(f"Unique students in result data: {len(df_result_unique)}")

# Define Fail Standard
# Standard: Average Math Plausible Value < 400 (Below Low International Benchmark)
print("Calculating Fail status...")
pv_cols = ['BSMMAT01', 'BSMMAT02', 'BSMMAT03', 'BSMMAT04', 'BSMMAT05']
df_result_unique['Math_Mean'] = df_result_unique[pv_cols].mean(axis=1)

# Create Fail column
# 1 = Fail (Mean Score < 400)
# 0 = Pass (Mean Score >= 400)
df_result_unique['Fail'] = np.where(df_result_unique['Math_Mean'] < 400, 1, 0)

print(f"Fail counts:\n{df_result_unique['Fail'].value_counts()}")

# Merge
print("Merging datasets...")
# Left merge to keep all students in the processed file
# Using IDCNTRY and IDSTUD as keys
df_merged = pd.merge(df_processed, 
                     df_result_unique[['IDCNTRY', 'IDSTUD', 'Math_Mean', 'Fail']], 
                     on=['IDCNTRY', 'IDSTUD'], 
                     how='left')

# Check for unmerged students
missing_scores = df_merged['Fail'].isna().sum()
if missing_scores > 0:
    print(f"Warning: {missing_scores} students in Processed_Original.csv did not have scores in OriginalResult.csv")

# Save
try:
    df_merged.to_csv(output_csv_path, index=False)
    print(f"Merged data saved to {output_csv_path}")
except PermissionError:
    print(f"Error: Could not save to {output_csv_path}. File might be open.")
    temp_output = output_csv_path.replace('.csv', '_new.csv')
    df_merged.to_csv(temp_output, index=False)
    print(f"Saved to {temp_output} instead.")

# Update Codebook with Fail definition
codebook_path = r'c:\Users\ssema\Desktop\FailDetect\codebook\FinalCodeBook.csv'
print(f"Updating codebook at {codebook_path}...")

try:
    codebook_df = pd.read_csv(codebook_path, encoding='utf-8-sig')
except:
    codebook_df = pd.read_csv(codebook_path, encoding='latin-1')

# Clean column names to ensure matching
codebook_df.columns = codebook_df.columns.str.strip().str.replace('^ï»¿', '', regex=True)
var_col = codebook_df.columns[0] # Assuming first column is Variable

# Check if Fail variable already exists
if 'Fail' not in codebook_df[var_col].astype(str).values:
    # Construct new row dictionary matching the columns
    new_row = {col: '' for col in codebook_df.columns}
    
    new_row[codebook_df.columns[0]] = 'Fail' # Variable
    new_row[codebook_df.columns[1]] = 'Student Failure Status (Math Mean < 400)' # Label
    new_row[codebook_df.columns[2]] = 'Derived' # Question Location
    new_row[codebook_df.columns[3]] = 'Nominal' # Level
    new_row[codebook_df.columns[4]] = '1' # Width
    new_row[codebook_df.columns[5]] = '0' # Decimals
    new_row[codebook_df.columns[6]] = '0' # Range Minimum
    new_row[codebook_df.columns[7]] = '1' # Range Maximum
    new_row[codebook_df.columns[8]] = '0: Pass; 1: Fail' # Value Scheme Detailed
    new_row[codebook_df.columns[13]] = 'Derived' # Domain
    new_row[codebook_df.columns[14]] = 'D' # Variable Class
    new_row[codebook_df.columns[15]] = 'Derived from BSMMAT01-05' # Comment
    
    # Create DataFrame for new row
    new_row_df = pd.DataFrame([new_row])
    
    # Append
    codebook_df = pd.concat([codebook_df, new_row_df], ignore_index=True)
    
    # Save
    codebook_df.to_csv(codebook_path, index=False, encoding='utf-8-sig')
    print("Added 'Fail' variable to Codebook.")
else:
    print("'Fail' variable already exists in Codebook.")
