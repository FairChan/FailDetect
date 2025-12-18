import pandas as pd
import numpy as np
import os
import re

# Paths
input_csv = r'c:\Users\ssema\Desktop\FailDetect\dataProcessing\Original.csv'
codebook_path = r'c:\Users\ssema\Desktop\FailDetect\codebook\FinalCodeBook.csv'
output_csv = r'c:\Users\ssema\Desktop\FailDetect\dataProcessing\Processed_Original.csv'

# 1. Load Codebook and Parse Metadata
print("Loading codebook...")
codebook = pd.read_csv(codebook_path, encoding='latin-1')

# Extract target variables from the codebook
target_vars = codebook['Variable'].tolist()
print(f"Found {len(target_vars)} target variables in codebook.")

# Parse missing values
# Example string: "9: Omitted or invalid; Sysmis: Not administered"
# We want to map {col_name: [9, 'Sysmis']}
missing_map = {}

def parse_missing_scheme(scheme_str):
    missing_vals = []
    if pd.isna(scheme_str):
        return missing_vals
    
    # Split by semicolon
    parts = str(scheme_str).split(';')
    for part in parts:
        part = part.strip()
        # Look for "Value: Label" pattern
        if ':' in part:
            val_str = part.split(':')[0].strip()
            # Try to convert to float/int if possible
            try:
                val = float(val_str)
                # Check if it's an integer
                if val.is_integer():
                    missing_vals.append(int(val))
                else:
                    missing_vals.append(val)
            except ValueError:
                # Keep as string if not numeric (e.g. 'Sysmis' though usually handled separately)
                if val_str.lower() != 'sysmis': # Sysmis is usually auto-handled or empty in CSV
                     missing_vals.append(val_str)
    return missing_vals

for idx, row in codebook.iterrows():
    var = row['Variable']
    scheme = row.get('Missing Scheme Detailed: SPSS', '')
    vals = parse_missing_scheme(scheme)
    if vals:
        missing_map[var] = vals

print("Missing value schemes parsed.")

# 2. Process Original.csv in chunks
chunksize = 50000  # Adjust based on memory
first_chunk = True

print(f"Processing {input_csv} in chunks of {chunksize}...")

# Determine columns to keep: IDs + Target Vars
# Try reading with utf-8-sig to handle BOM
try:
    header = pd.read_csv(input_csv, nrows=0, encoding='utf-8-sig').columns.tolist()
except UnicodeDecodeError:
    print("utf-8-sig failed, trying latin-1 for header...")
    header = pd.read_csv(input_csv, nrows=0, encoding='latin-1').columns.tolist()

print(f"Detected header columns: {header[:5]}...")

id_vars = [col for col in header if col.startswith('ID')]
# Also keep weights if present, usually important (MATWGT, etc.)
weight_vars = [col for col in header if 'WGT' in col or 'JK' in col]

# Combine keep list
keep_cols = id_vars + weight_vars + target_vars
# Ensure we only keep columns that actually exist in the CSV
keep_cols = [c for c in keep_cols if c in header]
# Remove duplicates
keep_cols = list(dict.fromkeys(keep_cols))

print(f"Keeping {len(keep_cols)} columns: {keep_cols[:10]} ...")

processed_count = 0

try:
    # Use the same encoding that worked for header
    encoding = 'utf-8-sig'
    try:
        pd.read_csv(input_csv, nrows=1, encoding=encoding)
    except:
        encoding = 'latin-1'

    with pd.read_csv(input_csv, chunksize=chunksize, usecols=keep_cols, encoding=encoding, low_memory=False) as reader:
        for chunk in reader:
            # Apply missing value handling
            for col in chunk.columns:
                if col in missing_map:
                    # Replace defined missing values with NaN
                    # chunk[col].replace(missing_map[col], np.nan, inplace=True) # deprecated
                    mask = chunk[col].isin(missing_map[col])
                    if mask.any():
                         chunk.loc[mask, col] = np.nan
            
            # Save to file
            mode = 'w' if first_chunk else 'a'
            header_arg = first_chunk
            chunk.to_csv(output_csv, index=False, mode=mode, header=header_arg)
            
            processed_count += len(chunk)
            first_chunk = False
            print(f"Processed {processed_count} rows...")

    print(f"Done! Processed data saved to {output_csv}")

except Exception as e:
    print(f"An error occurred: {e}")
