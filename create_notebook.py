import json
import os

files = [
    ('1process_large_csv.py', '1. Data Processing'),
    ('2merge_fail_data.py', '2. Data Merging & Target Definition'),
    ('3train_catboost.py', '3. CatBoost Training'),
    ('4visualize_results.py', '4. Visualization'),
    ('5train_tabnet_final.py', '5. TabNet Advanced Pipeline')
]

cells = []

# Header
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# FailDetect Complete Pipeline\n",
        "This notebook aggregates the scripts 1 through 5."
    ]
})

for filename, title in files:
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add Markdown Header
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"## {title} ({filename})"]
        })
        
        # Add Code Cell
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": content.splitlines(keepends=True)
        })
    else:
        print(f"Warning: {filename} not found.")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('FailDetect_Pipeline.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Notebook created successfully.")
