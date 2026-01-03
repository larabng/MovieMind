"""
Script to execute all notebooks and save outputs
"""
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import sys

def execute_notebook(notebook_path):
    """Execute a notebook and save with outputs"""
    print(f"\n{'='*60}")
    print(f"Executing: {notebook_path}")
    print(f"{'='*60}")

    try:
        # Read notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        # Execute with high timeout and allow errors
        ep = ExecutePreprocessor(
            timeout=600,
            kernel_name='python3',
            allow_errors=True  # Continue on errors
        )

        # Execute notebook
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})

        # Save notebook with outputs
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)

        print(f"[OK] Successfully executed and saved: {notebook_path}")
        return True

    except Exception as e:
        print(f"[ERROR] Error executing {notebook_path}: {str(e)[:300]}")
        return False

if __name__ == "__main__":
    notebooks = [
        'notebooks/01_exploratory_analysis.ipynb',
        'notebooks/02_model_training.ipynb',
        'notebooks/03_clustering_analysis.ipynb',
        'notebooks/04_geo_visualization.ipynb'
    ]

    results = {}
    for nb in notebooks:
        if os.path.exists(nb):
            results[nb] = execute_notebook(nb)
        else:
            print(f"[ERROR] Notebook not found: {nb}")
            results[nb] = False

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for nb, success in results.items():
        status = "[OK] SUCCESS" if success else "[ERROR] FAILED"
        print(f"{status}: {nb}")
