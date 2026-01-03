"""
Extract and save all plots from notebooks as PNG files
"""
import nbformat
import base64
import os
from pathlib import Path

def extract_plots_from_notebook(notebook_path, output_dir):
    """Extract all plots from a notebook and save as PNG files"""

    print(f"\nProcessing: {notebook_path}")
    print("-" * 60)

    # Read notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Create output directory
    notebook_name = Path(notebook_path).stem
    notebook_output_dir = os.path.join(output_dir, notebook_name)
    os.makedirs(notebook_output_dir, exist_ok=True)

    plot_count = 0

    # Iterate through cells
    for cell_idx, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' and 'outputs' in cell:
            for output_idx, output in enumerate(cell.outputs):
                # Check for PNG images
                if 'data' in output and 'image/png' in output['data']:
                    plot_count += 1

                    # Decode base64 image
                    image_data = output['data']['image/png']
                    image_bytes = base64.b64decode(image_data)

                    # Save as PNG
                    filename = f"cell_{cell_idx:02d}_plot_{output_idx:02d}.png"
                    filepath = os.path.join(notebook_output_dir, filename)

                    with open(filepath, 'wb') as f:
                        f.write(image_bytes)

                    print(f"  [OK] Saved: {filename}")

    print(f"Total plots extracted: {plot_count}")
    return plot_count

def main():
    """Main function to extract plots from all notebooks"""

    # Output directory
    output_dir = "presentation/screenshots/plots"
    os.makedirs(output_dir, exist_ok=True)

    # Notebooks to process
    notebooks = [
        'notebooks/01_exploratory_analysis.ipynb',
        'notebooks/02_model_training.ipynb',
        'notebooks/03_clustering_analysis.ipynb',
        'notebooks/04_geo_visualization.ipynb'
    ]

    print("=" * 60)
    print("EXTRACTING PLOTS FROM NOTEBOOKS")
    print("=" * 60)

    total_plots = 0
    results = {}

    for notebook_path in notebooks:
        if os.path.exists(notebook_path):
            count = extract_plots_from_notebook(notebook_path, output_dir)
            results[notebook_path] = count
            total_plots += count
        else:
            print(f"[ERROR] Notebook not found: {notebook_path}")
            results[notebook_path] = 0

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for nb, count in results.items():
        print(f"{count:2d} plots from {nb}")
    print("-" * 60)
    print(f"TOTAL: {total_plots} plots saved to {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
