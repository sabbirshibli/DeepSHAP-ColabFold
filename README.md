# DeepSHAP XAI integration with ColabFold to interpret the prediction of AlphaFold2

This project demonstrates how to apply DeepSHAP Explainable AI to interpret pLDDT predictions made by AlphaFold2 (via ColabFold). It includes scripts to process sequences, train a custom MLP model, and generate SHAP-based explanations and visualizations.

---

## Installation (Tested on NVIDIA DGX)

### Step 1: Create and activate a new virtual environment

```bash
conda create -n colabfold_py39_sabbir python=3.9 -y
conda activate colabfold_py39_sabbir
```

### Step 2: Install ColabFold and related dependencies

```bash
pip install "colabfold[batch,alphafold]"
```

---

## Folder Structure

```
colab_deepshap/
│
├── data/
│   └── input.fasta              # Your protein sequence(s)
│
├── cf_output/                   # ColabFold outputs (generated)
│
├── deepshap_colabfold.py        # Runs ColabFold on input FASTA
├── train_plddt_model.py         # Trains MLP to predict pLDDT scores
├── plddt_model.py               # Shared model class and loader
├── shap_mlp.py                  # DeepSHAP explanation + plotting
├── visualize_gradient_shap.py   # PyMOL-based 3D SHAP visualization
├── shap_viz.py                  # SHAP scatter + waterfall visualizations
└── README.md                    # This file
```

---

## How to Run

1. **Run AlphaFold2 predictions with ColabFold**

```bash
python deepshap_colabfold.py data/input.fasta cf_output
```

2. **Train MLP model using one-hot encoded sequence + pLDDT values**

```bash
python train_plddt_model.py data/input.fasta cf_output
```

3. **Run DeepSHAP to explain which residues influenced predictions**

```bash
python shap_mlp.py data/input.fasta cf_output
```

---

## Optional Visualization Scripts

- **Gradient-based SHAP coloring in PyMOL** (requires PyMOL installed):

```bash
python visualize_gradient_shap.py
```

- **SHAP score scatter + waterfall plots**:

```bash
python shap_viz.py
```

---

## Sample Visualization of GFP with mapped SHAP

![Sample Visualization](assets/gfp_shap_mapped.png)

---

## Bonus: Installing PyMOL locally

> For visualizing 3D colored SHAP scores in PyMOL:

```bash
# Create a safe Python environment
conda create -n pymol_env python=3.9 -y
conda activate pymol_env

# Install open-source PyMOL
conda install -c conda-forge pymol-open-source python=3.9 --yes

# Run PyMOL manually
python -m pymol
```

Inside PyMOL:

```text
@cf_output/colored_shap_gradient.pml
```

---

## Notes

- SHAP results are made reproducible by caching the background (`fixed_background.npy`).
- Model definitions are modularized in `plddt_model.py` for reuse.
- SHAP scores are saved alongside pLDDT predictions in `shap_plddt_combined.csv`.

---

## Citation

If you use this project in your research or tools, please cite the following article:
Sibli, Sabbir Ahmed, Vlasios Panagiotis Panagiotou, and Christos Makris. "Enhancing protein structure predictions: DeepSHAP as a tool for understanding AlphaFold2." Expert Systems with Applications (2025): 127853.

---

Happy coding and interpreting protein structures with SHAP! 🎉🧬
