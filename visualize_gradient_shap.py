import os
import re
import pandas as pd

# === Auto-detect files ===
cf_output = "cf_output"
csv_file = os.path.join(cf_output, "shap_plddt_combined.csv")

# Dynamically find the ranked_001 PDB file
pdb_file = None
for f in os.listdir(cf_output):
    if re.search(r"unrelaxed_rank_001_.*\.pdb$", f):
        pdb_file = os.path.join(cf_output, f)
        break

if pdb_file is None:
    raise FileNotFoundError("❌ Could not find ranked_001 unrelaxed PDB file in cf_output.")

output_pml = os.path.join(cf_output, "colored_shap_gradient.pml")

# === Load SHAP values ===
df = pd.read_csv(csv_file)
shap = df["SHAP_Score"].values
resi = df["Residue_Index"].values

# === Normalize SHAP scores for RGB mapping ===
min_val, max_val = shap.min(), shap.max()
pml_lines = [
    f'load {pdb_file}',
    'hide everything',
    'show cartoon',
]

# === Create coloring commands ===
for i, (val, r) in enumerate(zip(shap, resi), 1):
    norm = (val - min_val) / (max_val - min_val + 1e-9)
    r_val = round(norm, 2)       # red for high SHAP
    g_val = round(1 - norm, 2)   # green for low SHAP
    b_val = 0
    pml_lines.append(f'set_color shap{i}, [{r_val}, {g_val}, {b_val}]')
    pml_lines.append(f'color shap{i}, resi {int(r)}')

# === Save PML script ===
with open(output_pml, "w") as f:
    f.write("\n".join(pml_lines))

print(f"✅ Gradient-colored PyMOL script saved to: {output_pml}")

