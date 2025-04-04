import os
import pandas as pd
from Bio.PDB import PDBParser

# === Configuration ===
cf_output = "cf_output"
shap_csv_path = os.path.join(cf_output, "shap_plddt_combined.csv")

# === Load SHAP CSV ===
shap_df = pd.read_csv(shap_csv_path)

# === Dynamically find PDB file ===
pdb_files = [f for f in os.listdir(cf_output) if f.endswith(".pdb") and "unrelaxed_rank" in f]
if not pdb_files:
    raise FileNotFoundError("❌ No unrelaxed_rank_*.pdb file found in cf_output.")
pdb_path = os.path.join(cf_output, pdb_files[0])
print(f"📦 Using PDB: {pdb_path}")

# === Parse PDB file ===
parser = PDBParser(QUIET=True)
structure = parser.get_structure("model", pdb_path)

residue_info = []
for model in structure:
    for chain in model:
        for residue in chain:
            if residue.id[0] == " ":  # Exclude HETATM and waters
                res_id = residue.id[1]
                res_name = residue.resname
                residue_info.append((res_id, res_name))

# === 3-letter to 1-letter amino acid map ===
three_to_one = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
    'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
    'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}
residue_letters = [three_to_one.get(name, 'X') for (_, name) in residue_info]
residue_indices = [res_id for (res_id, _) in residue_info]

# === Align lengths if SHAP output is shorter than structure ===
min_len = min(len(shap_df), len(residue_indices))
shap_df = shap_df.iloc[:min_len].copy()
residue_indices = residue_indices[:min_len]
residue_letters = residue_letters[:min_len]

# === Append to DataFrame ===
shap_df["True_PDB_Residue_Index"] = residue_indices
shap_df["Residue_AA"] = residue_letters

# === Save Output ===
output_file = os.path.join(cf_output, "shap_with_real_residue_mapping.csv")
shap_df.to_csv(output_file, index=False)
print(f"✅ Saved mapping to: {output_file}")

