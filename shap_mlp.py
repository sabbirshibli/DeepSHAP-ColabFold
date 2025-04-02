# shap_mlp.py

import os
import json
import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import random
from plddt_model import load_trained_model

# === Fix randomness for reproducibility ===
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

fasta_file = sys.argv[1]
cf_output = sys.argv[2]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load sequence ===
def load_sequence(fasta_path):
    with open(fasta_path) as f:
        lines = f.readlines()
    return "".join([l.strip() for l in lines if not l.startswith(">")])

# === One-hot encoding ===
def one_hot_encode(seq):
    aa_dict = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    encoded = np.zeros((len(seq), 20))
    for i, aa in enumerate(seq):
        if aa in aa_dict:
            encoded[i, aa_dict[aa]] = 1
    return encoded.flatten()

sequence = load_sequence(fasta_file)
X = one_hot_encode(sequence).astype(np.float32)
test_input = np.array([X])

# === Generate or load mutated background ===
def generate_mutated_background(seq, num_samples=20, mutation_rate=0.15):
    aa_list = list("ACDEFGHIKLMNPQRSTVWY")
    original_seq = np.array(list(seq))
    bg_samples = []

    for _ in range(num_samples):
        mutated_seq = original_seq.copy()
        for idx in range(len(seq)):
            if np.random.rand() < mutation_rate:
                mutated_seq[idx] = np.random.choice(aa_list)
        mutated_X = one_hot_encode("".join(mutated_seq)).astype(np.float32)
        bg_samples.append(mutated_X)

    return np.stack(bg_samples)

bg_path = os.path.join(cf_output, "fixed_background.npy")
if os.path.exists(bg_path):
    print("📂 Loading saved background...")
    background = np.load(bg_path)
else:
    print("🧬 Generating new background...")
    background = generate_mutated_background(sequence, num_samples=20, mutation_rate=0.15)
    np.save(bg_path, background)
    print(f"💾 Background saved to {bg_path}")

# === Load pLDDT output ===
json_file = [f for f in os.listdir(cf_output) if f.endswith(".json") and "_scores" in f][0]
with open(os.path.join(cf_output, json_file)) as f:
    plddt = np.array(json.load(f)["plddt"], dtype=np.float32)

input_len = X.shape[0]
output_len = len(plddt)
model_path = os.path.join(cf_output, "plddt_mlp_model.pt")

# === Load trained model ===
model = load_trained_model(input_len, output_len, model_path, device)

# === SHAP using DeepExplainer ===
def predict_fn(x):
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32).to(device)).cpu().numpy()

print("🔍 Running DeepSHAP with Fixed Background...")
explainer = shap.DeepExplainer(model, torch.tensor(background).float().to(device))
shap_values = explainer.shap_values(torch.tensor(test_input).float().to(device))[0]

# === SHAP Matrix and Normalization ===
reshaped = shap_values.reshape(output_len, len(sequence), 20)
shap_scores = reshaped.sum(axis=(1, 2))
norm_scores = shap_scores / np.sum(np.abs(shap_scores))

# === DEBUGGING Predictions ===
print("\n🔎 DEBUGGING PREDICTIONS:")
mutated_input = background[0]
pred_mutated = predict_fn(np.array([mutated_input]))
pred_real = predict_fn(np.array([X]))
print(f"Mean prediction (real input): {pred_real.mean():.4f}")
print(f"Mean prediction (mutated input): {pred_mutated.mean():.4f}\n")

# === Plot SHAP Scores ===
plt.figure(figsize=(14,4))
plt.plot(shap_scores, color='blue', label='Raw SHAP Scores')
plt.axhline(0, color='gray', linestyle='--')
plt.title("Raw SHAP Scores (Fixed Background)")
plt.xlabel("Residue Index")
plt.ylabel("SHAP Score")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(cf_output, "raw_shap_plot.png"))
plt.show()

plt.figure(figsize=(14,4))
plt.plot(norm_scores, color='green', label='Normalized SHAP Scores')
plt.axhline(0, color='gray', linestyle='--')
plt.title("Normalized SHAP Scores (Fixed Background)")
plt.xlabel("Residue Index")
plt.ylabel("Normalized SHAP")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(cf_output, "normalized_shap_plot.png"))
plt.show()

plt.figure(figsize=(10,4))
plt.hist(shap_scores, bins=30, color='purple', edgecolor='black')
plt.title("Histogram of Raw SHAP Scores (Fixed Background)")
plt.xlabel("SHAP Score")
plt.ylabel("Frequency")
plt.axvline(0, color='gray', linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(cf_output, "shap_histogram.png"))
plt.show()

# === Save outputs ===
combined_df = pd.DataFrame({
    "Residue_Index": np.arange(1, output_len + 1),
    "SHAP_Score": shap_scores,
    "Normalized_SHAP": norm_scores,
    "pLDDT": plddt
})
combined_df.to_csv(os.path.join(cf_output, "shap_plddt_combined.csv"), index=False)
np.save(os.path.join(cf_output, "shap_values.npy"), shap_values)
np.save(os.path.join(cf_output, "input_vector.npy"), X)

print("\n✅ Done! SHAP results saved to:")
print(" - raw_shap_plot.png")
print(" - normalized_shap_plot.png")
print(" - shap_histogram.png")
print(" - shap_plddt_combined.csv")
print(" - fixed_background.npy")

