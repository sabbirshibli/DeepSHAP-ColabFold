import os
import json
import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from torch import nn

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

# === Realistic background: shuffled sequences ===
def generate_shuffled_background(X, num_samples=10):
    background_samples = []
    for _ in range(num_samples):
        shuffled_X = np.random.permutation(X.reshape(-1, 20)).flatten()
        background_samples.append(shuffled_X)
    return np.stack(background_samples)

background = generate_shuffled_background(X, num_samples=10)

# === Load pLDDT output ===
json_file = [f for f in os.listdir(cf_output) if f.endswith(".json") and "_scores" in f][0]
with open(os.path.join(cf_output, json_file)) as f:
    plddt = np.array(json.load(f)["plddt"], dtype=np.float32)

input_len = X.shape[0]
output_len = len(plddt)

# === MLP Model ===
class PlddtMLP(nn.Module):
    def __init__(self, input_len, output_len):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_len, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_len)
        )
    def forward(self, x):
        return self.net(x)

model = PlddtMLP(input_len, output_len).to(device)
model.load_state_dict(torch.load(os.path.join(cf_output, "plddt_mlp_model.pt"), map_location=device))
model.eval()

# === SHAP using DeepExplainer ===
def predict_fn(x):
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32).to(device)).cpu().numpy()

print("🔍 Running DeepSHAP...")
explainer = shap.DeepExplainer(model, torch.tensor(background).float().to(device))
shap_values = explainer.shap_values(torch.tensor(test_input).float().to(device))[0]

# === SHAP Matrix and Normalization ===
reshaped = shap_values.reshape(output_len, len(sequence), 20)
shap_scores = reshaped.sum(axis=(1, 2))
norm_scores = shap_scores / np.sum(np.abs(shap_scores))  # normalization considering absolute values

# === Plot ===
plt.figure(figsize=(14, 4))
plt.plot(norm_scores, label="Normalized SHAP Value", color="green")
plt.axhline(0, color='gray', linestyle='--')
plt.axvspan(10, 20, color="red", alpha=0.2, label="Motif (10–20)")
plt.title(f"Residue-wise Normalized SHAP for {os.path.basename(fasta_file)}")
plt.xlabel("Residue Index")
plt.ylabel("Normalized Importance")
plt.legend()
plt.tight_layout()

plot_path = os.path.join(cf_output, "shap_plot_deepshap.png")
plt.savefig(plot_path)
plt.show()

# === Save CSV ===
combined_df = pd.DataFrame({
    "Residue_Index": np.arange(1, output_len + 1),
    "SHAP_Score": shap_scores,
    "Normalized_SHAP": norm_scores,
    "pLDDT": plddt
})
combined_df.to_csv(os.path.join(cf_output, "shap_plddt_combined_deepshap.csv"), index=False)

# === Save SHAP + input vector ===
np.save(os.path.join(cf_output, "shap_values_deepshap.npy"), shap_values)
np.save(os.path.join(cf_output, "input_vector.npy"), X)

print(f"✅ DeepSHAP Plot saved to {plot_path}")
print("✅ CSV saved to shap_plddt_combined_deepshap.csv")
print("✅ SHAP and input vector saved as .npy")

