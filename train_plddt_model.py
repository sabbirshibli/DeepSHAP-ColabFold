# train_plddt_model.py

import os
import sys
import json
import torch
import numpy as np
from plddt_model import PlddtMLP

# ==== Inputs ====
fasta_file = sys.argv[1]
cf_output = sys.argv[2]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Helper Functions ====
def load_sequence(fasta_path):
    with open(fasta_path) as f:
        lines = f.readlines()
    return "".join([l.strip() for l in lines if not l.startswith(">")])

def one_hot_encode(seq):
    aa_dict = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    encoded = np.zeros((len(seq), 20))
    for i, aa in enumerate(seq):
        if aa in aa_dict:
            encoded[i, aa_dict[aa]] = 1
    return encoded.flatten()

# ==== Load Data ====
sequence = load_sequence(fasta_file)
X = one_hot_encode(sequence).astype(np.float32)
X_tensor = torch.tensor(X).unsqueeze(0).to(device)

json_file = [f for f in os.listdir(cf_output) if f.endswith(".json") and "_scores" in f][0]
with open(os.path.join(cf_output, json_file)) as f:
    plddt = np.array(json.load(f)["plddt"], dtype=np.float32)

# 🔥 Use full 456-length output, no slicing
y_tensor = torch.tensor(plddt).unsqueeze(0).to(device)

# ==== Build Model and Train ====
input_len = X.shape[0]
output_len = plddt.shape[0]  # This should now be 456
model = PlddtMLP(input_len, output_len).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

print("🧠 Training model...")
for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    output = model(X_tensor)
    loss = loss_fn(output, y_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1:>3} | Loss: {loss.item():.4f}")

torch.save(model.state_dict(), os.path.join(cf_output, "plddt_mlp_model.pt"))
print("✅ Model saved.")

