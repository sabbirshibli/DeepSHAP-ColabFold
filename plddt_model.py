# plddt_model.py

import torch
import torch.nn as nn

class PlddtMLP(nn.Module):
    def __init__(self, input_len, output_len):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_len, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_len)  # 🔥 Output matches full pLDDT length
        )

    def forward(self, x):
        return self.net(x)

def load_trained_model(input_len, output_len, model_path, device):
    model = PlddtMLP(input_len, output_len).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

