import os
import sys
import subprocess

fasta_path = sys.argv[1]
output_dir = sys.argv[2]

# 🔧 Fix for disk quota: redirect model download cache
os.environ["XDG_CACHE_HOME"] = "/storage/data2/ahmed_sibli/.cache"

print(f"📡 Running ColabFold on {fasta_path}...")

command = [
    "colabfold_batch",
    fasta_path,
    output_dir,
    "--model-type", "alphafold2_ptm",
    "--num-recycle", "3",
    "--rank", "plddt"
]

subprocess.run(command, check=True)
print("✅ ColabFold prediction complete.")
