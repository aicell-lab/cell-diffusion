from src.models.gcmol import GCMol

# Create model on CPU (default):
model = GCMol(model_name="models/gcmol")

# Or create on CUDA if available:
# model = GCMol(model_name="my_gcmol_model", device="cuda")

# Encode a list of SMILES (you get a CPU tensor back):
smiles_list = ["CCO", "c1ccc(cc1)C(=O)Nc1ccc(cc1)N"]
embeddings = model.mol_encode(smiles_list)

print("Embeddings shape:", embeddings.shape)
print("First embedding (first 10 dims):", embeddings[0, :10])
