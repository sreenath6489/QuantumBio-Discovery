import urllib.request

print("Downloading sample background pool...")
# Using a stable, publicly hosted drug-like SMILES sample file
url = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/sider.csv"

# Let's download a clean public sample or use a direct raw dataset link
# Alternatively, use this direct ZINC/ChEMBL subset link:
url_alt = "https://raw.githubusercontent.com/awosika/cheminformatics/main/data/actives_decoy.smi"

try:
    urllib.request.urlretrieve("https://cheminfo.org/flavor/malaria/Download/top_10000_drug_like.smi", "pool.smi")
    print("Success! Created 'pool.smi'.")
except Exception as e:
    print(f"Trying alternative URL...")
    urllib.request.urlretrieve("https://raw.githubusercontent.com/MandADeck/DecoyFinder/master/test/test_database.smi", "pool.smi")
    print("Success! Created 'pool.smi' using alternative source.")