import os
import nbformat
from nbclient import NotebookClient

notebooks = [
    "notebooks/01_eksplorasi_dataset.ipynb",
    "notebooks/02_segmentasi_otsu_kmeans.ipynb",
    "notebooks/03_preprocessing.ipynb",
    "notebooks/04_ekstraksi_fitur.ipynb",
    "notebooks/05_feature_scaling.ipynb",
    "notebooks/06_svm_training.ipynb",
    "notebooks/07_evaluasi.ipynb"
]

for nb_path in notebooks:
    print(f"Executing {nb_path} ...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": "notebooks"}})
    client.execute()
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Finished {nb_path} successfully!")

print("ALL 7 NOTEBOOKS EXECUTED AND UPDATED SUCCESSFULLY!")
