import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from common import PuzzleDatasetMetadata

def build_diabetes():
    output_dir = "data/diabetes"
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

    # 1. Charger les données (assurez-vous d'avoir le fichier diabete.csv à la racine)
    df = pd.read_csv("diabetes.csv")
    
    # Séparer features (X) et target (y)
    X = df.drop('Outcome', axis=1).values.astype(np.float32)
    y = df['Outcome'].values.astype(np.int64)

    # 2. Normalisation (Très important pour les réseaux de neurones avec données continues)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 3. Train/Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def save_split(X_data, y_data, split_name):
        n_samples = len(X_data)
        
        results = {
            "inputs": X_data,
            "labels": y_data.reshape(-1, 1),
            "puzzle_identifiers": np.zeros(n_samples, dtype=np.int32),
            "puzzle_indices": np.arange(n_samples + 1, dtype=np.int32), # <-- CORRIGÉ (+ 1)
            "group_indices": np.arange(n_samples + 1, dtype=np.int32)
        }
        
        save_dir = os.path.join(output_dir, split_name)
        for k, v in results.items():
            np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
            
        return n_samples

    n_train = save_split(X_train, y_train, "train")
    n_test = save_split(X_test, y_test, "test")

    # 4. Métadonnées
    metadata = PuzzleDatasetMetadata(
        seq_len=X.shape[1],      # <-- S'ADAPTE AUTOMATIQUEMENT A VOTRE CSV
        vocab_size=2,
        pad_id=0,
        ignore_label_id=-100,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=n_train,
        mean_puzzle_examples=1,
        total_puzzles=n_train,
        sets=["all"]
    )

    with open(os.path.join(output_dir, "train", "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
    with open(os.path.join(output_dir, "test", "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    print("Dataset Diabète généré avec succès !")

if __name__ == "__main__":
    build_diabetes()