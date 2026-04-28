import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


from utils.functions import load_model_class, get_model_source_path
# ==========================================
# 1. DÉFINITION DES SONDES (PROBES)
# ==========================================
class SudokuProbes(nn.Module):
    def __init__(self, puzzle_emb_len=16, hidden_size=512):
        super().__init__()
        # On aplatit les 16 jetons de 512 dimensions -> vecteur de 8192
        self.input_dim = puzzle_emb_len * hidden_size
        
        # Sonde 1 : Régression - Deviner le nombre de cases vides
        self.probe_empty_cells = nn.Linear(self.input_dim, 1)
        
        # Sonde 2 : Classification binaire - Le puzzle respecte-t-il les règles ?
        # (Aucun conflit ligne/colonne/carré)
        self.probe_valid_rules = nn.Linear(self.input_dim, 1) 
        
        # Sonde 3 : Classification binaire - Difficulté (ex: > 50 cases vides = Difficile)
        self.probe_difficulty = nn.Linear(self.input_dim, 1)

    def forward(self, z_H_puzzle):
        # z_H_puzzle shape: [Batch, 16, 512]
        batch_size = z_H_puzzle.size(0)
        
        # Aplatir le brouillon : [Batch, 8192]
        x = z_H_puzzle.reshape(batch_size, -1)
        
        pred_empty = self.probe_empty_cells(x)
        pred_valid = self.probe_valid_rules(x)
        pred_diff  = self.probe_difficulty(x)
        
        return pred_empty, pred_valid, pred_diff


# ==========================================
# 2. FONCTIONS DE CALCUL DES LABELS (VÉRITÉ TERRAIN)
# ==========================================
def count_empty_cells(boards):
    # boards: Tensor [Batch, 9, 9] (les cases vides sont souvent des 0)
    return (boards == 0).sum(dim=(1, 2)).float().unsqueeze(1)

def check_sudoku_rules_batch(boards):
    """
    Vérifie si les grilles respectent les règles du Sudoku.
    Retourne 1.0 si valide (aucun conflit), 0.0 si invalide.
    """
    batch_size = boards.size(0)
    validity = torch.ones(batch_size, 1, dtype=torch.float32, device=boards.device)
    
    for b in range(batch_size):
        board = boards[b]
        is_valid = True
        
        # Vérification des lignes et colonnes
        for i in range(9):
            row = board[i, :]
            col = board[:, i]
            # On ignore les 0 (cases vides)
            row_vals = row[row > 0]
            col_vals = col[col > 0]
            if len(row_vals) != len(torch.unique(row_vals)) or \
               len(col_vals) != len(torch.unique(col_vals)):
                is_valid = False
                break
                
        # Vérification des blocs 3x3
        if is_valid:
            for i in range(0, 9, 3):
                for j in range(0, 9, 3):
                    block = board[i:i+3, j:j+3].flatten()
                    block_vals = block[block > 0]
                    if len(block_vals) != len(torch.unique(block_vals)):
                        is_valid = False
                        break
        
        validity[b, 0] = 1.0 if is_valid else 0.0
        
    return validity


import random

def corrupt_sudoku_batch(boards_9x9):
    """
    Prend un batch [B, 9, 9] de sudokus valides.
    Corrompt 50% d'entre eux en créant un conflit (doublon sur une ligne).
    Retourne les grilles modifiées et le vecteur de labels (1=Valide, 0=Invalide).
    """
    corrupted_boards = boards_9x9.clone()
    batch_size = boards_9x9.size(0)
    
    # Par défaut, tout est valide (1.0)
    labels_rules = torch.ones(batch_size, 1, dtype=torch.float32, device=boards_9x9.device)

    for b in range(batch_size):
        if random.random() < 0.5: # 50% de chance de "casser" la grille
            labels_rules[b, 0] = 0.0 # On note pour la sonde que cette grille est fausse
            
            # On cherche toutes les cases non-vides (chiffres > 0)
            non_empty_indices = torch.nonzero(corrupted_boards[b])
            if len(non_empty_indices) < 2:
                continue 
            
            # On prend une case remplie au hasard
            r1, c1 = random.choice(non_empty_indices)
            valeur_a_dupliquer = corrupted_boards[b, r1, c1].item()
            
            # On choisit une autre colonne sur la MÊME ligne pour forcer une erreur
            c_erreur = (c1.item() + random.randint(1, 8)) % 9
            
            # On écrase la case avec notre doublon
            corrupted_boards[b, r1, c_erreur] = valeur_a_dupliquer
            
    return corrupted_boards, labels_rules

# ==========================================
# 3. BOUCLE D'ENTRAÎNEMENT DES SONDES
# ==========================================
# ==========================================
# 3. BOUCLE D'ENTRAÎNEMENT DES SONDES
# ==========================================
def train_probes(model, train_dataloader, device):
    # 1. Geler le modèle de base ! TRÈS IMPORTANT
    model.eval() # Mode évaluation pour désactiver Dropout/BatchNorm
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. Initialiser les sondes
    probes = SudokuProbes(puzzle_emb_len=16, hidden_size=512).to(device)
    optimizer = optim.Adam(probes.parameters(), lr=1e-3)
    
    # Critères (Loss)
    loss_mse = nn.MSELoss() # Pour compter les cases
    loss_bce = nn.BCEWithLogitsLoss() # Pour les classifications Oui/Non
    
    print("Début de l'entraînement des sondes...")
    
    for epoch in range(10):
        # Initialisation des compteurs pour l'époque
        total_loss_empty = 0.0
        total_loss_rules = 0.0
        total_correct_rules = 0
        total_samples = 0
        num_batches = 0
        
        for set_name, batch, global_batch_size in train_dataloader:
            num_batches += 1
            
            # 1. On récupère les grilles d'origine (shape: [Batch, 81])
            inputs = batch["inputs"].to(device) 
            inputs_9x9 = inputs.reshape(-1, 9, 9)
            
            # 2. ---> LE SABOTAGE <---
            inputs_9x9_corrupted, labels_rules = corrupt_sudoku_batch(inputs_9x9)
            
            # On remet sous forme [Batch, 81]
            batch["inputs"] = inputs_9x9_corrupted.reshape(-1, 81)
            
            # 3. Calcul des cases vides
            labels_empty = count_empty_cells(inputs_9x9_corrupted)
            
            # 4. Faire passer la donnée dans le modèle gelé
            with torch.no_grad():
                carry = model.initial_carry(batch)
                
                for _ in range(16):
                    carry, outputs = model(carry, batch)
                    if carry.halted.all():
                        break
                
                z_H_final = carry.inner_carry.z_H[:, :16, :].detach()

            # 5. Entraîner les sondes
            optimizer.zero_grad()
            pred_empty, pred_valid, pred_diff = probes(z_H_final)
            
            loss_empty = loss_mse(pred_empty, labels_empty)
            loss_rules = loss_bce(pred_valid, labels_rules) 

            # --- CALCUL DE L'ACCURACY ---
            with torch.no_grad():
                predictions_rules = (torch.sigmoid(pred_valid) > 0.5).float()
                correct_rules = (predictions_rules == labels_rules).sum().item()
                batch_samples = labels_rules.size(0)
            
            # Rétropropagation
            loss = loss_empty + loss_rules 
            loss.backward()
            optimizer.step()
            
            # --- ACCUMULATION POUR L'ÉPOQUE ---
            total_loss_empty += loss_empty.item()
            total_loss_rules += loss_rules.item()
            total_correct_rules += correct_rules
            total_samples += batch_samples
            
        # --- STATISTIQUES DE FIN D'ÉPOQUE ---
        avg_loss_empty = total_loss_empty / num_batches
        avg_loss_rules = total_loss_rules / num_batches
        epoch_accuracy_rules = (total_correct_rules / total_samples) * 100 # En pourcentage
        
        # Le fameux print mis à jour !
        print(f"Epoch {epoch} | Loss Empty: {avg_loss_empty:.4f} | Loss Rules: {avg_loss_rules:.4f} | Accuracy Validité: {epoch_accuracy_rules:.2f}%")

    return probes


