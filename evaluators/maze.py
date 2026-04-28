import torch
import torch.distributed as dist
import os
from typing import Dict, Optional, Any

class MAZE_HARD:
    def __init__(self, data_path: str, eval_metadata: Any, **kwargs):
        self.metadata = eval_metadata
        self.reset()
        # On utilise print uniquement sur le rank 0 pour ne pas polluer les logs
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f">>> MazeHardEvaluator initialisé")

    def reset(self):
        # Compteurs pour l'exact match (le labyrinthe entier est correct)
        self.all_correct_count = 0
        self.all_total_count = 0
        
        # Optionnel : Compteurs pour la précision par token/pixel si besoin
        self.token_correct_count = 0
        self.token_total_count = 0

    def begin_eval(self):
        self.reset()

    @property
    def required_outputs(self):
        # On demande les prédictions au modèle
        return ['preds'] 

    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        """
        Met à jour les statistiques pour le batch courant.
        Compare la solution prédite (chemin) avec la solution cible (ground truth).
        """
        # --- SÉCURITÉ ---
        if not preds:
            return

        # 1. Récupération des prédictions
        if 'preds' in preds:
            p_tensor = preds['preds']
        elif 'output' in preds:
            p_tensor = preds['output']
        elif 'logits' in preds:
            # Si logits [Batch, SeqLen, Vocab] ou [Batch, Channels, H, W]
            p_tensor = preds['logits'].argmax(dim=-1)
        else:
            try:
                p_tensor = next(iter(preds.values()))
            except StopIteration:
                return 

        local_preds = p_tensor.detach().cpu()

        # 2. Récupération de la vérité terrain (Target/Labels)
        # Pour Maze, 'labels' contient généralement le chemin optimal
        if 'labels' in batch:
            local_targets = batch['labels'].detach().cpu()
        elif 'target' in batch:
            local_targets = batch['target'].detach().cpu()
        elif 'output' in batch: # Parfois la target est dans output pour les autoencodeurs
            local_targets = batch['output'].detach().cpu()
        else:
            return

        # 3. Alignement des dimensions
        # Maze peut être 2D (Image) ou 1D (Séquence de tokens)
        # On aplatit tout en [Batch, N] pour comparer ligne par ligne
        if local_preds.shape != local_targets.shape:
            # Gestion basique des mismatches (ex: extra dimension 1)
            local_preds = local_preds.view(local_preds.shape[0], -1)
            local_targets = local_targets.view(local_targets.shape[0], -1)
        else:
            # Aplatissement standard
            local_preds = local_preds.view(local_preds.shape[0], -1)
            local_targets = local_targets.view(local_targets.shape[0], -1)

        # 4. Calcul local de l'exactitude (Exact Match)
        # Comparaison élément par élément
        match_element = (local_preds == local_targets)
        
        # Le labyrinthe est considéré "Résolu" (Correct) seulement si 
        # TOUT le chemin est identique à la target (all(dim=-1))
        match_row = match_element.all(dim=-1).int()

        self.all_correct_count += match_row.sum().item()
        self.all_total_count += match_row.shape[0]
        
        # Optionnel : mise à jour des stats par token
        self.token_correct_count += match_element.sum().item()
        self.token_total_count += match_element.numel()

    def result(self, save_path: Optional[str], rank: int, world_size: int, group: Optional[dist.ProcessGroup] = None) -> Optional[Dict[str, float]]:
        
        # 1. Préparation des tenseurs sur GPU pour la communication
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Metriques "Instance" (Labyrinthe entier)
        total_correct = torch.tensor([self.all_correct_count], dtype=torch.float64, device=device)
        total_samples = torch.tensor([self.all_total_count], dtype=torch.float64, device=device)
        
        # Metriques "Token" (Pixel/Case individuel)
        token_correct = torch.tensor([self.token_correct_count], dtype=torch.float64, device=device)
        token_total = torch.tensor([self.token_total_count], dtype=torch.float64, device=device)

        # 2. Agrégation Multi-GPU
        if dist.is_initialized() and world_size > 1:
            dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
            dist.all_reduce(token_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(token_total, op=dist.ReduceOp.SUM)

        # 3. Calcul final
        val_samples = total_samples.item()
        val_tokens = token_total.item()
        
        # Accuracy "Exact Match" (La plus importante pour Maze Hard)
        accuracy = (total_correct.item() / val_samples) if val_samples > 0 else 0.0
        
        # Accuracy "Token" (Pour le debug, souvent très haute même si le chemin est faux)
        token_accuracy = (token_correct.item() / val_tokens) if val_tokens > 0 else 0.0

        if rank == 0:
            print("-" * 40)
            print(f"MAZE HARD EVAL RESULTS:")
            print(f"Total Mazes Checked: {int(val_samples)}")
            print(f"Perfectly Solved Mazes: {int(total_correct.item())}")
            print(f"EXACT MATCH ACCURACY: {accuracy:.2%}")
            print(f"Token/Pixel Accuracy: {token_accuracy:.2%}") # Utile pour voir si le modèle apprend
            print("-" * 40)
            
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                with open(os.path.join(save_path, "maze_results.txt"), "w") as f:
                    f.write(f"Instance Accuracy: {accuracy}\n")
                    f.write(f"Token Accuracy: {token_accuracy}\n")
                    f.write(f"Correct Mazes: {total_correct.item()}\n")
                    f.write(f"Total Mazes: {val_samples}")

        return {
            "test/maze_exact_accuracy": accuracy,
            "test/maze_token_accuracy": token_accuracy
        }