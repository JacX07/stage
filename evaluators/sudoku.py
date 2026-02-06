import torch
import torch.distributed as dist
import os
from typing import Dict, Optional, Any

class SUDOKU:
    def __init__(self, data_path: str, eval_metadata: Any, **kwargs):
        self.metadata = eval_metadata
        self.reset()
        # On utilise print uniquement sur le rank 0 pour ne pas polluer les logs
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f">>> SudokuEvaluator initialisé")

    def reset(self):
        self.all_correct_count = 0
        self.all_total_count = 0

    def begin_eval(self):
        self.reset()

    @property
    def required_outputs(self):
        # --- CORRECTION IMPORTANTE ICI ---
        # On DEMANDE au modèle de renvoyer les prédictions.
        # Essayez 'preds' en priorité. Si ça ne marche pas, tentez 'logits' ou 'output'.
        return ['preds'] 

    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        # --- SÉCURITÉ ---
        if not preds:
            # Si le dict est vide, on ne fait rien pour éviter le crash StopIteration
            return

        # 1. Récupération des prédictions
        # On cherche la clé disponible
        if 'preds' in preds:
            p_tensor = preds['preds']
        elif 'output' in preds:
            p_tensor = preds['output']
        elif 'logits' in preds:
            # Si ce sont des logits, on prend l'argmax pour avoir les chiffres
            p_tensor = preds['logits'].argmax(dim=-1)
        else:
            # Fallback : on prend la première valeur disponible
            try:
                p_tensor = next(iter(preds.values()))
            except StopIteration:
                return # Toujours vide, on sort

        local_preds = p_tensor.detach().cpu()

        # 2. Récupération de la vérité terrain (Target/Labels)
        if 'labels' in batch:
            local_targets = batch['labels'].detach().cpu()
        elif 'output' in batch:
            local_targets = batch['output'].detach().cpu()
        elif 'target' in batch:
            local_targets = batch['target'].detach().cpu()
        else:
            # Pas de labels trouvés, on ne peut pas évaluer ce batch
            return

        # 3. Calcul local de l'exactitude (Exact Match)
        # Gestion des dimensions : [Batch, 9, 9] -> [Batch, 81]
        if local_preds.shape != local_targets.shape:
            local_preds = local_preds.view(local_preds.shape[0], -1)
            local_targets = local_targets.view(local_targets.shape[0], -1)

        # Comparaison
        match_element = (local_preds == local_targets)
        # Une grille est bonne seulement si TOUTE la ligne est True
        match_row = match_element.view(match_element.shape[0], -1).all(dim=-1).int()

        self.all_correct_count += match_row.sum().item()
        self.all_total_count += match_row.shape[0]

    def result(self, save_path: Optional[str], rank: int, world_size: int, group: Optional[dist.ProcessGroup] = None) -> Optional[Dict[str, float]]:
        
        # 1. Préparation des tenseurs sur GPU pour la communication
        # (Nécessaire pour dist.all_reduce)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        total_correct = torch.tensor([self.all_correct_count], dtype=torch.float64, device=device)
        total_samples = torch.tensor([self.all_total_count], dtype=torch.float64, device=device)

        # 2. Agrégation Multi-GPU
        if world_size > 1:
            dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

        # 3. Calcul final
        val_samples = total_samples.item()
        accuracy = (total_correct.item() / val_samples) if val_samples > 0 else 0.0

        if rank == 0:
            print("-" * 40)
            print(f"SUDOKU EVAL RESULTS:")
            print(f"Total Puzzles Checked: {int(val_samples)}")
            print(f"Correct Puzzles: {int(total_correct.item())}")
            print(f"EXACT MATCH ACCURACY: {accuracy:.2%}")
            print("-" * 40)
            
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                with open(os.path.join(save_path, "sudoku_results.txt"), "w") as f:
                    f.write(f"Accuracy: {accuracy}\nCorrect: {total_correct.item()}\nTotal: {val_samples}")

        return {"test/sudoku_accuracy": accuracy}