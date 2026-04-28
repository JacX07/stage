import torch
import torch.nn.functional as F  # <-- NOUVEAU: Pour calculer la loss
import os
from typing import Dict, Optional, Any

class DIABETES:
    def __init__(self, data_path: str, eval_metadata: Any, **kwargs):
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0
        self.total_loss = 0.0  # <-- NOUVEAU: Pour accumuler la loss

    def begin_eval(self):
        self.reset()

    @property
    def required_outputs(self):
        return ['logits'] 

    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        if 'logits' not in preds or 'labels' not in batch:
            return

        # Redimensionnement correct pour la loss et l'accuracy
        logits = preds['logits'].view(-1, 2)  # Shape: (Batch, 2)
        targets = batch['labels'].view(-1).long()    # Shape: (Batch)

        # 1. Calcul de l'accuracy
        p_tensor = logits.argmax(dim=-1).cpu()
        targets_cpu = targets.cpu()
        self.correct += (p_tensor == targets_cpu).sum().item()
        self.total += targets.shape[0]

        # 2. Calcul de la loss (cross entropy)
        loss = F.cross_entropy(logits, targets, reduction='sum').item()
        self.total_loss += loss

    def result(self, save_path: Optional[str], rank: int, world_size: int, group=None):
        acc = self.correct / self.total if self.total > 0 else 0
        avg_loss = self.total_loss / self.total if self.total > 0 else 0  # <-- NOUVEAU
        
        if rank == 0:
            print(f"DIABETES TEST - ACCURACY: {acc:.2%} ({self.correct}/{self.total}) | LOSS: {avg_loss:.4f}")
            
        # On renvoie les deux métriques pour WandB
        return {
            "test/diabetes_accuracy": acc,
            "test/diabetes_loss": avg_loss
        }