import torch

import os
from typing import Dict, Optional, Any

class DIABETES:
    def __init__(self, data_path: str, eval_metadata: Any, **kwargs):
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

    def begin_eval(self):
        self.reset()

    @property
    def required_outputs(self):
        return ['logits'] 

    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        if 'logits' not in preds or 'labels' not in batch:
            return

        # preds['logits'] est de shape (Batch, 1, 2). On prend l'argmax sur la dernière dim
        p_tensor = preds['logits'].argmax(dim=-1).view(-1).cpu()
        targets = batch['labels'].view(-1).cpu()

        self.correct += (p_tensor == targets).sum().item()
        self.total += targets.shape[0]

    def result(self, save_path: Optional[str], rank: int, world_size: int, group=None):
        acc = self.correct / self.total if self.total > 0 else 0
        if rank == 0:
            print(f"DIABETES ACCURACY: {acc:.2%} ({self.correct}/{self.total})")
        return {"test/diabetes_accuracy": acc}