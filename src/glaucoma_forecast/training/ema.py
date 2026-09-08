"""Exponential moving average utilities."""

from __future__ import annotations


class EMA:
    def __init__(self, model, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {name: param.detach().clone() for name, param in model.named_parameters() if param.requires_grad}

    def update(self, model) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)
