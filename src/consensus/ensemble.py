import torch


class Ensemble(torch.nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = models

    def forward(self, input_ids, attention_mask=None):
        # Get outputs from each model
        outputs = [
            model(input_ids=input_ids, attention_mask=attention_mask)
            for model in self.models
        ]

        # Example of averaging logits across models
        logits = torch.stack([output.logits for output in outputs], dim=0)
        ensemble_logits = logits.mean(dim=0)

        return ensemble_logits
