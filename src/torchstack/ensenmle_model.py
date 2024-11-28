from transformers import PreTrainedModel

# wrapper for pretrained model
class HFEnsembleModel(PreTrainedModel):
    def __init__(self, config, ensemble_model):
        super().__init__(config)
        self.ensemble_model = ensemble_model

    def forward(self, input_ids, attention_mask=None):
        return self.ensemble_model(input_ids, attention_mask=attention_mask)
