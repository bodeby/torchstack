from transformers import PreTrainedModel, PreTrainedConfig
from .models.causal_model import EnsembleModelForCausalLM


# wrapper for pretrained model
class EnsembleDistributable(PreTrainedModel):
    def __init__(
        self, config: PreTrainedConfig, ensemble_model: EnsembleModelForCausalLM
    ):
        super().__init__(config)
        self.ensemble_model = ensemble_model

    def forward(self, input_ids, attention_mask=None):
        return self.ensemble_model(input_ids, attention_mask=attention_mask)
