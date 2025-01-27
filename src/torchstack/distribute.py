from transformers import PreTrainedModel, PretrainedConfig
from .models.causal_model import EnsembleModelForCausalLM
import torch


class EnsembleConfig(PretrainedConfig):
    def __init__(self, model_configs, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.model_configs = model_configs
        self.vocab_size = vocab_size


class PreTrainedEnsemble(PreTrainedModel):
    config_class = EnsembleConfig

    def __init__(self, config, ensemble_model):
        super().__init__(config)
        self.ensemble_model = ensemble_model

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return self.ensemble_model(input_ids, attention_mask=attention_mask, **kwargs)

    def save_pretrained(self, save_directory):
        # Save config
        self.config.save_pretrained(save_directory)

        # Save ensemble logic
        torch.save(self.ensemble_model, f"{save_directory}/ensemble_model.pt")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Load config
        config = EnsembleConfig.from_pretrained(pretrained_model_name_or_path)

        # Load ensemble logic
        ensemble_model = torch.load(
            f"{pretrained_model_name_or_path}/ensemble_model.pt"
        )

        return cls(config, ensemble_model)


# wrapper for pretrained model
class EnsembleDistributable(PreTrainedModel):
    def __init__(
        self, config: PretrainedConfig, ensemble_model: EnsembleModelForCausalLM
    ):
        super().__init__(config)
        self.ensemble_model = ensemble_model

    def forward(self, input_ids, attention_mask=None):
        return self.ensemble_model(input_ids, attention_mask=attention_mask)
