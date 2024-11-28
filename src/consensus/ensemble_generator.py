from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from typing import List, Tuple, Dict

import torch
import logging
import warnings
import re

# local import
from torchstack.configuration import EnsembleConfig


class EnsembleGenerator:
    def __init__(
        self,
        models: List[AutoModelForCausalLM],
        tokenizers: List[AutoTokenizer],
        config: EnsembleConfig = None,
    ):
        if len(models) != len(tokenizers):
            raise ValueError("Number of models must match number of tokenizers")

        # Definitions
        self.models = models
        self.tokenizers = tokenizers
        self.config = config or EnsembleConfig()

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Validate device availability
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
        if self.config.device == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA requested but not available. Using CPU instead.")

        # Set up padding tokens
        self._setup_padding()

        # Create vocabulary mapping between models
        self.vocab_mappings = self._create_vocab_mappings()

        # Move models to device
        self._prepare_models()

    # Setup padding tokens for each tokenizer.
    def _setup_padding(self):
        for tokenizer in self.tokenizers:
            # If tokenizer doesn't have a pad token, use eos token
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    # Last resort: add a new padding token
                    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Move models to specified device and set to evaluation mode.
    # FIXME: moving all models to GPU at once is not necessary
    def _prepare_models(self) -> None:
        """"""
        for model in self.models:
            model.to(self.device)
            model.eval()

    # Create mappings between each model's vocabulary and the first model's vocabulary.
    def _create_vocab_mappings(self) -> List[Dict[int, int]]:
        """"""
        mappings = []
        base_tokenizer = self.tokenizers[0]
        base_vocab = base_tokenizer.get_vocab()

        for tokenizer in self.tokenizers:
            current_vocab = tokenizer.get_vocab()
            mapping = {}

            for token, idx in current_vocab.items():
                if token in base_vocab:
                    mapping[idx] = base_vocab[token]

            mappings.append(mapping)

        return mappings

    # Pad input sequences to the same length and create attention masks.
    def _pad_inputs(
        self, token_ids: List[List[int]]
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Returns:
            Tuple[List[torch.Tensor], torch.Tensor]: Padded inputs and attention mask
        """
        max_length = max(len(ids) for ids in token_ids)
        padded_inputs = []
        attention_masks = []

        for idx, (tokenizer, ids) in enumerate(zip(self.tokenizers, token_ids)):
            padding_length = max_length - len(ids)
            pad_token_id = tokenizer.pad_token_id

            # Pad the sequence
            padded_sequence = ids + [pad_token_id] * padding_length
            attention_mask = [1] * len(ids) + [0] * padding_length

            padded_inputs.append(torch.tensor([padded_sequence], device=self.device))
            attention_masks.append(torch.tensor([attention_mask], device=self.device))

        return padded_inputs, attention_masks

    # Align logits from a model to the vocabulary space of the first model.
    def _align_logits(self, logits: torch.Tensor, model_idx: int) -> torch.Tensor:
        if model_idx == 0:
            return logits

        mapping = self.vocab_mappings[model_idx]
        base_vocab_size = len(self.tokenizers[0].get_vocab())
        aligned_logits = torch.full(
            (logits.shape[0], logits.shape[1], base_vocab_size),
            float("-inf"),
            device=logits.device,
        )

        for src_idx, tgt_idx in mapping.items():
            # can be left here.
            if src_idx < logits.shape[-1]:
                aligned_logits[:, :, tgt_idx] = logits[:, :, src_idx]

        return aligned_logits

    # Check if a token is a special token.

    # TODO: Be very careful when cleaning token, example: [ĠParis] -> Paris
    def _is_special_token(self, token: str) -> bool:
        # Define patterns for special tokens
        special_patterns = [
            r"^\s+$",  # Only whitespace
            r"\\n",  # Newlines
            r"[^\w\s]",  # Special characters
        ]

        return any(re.search(pattern, token) for pattern in special_patterns)

    # Clean a token by removing leading/trailing spaces if configured.
    def _clean_token(self, token: str) -> str:
        if self.config.strip_spaces:
            return token.strip()
        return token

    @torch.no_grad()
    def _compute_ensemble_logits(
        self,
        token_ids: List[List[int]],
        padded_inputs: List[torch.Tensor] = None,
        attention_masks: List[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute and combine logits from all models, aligning vocabularies."""
        base_vocab_size = len(self.tokenizers[0].get_vocab())

        total_logits = torch.zeros(
            self.config.batch_size,
            padded_inputs[0].shape[1],
            base_vocab_size,
            device=self.device,
        )

        valid_model_count = torch.zeros(
            (self.config.batch_size, padded_inputs[0].shape[1], base_vocab_size),
            device=self.device,
        )

        # NOT SURE if i have to pad the inputs. !!! Because of different sizes in tokenized prompts
        for idx, (model, inputs, attention_mask) in enumerate(
            zip(self.models, padded_inputs, attention_masks)
        ):
            try:
                # Process input with attention mask
                outputs = model(inputs, attention_mask=attention_mask)
                logits = outputs.logits

                # Align logits with base vocabulary
                aligned_logits = self._align_logits(logits, idx)

                # Apply temperature scaling ! NICE TO HAVE, not needed
                if self.config.temperature != 1.0:
                    aligned_logits = aligned_logits / self.config.temperature

                # TODO: If the union vocab, then we dont need the mask
                # Add to total logits where valid (not -inf)
                mask = aligned_logits != float("-inf")
                total_logits[mask] += aligned_logits[
                    mask
                ]  # FIXME: We have to average Probabilities using the logits, not the logits.
                valid_model_count[mask] += 1

            except Exception as e:
                self.logger.warning(f"Error processing model {idx}: {str(e)}")
                continue

        # Average logits by the number of valid predictions for each token
        valid_model_count = torch.clamp(valid_model_count, min=1)
        return total_logits / valid_model_count

    def generate(
        self,
        prompt: str,
        custom_top_k: int = None,
        min_probability: float = None,
        filter_special: bool = None,
        strip_spaces: bool = None,
    ) -> List[Tuple[str, float]]:
        """
        Generate ensemble predictions for the given prompt.

        Args:
          prompt: Input text prompt
          custom_top_k: Override default top_k value
          min_probability: Override default minimum probability threshold
          filter_special: Override default special token filtering
          strip_spaces: Override default space stripping behavior
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")

        # Use provided parameters or fall back to config defaults
        filter_special = (
            filter_special
            if filter_special is not None
            else self.config.filter_special_tokens
        )
        strip_spaces = (
            strip_spaces if strip_spaces is not None else self.config.strip_spaces
        )
        min_prob = (
            min_probability
            if min_probability is not None
            else self.config.min_probability
        )

        try:
            # Encode prompt with each tokenizer
            token_ids = [
                tokenizer.encode(prompt, add_special_tokens=True)
                for tokenizer in self.tokenizers
            ]

            # Pad inputs and create attention masks
            padded_inputs, attention_masks = self._pad_inputs(token_ids)

            # Get ensemble logits
            averaged_logits = self._compute_ensemble_logits(token_ids, attention_masks)
            # averaged_logits = self._compute_ensemble_logits(padded_inputs, attention_masks)

            # Convert to probabilities (use only the last token)
            probs = torch.nn.functional.softmax(averaged_logits, dim=-1)
            last_token_probs = probs[0, -1, :].cpu().numpy()

            # Get token-probability pairs
            base_tokenizer = self.tokenizers[0]
            token_prob_pairs = []

            for idx, prob in enumerate(last_token_probs):
                if prob >= min_prob:
                    token = base_tokenizer.decode([idx])

                    # Apply filtering if enabled
                    if filter_special and self._is_special_token(token):
                        continue

                    # Clean token if enabled
                    cleaned_token = self._clean_token(token)
                    if cleaned_token:  # Skip empty tokens
                        token_prob_pairs.append((cleaned_token, float(prob)))

            # Sort by probability and get top-k
            token_prob_pairs.sort(key=lambda x: x[1], reverse=True)
            k = min(custom_top_k or self.config.top_k, len(token_prob_pairs))

            return token_prob_pairs[:k]

        except Exception as e:
            self.logger.error(f"Generation failed: {str(e)}")
            raise RuntimeError(f"Generation failed: {str(e)}")