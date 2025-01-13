from dataclasses import dataclass

@dataclass
class Configuration:
    """Configuration for ensemble generation."""
    top_k: int = 10
    min_probability: float = 0.001
    batch_size: int = 1
    pad_token_id: int = None
    filter_special_tokens: bool = True  # New parameter
    strip_spaces: bool = True  # New parameter
    device: str = "cpu"
    temperature: float = 1.0
    voting_stragety: str = "average_voting"
