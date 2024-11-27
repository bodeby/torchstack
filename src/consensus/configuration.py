from pydantic import Enum
from dataclasses import dataclass


class VotingTypes(str, Enum):
    average = ("average_voting",)
    majority = "majority_average_voting"


@dataclass
class Configuration:
    """Configuration for ensemble generation."""

    top_k: int = 10
    device: str = "cuda"
    temperature: float = 1.0
    min_probability: float = 0.001
    batch_size: int = 1
    pad_token_id: int = None
    filter_special_tokens: bool = True  # New parameter
    strip_spaces: bool = True  # New parameter
    voting_stragety: VotingTypes
