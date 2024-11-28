from abc import ABC, abstractmethod

class TokenizerAlignmentStrategy(ABC):
    @abstractmethod
    def align(self, tokenizers):
        """
        Align tokenizers and return the necessary mappings.
        Should return:
        - aligned_vocab: The shared vocabulary for alignment
        - mappings: Tokenizer-specific mappings to the aligned vocab
        """
        pass
