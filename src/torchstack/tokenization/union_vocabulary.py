from .base import TokenizerAlignmentStrategy


class UnionVocabularyStrategy(TokenizerAlignmentStrategy):
    def align(self, tokenizers):
        vocabularies = [set(tokenizer.get_vocab().keys()) for tokenizer in tokenizers]
        union_vocab = sorted(set().union(*vocabularies))  # Sorted for consistency
        union_vocab_index = {token: idx for idx, token in enumerate(union_vocab)}
        mappings = [
            {
                tokenizer.get_vocab()[token]: union_vocab_index[token]
                for token in tokenizer.get_vocab()
                if token in union_vocab_index
            }
            for tokenizer in tokenizers
        ]
        return union_vocab, mappings
