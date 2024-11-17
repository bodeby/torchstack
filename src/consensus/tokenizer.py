from transformers import AutoTokenizer

class Tokenizer(AutoTokenizer):
    def __init__(self):
        super().__init__()