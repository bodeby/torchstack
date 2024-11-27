from transformers import AutoTokenizer

# from consensus import Tokenizer # todo: since AutoTokenizer is not inheritable just use base.
from consensus import AutoModelMember
from consensus import Configuration
from consensus import Ensemble

# sub-level imports
# from consensus.voting import AverageAggregator
# from consensus.tokenization import UnionVocabularyStrategy
# from consensus.tokenization import ProjectionStrategy

MODEL_ONE = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_TWO = "Qwen/Qwen2.5-1.5B-Instruct"


def main():
    # Setup specialized Ensemble Member extending: AutoModelForCausalLM (transformers)
    m1 = AutoModelMember.from_pretrained(MODEL_ONE)
    m2 = AutoModelMember.from_pretrained(MODEL_TWO)

    # Setup specialized Ensemle Tokenizers extending: AutoTokenizer (transformers)
    t1 = AutoTokenizer.from_pretrained(MODEL_ONE)
    t2 = AutoTokenizer.from_pretrained(MODEL_TWO)

    config = Configuration(temperature=0.7, voting_stragety="average_voting")
    ensemble = Ensemble(config)

    # add ensemble members
    ensemble.add_member(model=m1, tokenizer=t1)
    ensemble.add_member(model=m2, tokenizer=t2)
    # ensemble.add_remote_member(url="") # TODO: Implemented remote model usage

    # create common vocabulary
    ensemble.create_union_vocab()

    # create mappings from local to union vocabulary
    # TODO: This should be injected in ensemble class
    ensemble.create_tokenizer_mapping(t1)
    ensemble.create_tokenizer_mapping(t2)

    # generate response with ensemble
    response = ensemble.generate(prompt="Finish this sentence: The quick brown ...")

    print(ensemble)  # repr print for ensemble setup
    print(response)  # responses from transformer ensemble


if __name__ == "__main__":
    main()
