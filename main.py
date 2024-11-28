from transformers import AutoTokenizer, AutoConfig

# from torchstack import Tokenizer # todo: since AutoTokenizer is not inheritable just use base.
from torchstack import AutoModelMember
from torchstack import Configuration
from torchstack import Ensemble

from torchstack import EnsembleModelForCausalLM
from torchstack import HFEnsembleModel

from huggingface_hub import login

# sub-level imports
# from torchstack.voting import AverageAggregator
# from torchstack.tokenization import UnionVocabularyStrategy
# from torchstack.tokenization import ProjectionStrategy

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
    ensemble.create_union_vocab()  # INVOKED IN THE add_member METHOD

    # create mappings from local to union vocabulary
    ensemble.create_tokenizer_mapping(t1)  # INVOKED IN THE add_member METHOD
    ensemble.create_tokenizer_mapping(t2)  # INVOKED IN THE add_member METHOD

    # generate response with ensemble
    response = ensemble.generate(prompt="Finish this sentence: The quick brown ...")

    print(ensemble)  # repr print for ensemble setup
    print(response)  # responses from transformer ensemble

    ### EXAMPLE WORKFLOW FOR TRAINING AND PUSHING MODEL

    # Save the custom model
    ensemble_model = EnsembleModelForCausalLM(
        model_names=["meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct"],
        weights=[0.6, 0.4],
    )
    config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    hf_model = HFEnsembleModel(config, ensemble_model)

    # Authenticate with Hugginface Before
    login(token="<insert_token>")
    hf_model.save_pretrained("./smol-llama-1.5b-ensemble")


if __name__ == "__main__":
    main()
