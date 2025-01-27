from transformers import AutoTokenizer, AutoConfig

from torchstack import AutoModelMember
from torchstack import Configuration
from torchstack import Ensemble

# sub-level imports
from torchstack.strategies import GenerationAsClassification

# for model distribution
from torchstack import EnsembleModelForCausalLM
from torchstack import EnsembleDistributable

from huggingface_hub import login

# constants
MODEL_ONE = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_TWO = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_ENS = "Qwen-2.5-Llama-3.2-cag-ensemble"

def main():
    # Setup specialized Ensemble Member extending: AutoModelForCausalLM (transformers)
    m1 = AutoModelMember.from_pretrained(MODEL_ONE)
    m2 = AutoModelMember.from_pretrained(MODEL_TWO)

    # Setup specialized Ensemle Tokenizers extending: AutoTokenizer (transformers)
    t1 = AutoTokenizer.from_pretrained(MODEL_ONE)
    t2 = AutoTokenizer.from_pretrained(MODEL_TWO)


    config = Configuration(temperature=0.7, voting_stragety="average_voting")
    strategy = GenerationAsClassification()
    ensemble = Ensemble(config=config, strategy=strategy)

    # add ensemble members
    ensemble.add_member(model=m1, tokenizer=t1)
    ensemble.add_member(model=m2, tokenizer=t2)

    # prepare model for usage
    ensemble.prepare()

    # generate response with ensemble
    response = ensemble.generate(prompt="Finish this sentence: The quick brown ...")
    print(response)  # responses from transformer ensemble

    ### EXAMPLE WORKFLOW FOR TRAINING AND PUSHING MODEL

    # Save the custom model
    ensemble_model = EnsembleModelForCausalLM(
        model_names=["meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct"],
        weights=[0.6, 0.4],
    )
    
    config = AutoConfig.from_pretrained(MODEL_ONE)
    hf_model = EnsembleDistributable(config, ensemble_model) # Needs a new name

    # Authenticate with Hugginface Before
    login(token="<insert_token>")
    hf_model.save_pretrained(MODEL_ENS)
    hf_model.push_to_hub(f"frederikbode/{MODEL_ENS}")


if __name__ == "__main__":
    main()
