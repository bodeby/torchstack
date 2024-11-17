from consensus import AutoModelMember
from consensus import Configuration
from consensus import Ensemble

def main():
    # Specialized Ensemble Member extending: AutoModelForCausalLM (transformers)
    # m1 = AutoModelMember(weight=0.5).from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    # m2 = AutoModelMember(weight=0.5).from_pretrained("Qwen/Qwen2.5-3B-Instruct")

    c = Configuration(temperature=0.7)
    ensemble = Ensemble(config=c)

    # add ensemble members
    #ensemble.add_member(m1)
    # ensemble.add_member(m2)

    print("hello world")

if __name__ == "__main__":
    main()
