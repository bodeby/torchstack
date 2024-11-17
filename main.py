from consensus.configuration import Configuration
from consensus.member import AutoModelMember
from consensus.ensemble import Ensemble

import consensus as co


def main():
    config = Configuration(temperature=0.7)

    # Specialized Ensemble Member extending: AutoModelForCausalLM (transformers)
    m1 = AutoModelMember(weight=0.5).from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    m2 = AutoModelMember(weight=0.5).from_pretrained("Qwen/Qwen2.5-3B-Instruct")

    ensemble = Ensemble(config=config)

    # add ensemble members
    ensemble.add_member(m1)
    ensemble.add_member(m2)

    # add
    ensemble.build_vocabulary()

    print("hello world", config)

    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # sentence = "Hello World!"
    # tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
    # model = BertModel.from_pretrained("bert-large-uncased")

    # inputs = tokenizer(sentence, return_tensors="pt").to(device)
    # model = model.to(device)
    # outputs = model(**inputs)


if __name__ == "__main__":
    main()
