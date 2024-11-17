from consensus.configuration import Configuration
from consensus.member import AutoModelMember
from consensus.ensemble import Ensemble


def main():
    config = Configuration(temperature=0.7)
    m1 = AutoModelMember(weight=0.5).from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    m2 = AutoModelMember(weight=0.5).from_pretrained("Qwen/Qwen2.5-3B-Instruct")

    ensemble = Ensemble(config=config)
    
    # add ensemble members
    ensemble.add_member(m1)
    ensemble.add_member(m2)

    ensemble.add_member("")

    # add 

    print("hello world", config)


if __name__ == "__main__":
    main()
