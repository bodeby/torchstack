import pytest
from torchstack import EnsembleForCausalLM
from torchstack import AutoModelMember
from transformers import AutoTokenizer

from torchstack.strategies import GenerationAsClassification

@pytest.fixture
def setup_ensemble():
    m1 = AutoModelMember.from_pretrained("MODEL_ONE")
    m2 = AutoModelMember.from_pretrained("MODEL_TWO")
    t1 = AutoTokenizer.from_pretrained("MODEL_ONE")
    t2 = AutoTokenizer.from_pretrained("MODEL_TWO")
    strategy = GenerationAsClassification()
    ensemble = EnsembleForCausalLM(strategy=strategy, device="cpu")
    ensemble.add_member(m1, t1)
    ensemble.add_member(m2, t2)
    return ensemble

def test_ensemble_initialization(setup_ensemble):
    ensemble = setup_ensemble
    assert len(ensemble.members) == 2
    assert isinstance(ensemble.strategy, GenerationAsClassification)

def test_ensemble_generation(setup_ensemble):
    ensemble = setup_ensemble
    ensemble.prepare()
    result = ensemble.generate(prompt="Hello world")
    assert isinstance(result, str)
