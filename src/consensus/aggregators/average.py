from consensus.aggregators.aggregator import Aggregator


class Average(Aggregator):

    def __init__(self) -> None:
        super().__init__()
