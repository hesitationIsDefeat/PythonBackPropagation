from enum import Enum, auto


class ActivationFunctionName(Enum):
    SIGMOID = auto()
    RELU = auto()
    TANH = auto()

class LossFunctionName(Enum):
    DIFFERENCE_SQUARE= auto()