import operator

from .package import compare_version


TORCH_GREATER_EQUAL_2_0 = compare_version("torch", operator.ge, "2.0.0")
