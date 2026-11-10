"""ikgpc: indicator kriging as a first-order approximation of softmax
Gaussian process classification. Reproduction code for Li and You,
Mathematical Geosciences (revised manuscript)."""
from .kernels import Matern12, Matern32, RBF
from .softmax_gpc import SoftmaxGPC, laplace_mode
from .indicator_kriging import IndicatorKriging, one_hot
from .linearization import softmax, linearized_link, second_order_error, simple_ik_from_latent
from . import metrics

__version__ = "1.0.0"
__all__ = ["Matern12", "Matern32", "RBF", "SoftmaxGPC", "laplace_mode", "IndicatorKriging", "one_hot",
           "softmax", "linearized_link", "second_order_error", "simple_ik_from_latent", "metrics"]
