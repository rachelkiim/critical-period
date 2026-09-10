"""Model registry and factory helpers."""

from src.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from src.models.densenet import densenet121, densenet161, densenet169, densenet201

MODEL_REGISTRY = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "densenet121": densenet121,
    "densenet161": densenet161,
    "densenet169": densenet169,
    "densenet201": densenet201,
}


def get_model(name, **kwargs):
    """Create a model by name.

    Args:
        name: Model name (e.g. 'resnet18', 'densenet121').
        **kwargs: Passed to the model constructor (e.g. num_classes=10).

    Returns:
        nn.Module instance.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](**kwargs)
