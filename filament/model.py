"""Single model factory; saved ModifiedUNet is always the default."""
from unet import ModifiedUNet, DoubleConv


def build_model(config: dict):
    options = dict(config)
    name = options.pop("name", "modified")
    if name == "modified": return ModifiedUNet(**options)
    if name == "full_image":
        from .experimental import FullImageUNet
        return FullImageUNet(**options)
    raise ValueError(f"Unknown model {name}")
