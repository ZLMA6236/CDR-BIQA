
from .swin_transformer import swintransformer_tiny
from .resnet import resnet50
from .convnext import convnext_tiny

def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'swin':
        model = swintransformer_tiny(pretrained=True)
    elif model_type == 'resnet':
        model = resnet50(pretrained=True, model_root=None)
    elif model_type == 'convnext':
        model = convnext_tiny(pretrained=True, in_22k=False, )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")
    return model
