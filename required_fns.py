from vit.model import ViT


def get_model(name, hps):
    if name == 'vit':
        return ViT()
    else:
        raise ValueError(f"{name} does not exist")


