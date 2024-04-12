from Model.model import CatClassificationModel
from Model.VGG16 import VGG16


def print_model_parameters(model):
    for name, parameter in model.named_parameters():
        print(f"{name}: {parameter.numel()}")


model_dict = {
    'CatClassificationModel': CatClassificationModel,
    'VGG16': VGG16,
}
