import torch.optim as optim


def get_SGD(model_params, **kwargs):
    return optim.SGD(model_params, **kwargs)


def get_Adam(model_params, **kwargs):
    return optim.Adam(model_params, **kwargs)


OPTIMIZERS = {"SGD": get_SGD,
              "Adam": get_Adam}
