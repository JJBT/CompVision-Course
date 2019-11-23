import pretrainedmodels
import torch

model = pretrainedmodels.models.pnasnet5large(num_classes=1000)
print(model)


def make_pnasnet():
    import pretrainedmodels
    model = pretrainedmodels.pnasnet5large(num_classes=1000)
    model.last_linear = torch.nn.Linear(model.last_linear.in_features, 2)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model


def make_xception():
    import pretrainedmodels
    model = pretrainedmodels.xception()
    model.last_linear = torch.nn.Linear(model.last_linear.in_features, 2)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model


def make_resnext():
    import pretrainedmodels
    model = pretrainedmodels.se_resnext50_32x4d()
    model.last_linear = torch.nn.Linear(model.last_linear.in_features, 2)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model
