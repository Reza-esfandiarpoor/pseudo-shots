import torch


models = {}
def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(name, **kwargs):
    if name is None:
        return None
    model = models[name](**kwargs)
    if torch.cuda.is_available():
        model.cuda()
    return model


def load(model_sv, name=None, test_model=None):
    if name is None:
        name = 'model'
    if test_model is not None:
        model = make(test_model, **model_sv[name + '_args'])
    else:
        model = make(model_sv[name], **model_sv[name + '_args'])

    model.load_state_dict(model_sv[name + '_sd'])

    return model

