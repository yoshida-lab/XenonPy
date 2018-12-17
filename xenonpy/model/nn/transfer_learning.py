import torch as tc
import numpy as np
from pathlib import Path
from .checker import Checker


def retrieve_hidden_features(model, descriptors, max_layers=2):

    x_ = tc.from_numpy(descriptors).type(tc.FloatTensor)
    layers_val_train = []
    for l in model:
        x_ = l.layer(x_)
        layers_val_train.append(x_.data.numpy())

    ret = layers_val_train[-max_layers - 1:-1]
    return np.concatenate(ret, axis=1)


def retrieve_models(*props, name=None):
    for prop in props:
        p = Path(prop)
        models = [x for x in p.iterdir() if x.is_dir() and x.name != '.ipynb_checkpoints']
        for m in models:
            if not name:
                checker = Checker.load(m.name, m.parent)
                model = checker.trained_model
                model.cpu()
                yield model
            else:
                if m.name in name:
                    checker = Checker.load(m.name, m.parent)
                    model = checker.trained_model
                    model.cpu()
                    yield model
    return
