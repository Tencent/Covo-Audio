import torch
from collections import OrderedDict


class JsonHParams():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = JsonHParams(**v)
            if type(v) == str and v.lower() in ["non", "none", "nil", "null"]:
                v = None
            self[k] = v

    def to_dict(self):
        return self

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()

    def pop(self, key):
        return self.__dict__.pop(key)

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def set(self, key, value):
        return setattr(self, key, value)

    def exist(self, key):
        return hasattr(self, key)


def load_ckpt(model, model_path, map_location="cpu"):
    state_dict = torch.load(model_path, map_location=map_location)
    state_dict = state_dict['model'] if 'model' in state_dict.keys() else state_dict
    clean_dict = OrderedDict()

    for k, v in state_dict.items():
        if k.startswith('module.'):
            clean_dict[k[7:]] = v
        else:
            clean_dict[k] = v
    
    model.load_state_dict(clean_dict)
    
    return model


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d

