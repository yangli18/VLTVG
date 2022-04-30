import os.path as osp
import sys

from importlib import import_module


class Config:
    def __init__(self, config_file):
        cfg_dict = self.file2dict(config_file)
        self.__setattr__('_cfg_dict', cfg_dict)

    def file2dict(self, config_file):
        config_dir = osp.dirname(osp.abspath(config_file))
        config_name = osp.basename(config_file)
        module_name = osp.splitext(config_name)[0]

        sys.path.insert(0, config_dir)
        mod = import_module(module_name)
        sys.path.pop(0)

        cfg_dict = {
            name: value
            for name, value in mod.__dict__.items()
            if not name.startswith('__')
        }
        del sys.modules[module_name]
        return cfg_dict

    def merge_to_args(self, args):
        for k, v in self._cfg_dict.items():
            assert k in args.__dict__.keys(), f'Argument {k} is not defined'
            args.__dict__.update({k: v})


    def __repr__(self):
        return self._cfg_dict.__repr__()

    def __getitem__(self, item):
        return self._cfg_dict.__getitem__(item)

    def __getattr__(self, name):
        return self._cfg_dict[name]