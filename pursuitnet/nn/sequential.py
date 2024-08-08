from pursuitnet.nn.module import Module

class Sequential(Module):
    def __init__(self, *args):
        super(Sequential, self).__init__()
        self._modules = {}
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def add_module(self, name, module):
        if not isinstance(module, Module):
            raise TypeError("{} is not a Module subclass".format(type(module)))
        self._modules[name] = module

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

    @property
    def layers(self):
        return list(self._modules.values())

    def __repr__(self):
        module_str = self.__class__.__name__ + '('
        for idx, module in enumerate(self._modules.values()):
            module_str += f'\n  ({idx}): {repr(module)}'
        module_str += '\n)'
        return module_str