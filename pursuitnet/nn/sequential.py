from .module import Module
import pursuitnet as pn

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
            assert isinstance(x, pn.Tensor), f"Input to {module} is not a Tensor"  # Debugging statement
            x = module(x)
            assert isinstance(x, pn.Tensor), f"Output from {module} is not a Tensor"  # Debugging statement
        return x

    def __repr__(self):
        module_str = self.__class__.__name__ + '('
        for name, module in self._modules.items():
            module_str += '\n  (' + name + '): ' + repr(module)
        module_str += '\n)'
        return module_str
