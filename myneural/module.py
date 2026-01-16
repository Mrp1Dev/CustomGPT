import torch
from abc import abstractmethod, ABC
from dataclasses import dataclass

@dataclass
class MyConfig:
    n_embd = 768,
    n_head = 4


class Module(ABC):
    
    @abstractmethod
    def forward(self, x):
        pass    
    
    def parameters(self):
        params = []
        for attribute in self.__dict__.values():
            def try_append_attribute(attr):    
                if isinstance(attr, torch.Tensor) and attr.requires_grad:
                    params.append(attr)
                        
                if isinstance(attr, Module):
                    params.extend(attr.parameters())
            try_append_attribute(attribute)
            if isinstance(attribute, list):
                for item in attribute:
                    try_append_attribute(item)
                    
        return params
    
class Param(torch.Tensor):
    
    def __new__(cls, data, **kwargs):
        return torch.Tensor.as_subclass(data, cls).requires_grad_(True)
        