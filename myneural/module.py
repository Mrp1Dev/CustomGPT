from typing import Any
import torch
from abc import abstractmethod, ABC
from dataclasses import dataclass
@dataclass
class Config:
    n_embd = 192
    n_head = 6
    context_length = 256
    vocab_size = 65
    n_layer = 6
    batch_size = 64
    device = 'cuda'


class Module(ABC):
    
    @abstractmethod
    def forward(self, x, **kwargs):
        pass    
    
    def __init__(self) -> None:
        super().__init__()
        self.buffers = dict()
    
    def parameters(self):
        params = []
        for attribute in self.__dict__.values():
            def try_append_attribute(attr):    
                if isinstance(attr, Param):
                    params.append(attr)
                        
                if isinstance(attr, Module):
                    params.extend(attr.parameters())
                    
                if isinstance(attr, list):
                    for item in attr:
                        try_append_attribute(item)
            try_append_attribute(attribute)
                    
        return params
    
    def register_buffer(self, key, buffer: torch.Tensor):
        '''make sure to name key as the variable you are setting it to.'''
        self.buffers[key] = buffer
        return buffer
    
    def to(self, device, dtype):
        for key in self.__dict__.keys():
            attribute = getattr(self, key)
        
            if isinstance(attribute, Param):
                setattr(self, key,Param(attribute.to(device=device, dtype=dtype)))
            if isinstance(attribute, Module):
                setattr(self, key,attribute.to(device, dtype))
            if isinstance(attribute, list):
                def list_to(list):
                    for i, item in enumerate(list):
                        if isinstance(item, Param):
                            list[i] = Param(item.to(device=device, dtype=dtype))
                        if isinstance(item, Module):
                            list[i] = item.to(device, dtype)
                        if isinstance(item, list):
                            list_to(item)
                list_to(attribute)
                
        for key in self.buffers.keys():
            new_buffer = self.buffers[key].to(device=device, dtype=dtype)
            self.buffers[key] = new_buffer 
            if hasattr(self, key):
                setattr(self, key, new_buffer)
        return self
        
            
    def __call__(self, x, **kwargs)->Any:
        return self.forward(x, **kwargs)
    
    
class Param(torch.Tensor):
    
    def __new__(cls, data, **kwargs):
        if isinstance(data, torch.Tensor):
            data = data.detach()
        return torch.Tensor.as_subclass(data, cls, **kwargs).requires_grad_(True)
        