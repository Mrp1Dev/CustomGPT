from dataclasses import dataclass
from typing import Any
from abc import abstractmethod, ABC
import torch

@dataclass
class Config:
    n_embd = 768
    n_head = 12
    n_layer = 16
    mini_batch_size = 8 # B
    context_length = 2048 # T
    total_batch_size = 8*mini_batch_size*context_length # MUST BE DIVISBLE by (B*T)
    device = 'cuda'
    vocab_size = 32768

class Module(ABC):
    
    @abstractmethod
    def forward(self, x, **kwargs):
        pass    
    
    def __init__(self) -> None:
        super().__init__()
        self.buffers = dict()
        self.is_train = True
    
    def parameters(self):
        params = []
        def try_append_attribute(attr):    
            if isinstance(attr, torch.Tensor) and attr.requires_grad:
                params.append(attr)
                    
            if isinstance(attr, Module):
                params.extend(attr.parameters())
                
            if isinstance(attr, list):
                for item in attr:
                    try_append_attribute(item)
        for attribute in self.__dict__.values():
            try_append_attribute(attribute)
                    
        return params
    
    def state_dict(self, destination = None, prefix = ''):
        
        if destination is None:
            destination = dict()
        
        def handle_item(attr, current_key):
            if isinstance(attr, torch.Tensor) and attr.requires_grad:
                destination[current_key] = attr.detach()
                
            if isinstance(attr, Module):
                attr.state_dict(destination, current_key + '.')  
            
            if isinstance(attr, list):
                for i, item in enumerate(attr):      
                    handle_item(item, f'{current_key}.{i}')
            
        for name, value in self.__dict__.items():
            handle_item(value, prefix+name)
    
        return destination
    
    def load_state_dict(self, state_dict: dict, prefix = ''):
        
        def handle_item(attr, current_key):
            if isinstance(attr, torch.Tensor) and attr.requires_grad:
                if current_key in state_dict:
                    loaded_value = state_dict[current_key]
                    if attr.shape != loaded_value.shape:
                        print(f"Shape mismatch for {current_key}: expected {attr.shape}, got {loaded_value.shape}")
                    attr.data.copy_(loaded_value)
                else:
                    print(f"Key {current_key} not found in state_dict")
                
            if isinstance(attr, Module):
                attr.load_state_dict(state_dict, current_key + '.')  
            
            if isinstance(attr, list):
                for i, item in enumerate(attr):      
                    handle_item(item, f'{current_key}.{i}')
            
        for name, value in self.__dict__.items():
            handle_item(value, prefix+name)
    
    def to(self, device, dtype):
        def list_to(list):
            for i, item in enumerate(list):
                if isinstance(item, torch.Tensor):
                    was_param = item.requires_grad
                    moved = item.to(device=device, dtype=dtype)
                    if was_param:
                        moved = Param(moved)
                    list[i] = moved
                if isinstance(item, Module):
                    list[i] = item.to(device, dtype)
                if isinstance(item, list):
                    list_to(item)
        for key in self.__dict__.keys():
            attribute = getattr(self, key)
        
            if isinstance(attribute, torch.Tensor):
                was_param = attribute.requires_grad
                moved = attribute.to(device=device, dtype=dtype)
                if was_param:
                    moved = Param(moved)
                setattr(self, key, moved)
            if isinstance(attribute, Module):
                setattr(self, key,attribute.to(device, dtype))
            if isinstance(attribute, list):
                list_to(attribute)
        
        return self
        
    def __call__(self, x, **kwargs)->Any:
        return self.forward(x, **kwargs)
    
    def train(self):
        self._set_train_(True)
        return self
    def eval(self):
        self._set_train_(False)
        return self
        
    def _set_train_(self, new_train):
        self.is_train = new_train
        def recursive_set_train(attr):                
            if isinstance(attr, Module):
                attr._set_train_(new_train)
            if isinstance(attr, list):
                for item in attr:
                    recursive_set_train(item)
        for key in self.__dict__.keys():
            attribute = getattr(self, key)
            recursive_set_train(attribute)
    
def Param(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.requires_grad_(True)
