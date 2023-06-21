from abc import ABC
import numpy as np
from torch.nn import Module
from typing import Callable

class AttackBase(ABC):
    def __init__(self, target_generator, loss, model) -> None:
        self.__target_generator = target_generator
        self.__loss = loss
        self.__model = model
        pass

    def __disable_weights_grad(self):
        # this will stop the calculation for the graident with respect to input which is faster 
        for param in self.__model.parameters():
            param.requires_grad = False

    @property
    def model(self):
        return self.__model
    
    @property
    def loss(self):
        return self.__loss
    
    @property
    def target_generator(self):
        return self.__target_generator

    @model.setter
    def model(self, value:Module):
        assert not isinstance(value, type(Module)), f"model must be of type{type(Module)} "
        self.__model = value

    @loss.setter
    def loss(self, value):
        # assert not isinstance(value, type(Module)), f"model must be of type{type(Module)} "
        self.__loss = value

    @target_generator.setter
    def target_generator(self, value:Callable):
        assert not isinstance(value, type(Callable)), f"target_generator must be of type{type(Callable)} "
        self.__target_generator = value
    
    def generate_attack(
            self,
            images,
            targets=None,
            eps=4, 
            return_numpy=False,
            ):
        self.__disable_weights_grad()


