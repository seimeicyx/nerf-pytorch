from dataclasses import dataclass,field
from typing import Any,List
import numpy as np
import torch
@dataclass
class FreEncoder():
    L:int
    out_dim:int=0
    fns:List=field(default_factory=list,init=False)
    def __post_init__(self) -> Any:
        self.fns=[]
        fres=[torch.sin,torch.cos]
        self.fns.append(lambda x:x)
        self.out_dim+=3
        for i in torch.range(0,self.L-1):
            for fre in fres:
                self.fns.append(lambda x:fre(2.**i*x*torch.pi))
                self.out_dim+=3
    def __call__(self,inputs:Any) -> Any:
        return torch.cat([fn(inputs) for fn in self.fns],dim=-1)