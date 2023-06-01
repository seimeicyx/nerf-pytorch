from dataclasses import dataclass,field
from typing import List
import numpy as np
import torch
@dataclass(frozen=True)
class SceneMeta():
    H:int
    W:int
    K:List
    def get_scene(self):
        return self.H,self.W,self.K
@dataclass
class PinholeCamera():
    scene_meta:SceneMeta
    camera_pose:np.array
    def parse_scene(self):
        K=self.scene_meta.K
        fx=K[0][0]
        fy=K[1][1]
        x0=K[0][2]
        y0=K[1][2]
        return fx,fy,x0,y0
    def cast_ray(self):
        H,W,K=self.scene_meta.get_scene()
        c2w=torch.tensor(self.camera_pose)
        u,v=torch.meshgrid(torch.arange(W),torch.arange(H))
        u=u.t()
        v=v.t()
        fx,fy,x0,y0=self.parse_scene()
        dirs=torch.stack([(u-x0)/fx,-(v-y0)/fy,-torch.ones_like(u)],dim=-1)
        rays_d=torch.sum(dirs[...,None,:]*c2w[:3,:3],dim=-1)
        rays_o=c2w[:3,3].expand(rays_d.shape)
        return rays_o,rays_d