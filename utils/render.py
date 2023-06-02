import torch
import torch.nn.functional as F
def raw2outputs(raw:torch.Tensor,z_vals:torch.Tensor,rays_d:torch.Tensor):
    rgb=raw[...,:3]
    delta=raw[...,-1]
    dists=z_vals[...,1:]-z_vals[...,:-1]
    dists=torch.cat([torch.Tensor([1e-10]).expand(dists[...,:1].shape),dists],dim=-1)
    dists*=torch.norm(rays_d[...,None,:],dim=-1)
    
    alpha=1-torch.exp(-delta*dists)
    weights=alpha*torch.cumprod(torch.cat([torch.ones_like(alpha[...,:1]),1.-alpha+1e-10],dim=-1),dim=-1)[...,:-1]
    rgb_map=torch.sum(rgb*weights[...,None],dim=-2)
    depth_map=torch.sum(weights*z_vals,dim=-1)
    acc_map=torch.sum(weights,dim=-1)
    disp_map=1./torch.max((depth_map/acc_map),1e-10*torch.ones_like(depth_map))
    
    #backgroundcolor
    rgb_map=rgb_map+(1-acc_map[...,None])*1.0
    
    return rgb_map, disp_map, acc_map, weights, depth_map
