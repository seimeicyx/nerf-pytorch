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
def sample_pdf(bins:torch.Tensor,weights:torch.Tensor,Nf_samples:int,isTesting):    
    #weights(N_ray,N_sample-2)
    weights=weights+1e-5
    a=torch.sum(weights,dim=-1,keepdim=True)
    b=torch.sum(weights,dim=-1)
    pdf=weights/torch.sum(weights,dim=-1,keepdim=True)#(N_ray,Nc_sample-2)
    cdf=torch.cat([torch.zeros_like(pdf[:,:1]),torch.cumsum(pdf,dim=-1)],dim=-1)#(N_ray,Nc_sample-1)
    
    #sample
    u=torch.rand([cdf.shape[0],Nf_samples])#(N_ray,Nf_sample)
    
    #find cdfs&bins
    u.contiguous()
    inds=torch.searchsorted(cdf,u,right=True)#(N_ray,Nf_sample)
    lower_inds=torch.max(torch.zeros_like(inds),inds-1)
    upper_inds=torch.min(torch.ones_like(inds)*(cdf.shape[-1]-1),inds)
    bound_inds=torch.stack([lower_inds,upper_inds],dim=-1)
    
    _size=[inds.shape[0],inds.shape[1],cdf.shape[-1]]#[N_ray,Nf_sample,Nc_sample]
    bins_g=torch.gather(bins.unsqueeze(1).expand(_size),-1,bound_inds)#(N_ray,Nf_sample,2)
    cdfs_g=torch.gather(cdf.unsqueeze(1).expand(_size),-1,bound_inds)#(N_ray,Nf_sample,2)
    diff_cdf=cdfs_g[...,1]-cdfs_g[...,0]
    diff_cdf=torch.where(diff_cdf<1e-5,torch.ones_like(diff_cdf),diff_cdf)
    samples=bins_g[...,0]+((u-cdfs_g[...,0])/diff_cdf)*(bins_g[...,1]-bins_g[...,0])
    return samples