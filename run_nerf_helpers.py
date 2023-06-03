import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# #Positional encoding (section 5.1)
# class Embedder:
#     def __init__(self, **kwargs):
#         self.kwargs = kwargs
#         self.create_embedding_fn()
        
#     def create_embedding_fn(self):
#         embed_fns = []
#         d = self.kwargs['input_dims']
#         out_dim = 0
#         if self.kwargs['include_input']:
#             embed_fns.append(lambda x : x)
#             out_dim += d
            
#         max_freq = self.kwargs['max_freq_log2']
#         N_freqs = self.kwargs['num_freqs']
        
#         if self.kwargs['log_sampling']:
#             freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
#         else:
#             freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
#         for freq in freq_bands:
#             for p_fn in self.kwargs['periodic_fns']:
#                 embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
#                 out_dim += d
                    
#         self.embed_fns = embed_fns
#         self.out_dim = out_dim
        
#     def embed(self, inputs):
#         ans=[fn(inputs) for fn in self.embed_fns]
#         return torch.cat(ans, -1)


# def get_embedder(multires, i=0):
#     #输出编码的f(x),和总共的维度
#     if i == -1:
#         return nn.Identity(), 3
    
#     embed_kwargs = {
#                 'include_input' : True,
#                 'input_dims' : 3,
#                 'max_freq_log2' : multires-1,
#                 'num_freqs' : multires,
#                 'log_sampling' : True,
#                 'periodic_fns' : [torch.sin, torch.cos],
#     }
    
#     embedder_obj = Embedder(**embed_kwargs)
#     embed = lambda x, eo=embedder_obj : eo.embed(x)
#     return embed, embedder_obj.out_dim

class Embedder:#!!以后可以把x归一化一下，然后还乘以pi
    def __init__(self,**kwargs):
        self.kwargs=kwargs
        self.out_dim=0
        #self.fns=[]
        embed_fns=[]
        out_dim=0
        d=kwargs['input_dims']
        max_freq=kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        #是否添加原始数据
        if kwargs['include_input']:
            embed_fns.append(lambda x:x)
            out_dim+=d
        #系数列表
        #indexs=2.**torch.linspace(0., max_freq, steps=N_freqs)
        indexs=2.**np.arange(0,max_freq+1)
        #创建fn
        for freq in indexs:
            for per_fn in kwargs['periodic_fns']:
                embed_fns.append(lambda x,p_fn=per_fn,freq=freq:p_fn(freq*x))
                out_dim+=d
        self.embed_fns = embed_fns
        self.out_dim = out_dim
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns],-1)
        
        
def get_embedder(multires, i=0):
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    # embedder_obj = Embedder(**embed_kwargs)
    # embed = lambda x, eo=embedder_obj : eo.embed(x)
    # return embed, embedder_obj.out_dim
    embedder=Embedder(**embed_kwargs)
    embed=embedder.embed
    return embed, embedder.out_dim

#Model
class NeRF(nn.Module):
    def __init__(self,W=256,D=8,input_ch=3,input_ch_views=3,output_ch=4,skips=[4],use_viewdirs=False) -> None:
        super(NeRF,self).__init__()
        self.W=W
        self.D=D
        self.input_ch=input_ch
        self.input_ch_views=input_ch_views
        self.output_ch=output_ch
        self.skips=skips
        self.use_viewdirs=use_viewdirs
        self.pts_linears=nn.ModuleList([nn.Linear(input_ch,W)]+[nn.Linear(W,W) if i not in skips else nn.Linear(W+input_ch,W) for i in range(D-1)])
        self.feature_linear=nn.Linear(W,W)
        self.alpha_linear=nn.Linear(W,1)
        self.view_linear=nn.Linear(input_ch_views+W,W//2)
        self.rgb_linear=nn.Linear(W//2,3)
        self.output_linear=nn.Linear(W,output_ch)
        
    def forward(self,x):
        pts_input,views_input=torch.split(x,[self.input_ch,self.input_ch_views],-1)
        h=pts_input
        for i,layer in enumerate(self.pts_linears):
            h=layer(h)
            h=F.relu(h)
            if i in self.skips:
                h=torch.cat([pts_input,h],-1)
        if self.use_viewdirs:
            h=self.feature_linear(h)
            h=F.relu(h)
            alpha=self.alpha_linear(h)
            h=torch.cat([views_input,h],-1)
            h=self.view_linear(h)
            h=F.relu(h)
            rgb=self.rgb_linear(h)
            outputs=torch.cat([rgb,alpha],-1)
        else:
            outputs=self.output_linear(h)
        return outputs
# class NeRF(nn.Module):
#     def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
#         """ 
#         """
#         super(NeRF, self).__init__()
#         self.D = D
#         self.W = W
#         self.input_ch = input_ch
#         self.input_ch_views = input_ch_views
#         self.skips = skips
#         self.use_viewdirs = use_viewdirs
        
#         self.pts_linears = nn.ModuleList(
#             [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
#         ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
#         self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
        
#         ### Implementation according to the paper
#         # self.views_linears = nn.ModuleList(
#         #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
#         if use_viewdirs:
#             self.feature_linear = nn.Linear(W, W)
#             self.alpha_linear = nn.Linear(W, 1)
#             self.rgb_linear = nn.Linear(W//2, 3)
#         else:
#             self.output_linear = nn.Linear(W, output_ch)

#     def forward(self, x):
#         input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
#         h = input_pts
#         for i, l in enumerate(self.pts_linears):
#             h = self.pts_linears[i](h)
#             h = F.relu(h)
#             if i in self.skips:
#                 h = torch.cat([input_pts, h], -1)

#         if self.use_viewdirs:
#             alpha = self.alpha_linear(h)
#             feature = self.feature_linear(h)
#             h = torch.cat([feature, input_views], -1)
        
#             for i, l in enumerate(self.views_linears):
#                 h = self.views_linears[i](h)
#                 h = F.relu(h)

#             rgb = self.rgb_linear(h)
#             outputs = torch.cat([rgb, alpha], -1)
#         else:
#             outputs = self.output_linear(h)

#         return outputs    

    # def load_weights_from_keras(self, weights):
    #     assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
    #     # Load pts_linears
    #     for i in range(self.D):
    #         idx_pts_linears = 2 * i
    #         self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
    #         self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
    #     # Load feature_linear
    #     idx_feature_linear = 2 * self.D
    #     self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
    #     self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

    #     # Load views_linears
    #     idx_views_linears = 2 * self.D + 2
    #     self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
    #     self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

    #     # Load rgb_linear
    #     idx_rbg_linear = 2 * self.D + 4
    #     self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
    #     self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

    #     # Load alpha_linear
    #     idx_alpha_linear = 2 * self.D + 6
    #     self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
    #     self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))



# Ray helpers
def get_rays(H, W, K, c2w):
    #meshgrid(A,B),生成A*B的网格
    #torch.linspace(0, W-1, W) 从[0,W-1]中均匀采样W个
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)#（400,400,3）
    _dirs1=dirs[..., np.newaxis, :]
    _dirs2=_dirs1* c2w[:3,:3]
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(_dirs2, -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]#（400,400,3）
    _c2w=c2w[:3,-1]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = _c2w.expand(rays_d.shape)#（400,400,3）c2w[:3,-1](3,1)

    return rays_o, rays_d

#https://blog.csdn.net/guyuealian/article/details/104184551
def get_rays_np(H, W, K, c2w):
    #H：长 W：宽 K：相机参数，c2w：相机坐标系转化为世界坐标系的矩阵
    
    # 2D点到3D点的映射计算，[x,y,z]=[(u-cx)/fx,-(-v-cy)/fx,-1]
    # 在y和z轴均取相反数，因为nerf使用的坐标系x轴向右，y轴向上，z轴向外；
    # dirs的大小为(378, 504, 3)
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')# meshgrid函数将图像的坐标id分别取出存入i（列号）、j（行号），shape为（378,504）
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    # 将ray方向从相机坐标系转到世界坐标系，矩阵不变
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    # 相机原点在世界坐标系的坐标，同一个相机所有ray的起点；
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))# [1024,3]
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):

    # Get pdf
    weights = weights + 1e-5 # prevent nans[N_rays, N_samples]
    pdf = weights / torch.sum(weights, -1, keepdim=True)#归一化得到概率密度函数[N_rays, N_samples-2]
    cdf = torch.cumsum(pdf, -1)#累计密度函数[N_rays, N_samples]
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # 在前面添加一列0,[N_rays, N_samples-1]

    # Take uniform samples
    if det:#确定性抽样
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:#分层抽样
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])#[N_rays，N_samples]

    #u的shape:()
    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]#test中新的shape:[N_rays+N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)#[N_rays+N_samples]

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)#查找到cdf值大于或等于u的下标[N_rays，N_samples]
   
    below = torch.max(torch.zeros_like(inds-1), inds-1)#(N_rays, N_samples）
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)#(N_rays, N_samples）
    inds_g = torch.stack([below, above], -1)  # (N_rays, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]# =(N_rays, N_samples（u）, N_samples（一条射线）)

    #cdf_g的shape:# (N_rays, N_samples, 2)

    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    #bins_g的shape:# (N_rays, N_samples, 2)

    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
    

    denom = (cdf_g[...,1]-cdf_g[...,0])

    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples
