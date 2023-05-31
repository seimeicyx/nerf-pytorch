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
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))



# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


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
    #sample_pdf用于从一组权重和对应的离散化直方图中生成一定数量的样本点的函数，直方图指的是：将每条射线从近到远分成若干个深度区间，
    # 每个区间的密度值进行归一化，我猜这里的区间与密度值应该是分层采样得到的，粗糙网络的权重已经被训练好了。
    
    #bins边界
    #weights 每个区间对应权重[N_rays, N_samples]
    #N_samples 需要生成的样本数量
    #det 是否以确定性的方式生成样本
    #pytest 是否在运行测试时使用该函数
    
    # Get pdf
    weights = weights + 1e-5 # prevent nans[N_rays, N_samples]
    pdf = weights / torch.sum(weights, -1, keepdim=True)#归一化得到概率密度函数[N_rays, N_samples]
    cdf = torch.cumsum(pdf, -1)#累计密度函数[N_rays, N_samples]
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # 在前面添加一列0,[N_rays, N_samples+1]

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
    u = u.contiguous()#重新拷贝当前变量，断开依赖,首先，u 被转化为连续的张量，这一步可以确保之后的计算中能够正确使用 PyTorch 的 API 
    inds = torch.searchsorted(cdf, u, right=True)#查找到cdf值大于或等于u的下标[N_rays，N_samples]
    #接下来，根据下标 inds，找到比 u 小的最大柱子位置和比 u 大的最小柱子位置，并将两者合并成一个形状为 (batch, N_samples, 2) 的张量 inds_g。
    below = torch.max(torch.zeros_like(inds-1), inds-1)#(N_rays, N_samples）
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)#(N_rays, N_samples）
    inds_g = torch.stack([below, above], -1)  # (N_rays, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    
    #然后，使用 torch.gather 函数将直方图柱子位置和权重按照 inds_g 中的下标进行匹配，得到形状为 (batch, N_samples, 2) 的 bins_g 和 cdf_g。
    #ds_g 指定的位置处的取值，而 bins_g 表示对应位置的直方图柱子的位置。
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]# =(N_rays, N_samples, N_samples)
    #cdf.unsqueeze(1)的shape:[N_rays, 1,N_samples]
    #cdf.unsqueeze(1).expand(matched_shape)的shape:[N_rays, N_samples,N_samples]
    #inds_g的shape:[N_rays, N_samples,2]
    #gather的作用：inds_g目的是指示下标位置，得到两个[N_rays, N_samples]的张量值，
    #cdf_g的shape:# (N_rays, N_samples, 2)
    # 一层是累计密度函数恰好小于u的cdf张量值，
    # 一层是累计密度函数大于等于u的cdf张量值，
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    #bins_g的shape:# (N_rays, N_samples, 2)
    # 一层是累计密度函数恰好小于u的边界，
    # 一层是累计密度函数大于等于u的边界，
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
    
    #最后，根据 cdf_g 和 bins_g，计算出采样点的位置 samples，并返回结果。
    denom = (cdf_g[...,1]-cdf_g[...,0])
    #在计算 denom 时，需要进行特判，避免分母为 0 的情况。,
    # 这里使用 torch.where 函数来判断分母是否小于一个阈值（这里的阈值为 1e-5），
    # 如果小于阈值，则将分母的值替换为 1。这一步操作是为了避免分母为 0 的情况，从而导致采样出现错误。
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples
