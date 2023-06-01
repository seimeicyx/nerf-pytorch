import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.args import NeRFTrainingArgs
from copy import deepcopy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class NeRF(nn.Module):
    def __init__(self,pts_ch=3,view_ch=3,D=8,W=256,skips=[4]):
        super(NeRF,self).__init__()
        self.pts_ch=pts_ch
        self.view_ch=view_ch
        self.D=D
        self.W=W
        self.skips=skips
        #pts part
        self.pts_linears=nn.ModuleList([nn.Linear(self.pts_ch,self.W)]+
                                       [nn.Linear(self.W,self.W) if i not in self.skips
                                        else nn.Linear(self.W+self.pts_ch,self.W) 
                                        for i in range(self.D-1)])
        #view_dir part
        self.feature=nn.Linear(self.W,self.W)
        self.out_alpha=nn.Linear(self.W,1)
        self.view_linear=nn.Linear(self.W+self.view_ch,self.W//2)
        self.out_rgb=nn.Linear(self.W//2,3)
    def forward(self,x):
        pts_x,view_x=torch.split(x,[self.pts_ch,self.view_ch],dim=-1)
        h=pts_x
        for i,linear in enumerate(self.pts_linears):
            h=linear(h)
            h=F.relu(h)
            if i in self.skips:
                h=torch.cat([pts_x,h],dim=-1)
        #view
        alpha=self.out_alpha(h)
        h=self.feature(h)
        h=torch.cat([h,view_x],dim=-1)
        
        h=self.view_linear(h)
        h=F.relu(h)
        rgb=self.out_rgb(h)
        return torch.cat([rgb,alpha],dim=-1)

def batch_run_model(pts,views,model,chunk):
    rets=[]
    for i in range(0,pts.shape[0],chunk):
        rets.append(model(torch.cat([pts[i:i+chunk],views[i:i+chunk]],dim=-1)))
    return torch.cat(rets,dim=0)
def apply_model(input_pts,input_views,pts_embed_fn,view_embed_fn,model,batch_chunk):
    pts_flat=torch.reshape(input_pts,shape=([-1,input_pts.shape[-1]]))
    pts_embeded=pts_embed_fn(pts_flat)
    views_exp=input_views[:,None,:].expand(input_pts.shape)
    views_flat=torch.reshape(views_exp,shape=([-1,views_exp.shape[-1]]))
    views_embeded=view_embed_fn(views_flat)
    outputs=batch_run_model(pts_embeded,views_embeded,model,batch_chunk)
    return torch.reshape(outputs,shape=(list(input_pts.shape[:-1])+[outputs.shape[-1]]))

def create_nerf(args):
    from models.encoder import FreEncoder
    from models.nerf import NeRF
    pts_embedded=FreEncoder(args.multires)
    view_embedded=FreEncoder(args.multires_views)
    corse_model=NeRF(pts_ch=pts_embedded.out_dim,
                    view_ch=view_embedded.out_dim,
                    D=args.netdepth,W=args.netwidth,skips=[4]).to(device)
    fine_model=NeRF(pts_ch=pts_embedded.out_dim,
                    view_ch=view_embedded.out_dim,
                    D=args.netdepth_fine,W=args.netwidth_fine,skips=[4]).to(device)
    grad_vars=[]
    grad_vars+=list(corse_model.parameters())
    grad_vars+=list(fine_model.parameters())
    optimizer=torch.optim.Adam(params=grad_vars,lr=args.lrate,betas=[0.9,0.999])
    network_query_fn=lambda inputs,\
                            viewdirs,\
                            network_fn,\
                            apply_model=apply_model,\
                            pts_embed_fn=pts_embedded,\
                            view_embed_fn=view_embedded,\
                            batch_chunk=args.netchunk:\
                                apply_model(input_pts=inputs,
                                            input_views=viewdirs,
                                            pts_embed_fn=pts_embed_fn,
                                            view_embed_fn=view_embed_fn,
                                            model=network_fn,
                                            batch_chunk=batch_chunk)
    start=0
    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : fine_model,
        'N_samples' : args.N_samples,
        'network_fn' : corse_model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
        'ndc':False,
        'lindisp':args.lindisp
    }   
    
    # render_kwargs_test=deepcopy(render_kwargs_train)
    # render_kwargs_test['network_query_fn'] = network_query_fn
    # render_kwargs_test['network_fine']=render_kwargs_test['network_fine'].to(device)
    # render_kwargs_test['network_fn']=render_kwargs_test['network_fn'].to(device)
    # render_kwargs_test['network_fine'] = fine_model
    # render_kwargs_test['network_fn'] = corse_model
    
    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train,render_kwargs_test,start,grad_vars,optimizer   