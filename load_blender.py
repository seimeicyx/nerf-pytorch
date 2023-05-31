import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2

trans_t=lambda t: torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()
rot_phi=lambda phi:torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi),np.cos(phi),0], 
    [0,0,0,1]]).float()
rot_theta= lambda theta:torch.Tensor([
    [np.cos(theta),0,-np.sin(theta),0],
    [0,1,0,0],
    [np.sin(theta),0,np.cos(theta),0],
    [0,0,0,1]]).float()
def pose_spherical(theta, phi, radius):
    c2w=trans_t(radius)
    c2w=rot_phi(phi/180.*np.pi)@c2w
    c2w=rot_theta(theta/180.*np.pi)@c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w
def load_blender_data(basedir, half_res=False, testskip=1):
    #读入json数据
    splits=["test",
           "train",
           "val"]
    meta={}
    for item in splits:
        path=os.path.join(basedir,"transforms_{}.json".format(item))
        with open(path,'r') as fp:
            meta[item]=json.load(fp)
    #解析数据，包括读入原始图片,camera_fovx,pose
    imgs=[]
    poses=[]
    indexs=[[-1,-1]]
    
    for item in splits:

        frames=meta[item]["frames"]
        if item=="train":
            skip=1
        else:
            skip=testskip
        for frame in frames[::skip]:
            file_path=frame['file_path']
            #image
            img=imageio.imread(os.path.join(basedir,file_path)+".png")
            img=np.array(img).astype(np.float32)/255.
            imgs.append(img)
            pose=np.array(frame["transform_matrix"]).astype(np.float32)
            poses.append(pose)
        indexs.append([indexs[-1][1]+1,len(imgs)])
    imgs=np.array(imgs)
    poses=np.array(poses)
        
    #i_split
    i_split=[np.arange(indexs[i][0],indexs[i][1]) for i in range(1,len(indexs))]
    H,W=imgs[0].shape[:2]
    #resize images
    if half_res:
        H=H//2
        W=W//2
        res_imgs=np.zeros((imgs.shape[0],H,W,4))
        for i in range(imgs.shape[0]):
            res_imgs[i]=cv2.resize(imgs[i],(W,H),interpolation=cv2.INTER_AREA)
        imgs=res_imgs
    camera_angle_x=meta[item]["camera_angle_x"]
    focal=0.5*W/np.tan(0.5*camera_angle_x)
    render_poses=torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,41)[:-1]],dim=0)
            #camera
    #处理图片,camera_fovx,pose至需要的结果
    #获取渲染的相机c2w
    return imgs, poses, render_poses, [int(H), int(W), focal], i_split
# trans_t = lambda t : torch.Tensor([
#     [1,0,0,0],
#     [0,1,0,0],
#     [0,0,1,t],
#     [0,0,0,1]]).float()

# rot_phi = lambda phi : torch.Tensor([
#     [1,0,0,0],
#     [0,np.cos(phi),-np.sin(phi),0],
#     [0,np.sin(phi), np.cos(phi),0],
#     [0,0,0,1]]).float()

# rot_theta = lambda th : torch.Tensor([
#     [np.cos(th),0,-np.sin(th),0],
#     [0,1,0,0],
#     [np.sin(th),0, np.cos(th),0],
#     [0,0,0,1]]).float()


# def pose_spherical(theta, phi, radius):
#     c2w = trans_t(radius)
#     c2w = rot_phi(phi/180.*np.pi) @ c2w
#     c2w = rot_theta(theta/180.*np.pi) @ c2w
#     c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
#     return c2w


# def load_blender_data(basedir, half_res=False, testskip=1):
#     splits = ['train', 'val', 'test']
#     metas = {}
#     for s in splits:
#         with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
#             metas[s] = json.load(fp)

#     all_imgs = []
#     all_poses = []
#     counts = [0]
#     for s in splits:
#         meta = metas[s]
#         imgs = []
#         poses = []
#         if s=='train' or testskip==0:
#             skip = 1
#         else:
#             skip = testskip
            
#         for frame in meta['frames'][::skip]:
#             fname = os.path.join(basedir, frame['file_path'] + '.png')
#             imgs.append(imageio.imread(fname))
#             poses.append(np.array(frame['transform_matrix']))
#         imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
#         poses = np.array(poses).astype(np.float32)
#         counts.append(counts[-1] + imgs.shape[0])
#         all_imgs.append(imgs)
#         all_poses.append(poses)
    
#     i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
#     imgs = np.concatenate(all_imgs, 0)
#     poses = np.concatenate(all_poses, 0)
    
#     H, W = imgs[0].shape[:2]
#     camera_angle_x = float(meta['camera_angle_x'])
#     focal = .5 * W / np.tan(.5 * camera_angle_x)
    
#     render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
#     if half_res:
#         H = H//2
#         W = W//2
#         focal = focal/2.

#         imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
#         for i, img in enumerate(imgs):
#             imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
#         imgs = imgs_half_res
#         # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        
#     return imgs, poses, render_poses, [H, W, focal], i_split


