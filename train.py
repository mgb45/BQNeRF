import torch
import torch.nn as nn
from models.nerf import NeRF
import numpy as np
from tqdm import tqdm
import imageio
import torch.autograd.profiler as profiler
import random
from torch.utils.tensorboard import SummaryWriter


# random.seed(0)
# np.random.seed(0)
# torch.manual_seed(0)

# Class to generate Poses (4x4 transformations) from spherical coordinates
class CWT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def forward(self, t, phi, th):
        from math import cos,sin
        
        phi = phi/180.*np.pi
        th = th/180.*np.pi
        
        trans = torch.from_numpy(np.array([[1,0,0,0],
                [0,1,0,0],
                [0,0,1,t],
                [0,0,0,1]])).float()
        
        rot_phi = torch.from_numpy(np.array([[1,0,0,0],
                [0,cos(phi),-sin(phi),0],
                [0,sin(phi),cos(phi),0],
                [0,0,0,1]])).float()
        
        rot_theta = torch.from_numpy(np.array([[cos(th),0,-sin(th),0],
                [0,1,0,0],
                [sin(th),0,cos(th),0],
                [0,0,0,1]])).float()
        
        c2w = trans
        c2w = rot_phi @ c2w
        c2w = rot_theta @ c2w
        c2w = torch.from_numpy(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])).float() @ c2w
        
        return c2w


if __name__ == "__main__":

    #Logging path
    logdir = './logs/'
    writer = SummaryWriter(logdir)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    data = np.load('tiny_nerf_data.npz')

    images = torch.from_numpy(data['images']).to(device)
    poses = torch.from_numpy(data['poses']).to(device)
    focal = float(data['focal'])

    height, width = images.shape[1:3]
    near, far = 2., 6.

    n_training = 100
    testimg_idx = 101
    testimg, testpose = images[testimg_idx], poses[testimg_idx]

    #Use BQ?
    bq = True

    # Initialise renderer
    renderer = NeRF(d_input=3, n_layers=8, d_filter=256, skip=(4,),log_space=False,n_freqs_views=4, n_freqs=10,bq=bq,chunksize=16384)
    renderer.to(device)

    # Train loop
    lr = 1e-3
    train_iters = 1000
    batch_size = 10
    batch_idxs = np.arange(100)

    optimizer = torch.optim.AdamW(renderer.parameters(), lr=lr)
    renderer.train()
    for i in range(train_iters):

        batch_losses = []
        for j in range(len(batch_idxs)):
        
            np.random.shuffle(batch_idxs)
            idx = batch_idxs[j]
            # with profiler.profile(with_stack=True, profile_memory=True) as prof:
            img, depth_map, acc_map, weights, uncertainty = renderer.render(height,width,focal,poses[idx])

            # print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
            target_img = images[idx].reshape([-1, 3])

            # loss = torch.nn.functional.mse_loss(img, target_img)
            uncertainty = torch.nn.functional.relu(uncertainty)
            loss = torch.nn.functional.gaussian_nll_loss(img, target_img,uncertainty*torch.ones_like(target_img))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batch_losses.append(loss.item())

            print("Loss %d %d: %f"%(i,j,np.mean(batch_losses)),end="\r")
        
        renderer.eval()
        img, depth_map, acc_map, weights,uncertainty = renderer.render(height,width,focal,testpose)
        uncertainty = torch.nn.functional.relu(uncertainty)
        val_loss = torch.nn.functional.gaussian_nll_loss(img, target_img,uncertainty*torch.ones_like(target_img))
        renderer.train()

        # Logs
        writer.add_scalar("Loss/train", np.mean(batch_losses), i)
        writer.add_scalar("Loss/test", val_loss.item(), i)
        
        # Save test img
        img = (255*np.clip(img.detach().cpu().numpy(),0,1).reshape([100, 100, 3])).astype(np.uint8)
        imageio.imwrite(logdir+'Test_render_%05d.jpg'%i,img)
        imageio.imwrite(logdir+'Test_depth_%05d.jpg'%i,(depth_map.detach().cpu().numpy().reshape([100, 100, 1])).astype(np.uint8))

        if (i%10)==0:
            # Pose generator
            pose_spherical = CWT().to(device)

            # Export rendered sequence to a video 
            renderer.eval()
            frames = []
            for th in tqdm(np.linspace(0., 360., 100, endpoint=False)):
                c2w = pose_spherical(3, -30., th).to(device)
            
                img, depth_map, acc_map, weights,_ = renderer.render(height,width,focal,c2w)
                
                img = np.clip(img.detach().cpu(),0,1).reshape([100, 100, 3])
                
                frames.append((255*img.numpy()).astype(np.uint8))

            f = logdir+'video.mp4'
            imageio.mimwrite(f, frames, fps=30, quality=7)

            torch.save(renderer.state_dict(), logdir+'%05d.npy'%i)