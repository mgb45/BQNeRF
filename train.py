import torch
import torch.nn as nn
from models.nerf import NeRF
from models.pose import CWT
import numpy as np
from tqdm import tqdm
import imageio
import torch.autograd.profiler as profiler
import random
from torch.utils.tensorboard import SummaryWriter
import argparse
import uuid
import csv

parser = argparse.ArgumentParser(description='Train a basic NeRF')
parser.add_argument('--bq',type=str,default='BQ',help="Quadrature method (BQ/Std)")
parser.add_argument('--epochs',type=int,default=5000,help="Number of epochs to train for")
parser.add_argument('--lr',type=float,default=5e-4,help="Learning rate")
parser.add_argument('--logdir',type=str,default='./logs/',help="Log directory")
parser.add_argument('--chunksize',type=int,default=16384,help="Chunks to render at a time (memory issues)")
parser.add_argument('--nsamples',type=int,default=64,help="Number of samples along ray")
parser.add_argument('--video_log_freq',type=int,default=10,help="Frequency to log test render video")
parser.add_argument('--frame_log_freq',type=int,default=10,help="Frequency to log test render frame")
parser.add_argument('--model_save_freq',type=int,default=10,help="Frequency to save model")
parser.add_argument('--seed',type=int,default=42,help="Random seed")

args = parser.parse_args()

if __name__ == "__main__":

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    #Logging path

    proc_id = str(uuid.uuid4())

    logdir = args.logdir + proc_id + '/'
    writer = SummaryWriter(logdir)

    print(args)

    f = open(logdir+'parameters.csv',"w")
    csv_writer = csv.writer(f)
    for name,val in dict(vars(args)).items():
        csv_writer.writerow([name,val])
    f.close()


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

    bq = False
    if args.bq == 'BQ':
        bq = True

    # Initialise renderer
    renderer = NeRF(d_input=3, n_layers=8, d_filter=256, skip=(4,),log_space=False,n_freqs_views=4, n_freqs=10,bq=bq,chunksize=args.chunksize,nsamples=args.nsamples)
    renderer.to(device)

    # Train loop
    lr = args.lr
    train_iters = 10000
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
            loss = torch.nn.functional.gaussian_nll_loss(img, target_img,uncertainty*torch.ones_like(target_img))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batch_losses.append(loss.item())

        print("Loss %d: %f"%(i,np.mean(batch_losses)),end="\r")
        
        renderer.eval()
        img, depth_map, acc_map, weights,uncertainty = renderer.render(height,width,focal,testpose)
        val_loss = torch.nn.functional.gaussian_nll_loss(img, target_img,uncertainty*torch.ones_like(target_img))
        renderer.train()

        # Logs
        writer.add_scalar("Loss/train", np.mean(batch_losses), i)
        writer.add_scalar("Loss/test", val_loss.item(), i)
        
        # Save test img
        if (i %args.frame_log_freq)==0:
            img = (255*np.clip(img.detach().cpu().numpy(),0,1).reshape([100, 100, 3])).astype(np.uint8)
            imageio.imwrite(logdir+'Test_render_%05d.jpg'%i,img)
            imageio.imwrite(logdir+'Test_depth_%05d.jpg'%i,(((depth_map.detach().cpu().numpy()-renderer.near)/(renderer.far-renderer.near)).reshape([100, 100, 1])).astype(np.uint8))

        if (i%args.video_log_freq)==0:
            # Pose generator
            pose_spherical = CWT().to(device)

            # Export rendered sequence to a video 
            renderer.eval()
            frames = []
            for th in tqdm(np.linspace(0., 360., 100, endpoint=False)):
                c2w = pose_spherical(4, -30., th).to(device)
            
                img, depth_map, acc_map, weights,_ = renderer.render(height,width,focal,c2w)
                
                img = np.clip(img.detach().cpu(),0,1).reshape([100, 100, 3])
                
                frames.append((255*img.numpy()).astype(np.uint8))

            f = logdir+'video.mp4'
            imageio.mimwrite(f, frames, fps=30, quality=7)
            renderer.train()

        if (i%args.model_save_freq)==0:
            torch.save(renderer.state_dict(), logdir+'%05d.npy'%i)