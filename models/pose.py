import torch
import torch.nn as nn
import numpy as np

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