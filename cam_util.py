import torch
import pytorch3d
from pytorch3d.transforms import random_rotation
from pytorch3d.renderer.cameras import look_at_rotation


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UP = torch.tensor([0.,0.,1.]).unsqueeze(dim=0).to(device)


def get_new_pose(center, radius):
    """
    generates a new pose / perspective for the NeRF model
    """

    rot = random_rotation().to(device)
    unit_x = torch.tensor([1.,0.,0.]).to(device)
    new_t = center + radius*(rot @ unit_x)

    new_t[2] = torch.abs(new_t[2])


    cam_rot = look_at_rotation(at=new_t.unsqueeze(dim=0), 
                               camera_position=center.unsqueeze(dim=0),up=UP)


    transform = torch.zeros((4,4)).to(device)
    
    transform[:3, :3] = cam_rot
    transform[:3, 3] = new_t
    transform[3, 3] = 1

    return transform