import torch
import torch.nn.functional as F

'''
Paper    : Few-shot Phase-Amplitude Aberration Correction for Phased Array Transducer in Real-time for Transcranial Focused Ultrasound
Authors  : Minju Seol, Minjee Seo, Seonaeng Cho, Kyungho Yoon
Venue    : MICCAI 2026 DT4H Workshop (Accepted)

Utility functions for geometry computation and label/amplitude preprocessing.

Functions:
    - find_intersections_4        
        : computes ray-skull intersection points (water-bone-trabecular-bone-tissue boundaries)
        
    - make_circular_soft_label    
        : generates smoothed circular soft labels for phase classification
        
    - scale_amp / denormalize_amp 
        : normalize / denormalize amplitude values
'''

@torch.no_grad()
def find_intersections_4(vol, td, so):

    dtype = torch.float32
    dev   = vol.device
    vol_c = vol.to(dtype).unsqueeze(0).unsqueeze(0)    
    _, _, Dv, Hv, Wv = vol_c.shape
    td = td.to(dev, dtype=dtype)   
    so = so.to(dev, dtype=dtype)   

    v = so - td
    u = F.normalize(v, dim=1)               
    def sample_single_vol(pts):
        B_, N_, _ = pts.shape
        g = to_grid(pts).reshape(1, B_ * N_, 1, 1, 3)
        hu = F.grid_sample(vol_c, g, mode='bilinear', align_corners=True)
        return hu.view(1, 1, B_, N_, 1, 1)[0, 0, :, :, 0, 0]  
    def to_grid(pts):
        xg = (2. * pts[..., 2] / (Wv - 1.) - 1.).clamp(-1., 1.)
        yg = (2. * pts[..., 1] / (Hv - 1.) - 1.).clamp(-1., 1.)
        zg = (2. * pts[..., 0] / (Dv - 1.) - 1.).clamp(-1., 1.)
        return torch.stack([xg, yg, zg], dim=-1)

    N_CHK  = 64
    lens   = torch.norm(v, dim=1, keepdim=True)                      
    t_chk  = torch.linspace(0, 1, N_CHK, device=dev).view(1, N_CHK)  
    pts_c  = td.unsqueeze(1) + (t_chk * lens).unsqueeze(2) * u.unsqueeze(1)  

    hu_chk = sample_single_vol(pts_c)  
    flip   = hu_chk.amax(dim=1) < 0.05                              
    u      = torch.where(flip.unsqueeze(1), -u, u)

    N_MCH  = 256
    max_len = lens.amax().item()
    ts_m   = torch.linspace(0, max_len, N_MCH, device=dev)          
    pts_m  = td.unsqueeze(1) + ts_m[None, :, None] * u.unsqueeze(1) 

    hu_m = sample_single_vol(pts_m)
    th     = (hu_m.mean(dim=1, keepdim=True)
              + 0.2 * hu_m.std(dim=1, keepdim=True)).clamp(min=0.05)
    bone   = hu_m > th                                            
    has_b  = bone.any(dim=1)                                      

    ei     = bone.int().argmax(dim=1)                             
    xi     = (N_MCH - 1) - bone.flip(dims=[1]).int().argmax(dim=1)
    entry  = td + u * ts_m[ei].unsqueeze(1) 
    exit_p = td + u * ts_m[xi].unsqueeze(1) 
    entry  = torch.where(has_b.unsqueeze(1), entry,  td)
    exit_p = torch.where(has_b.unsqueeze(1), exit_p, so)

    HU_CORTICAL   = 1000.0
    HU_TRABECULAR = 0.0

    t_ei = ts_m[ei]  
    t_xi = ts_m[xi]  
    in_skull = (ts_m[None, :] >= t_ei[:, None]) & \
            (ts_m[None, :] <= t_xi[:, None])   

    cortical_mask   = (hu_m > HU_CORTICAL) & in_skull
    trabecular_mask = (hu_m > HU_TRABECULAR) & \
                    (hu_m <= HU_CORTICAL) & in_skull

    c1_exit_i = (N_MCH - 1) - cortical_mask[:, :N_MCH//2].flip(dims=[1]).int().argmax(dim=1)

    t_exit_i  = (N_MCH - 1) - trabecular_mask.flip(dims=[1]).int().argmax(dim=1)

    c1_exit = td + u * ts_m[c1_exit_i].unsqueeze(1)
    t_exit  = td + u * ts_m[t_exit_i].unsqueeze(1) 

    has_c = cortical_mask.any(dim=1)
    has_t = trabecular_mask.any(dim=1)
    c1_exit = torch.where(has_c.unsqueeze(1), c1_exit, entry)
    t_exit  = torch.where(has_t.unsqueeze(1), t_exit,  exit_p)

    return entry, c1_exit, t_exit, exit_p

def make_circular_soft_label(class_index, num_bins=314, smoothing_radius=1, sigma=0.5):
    B = class_index.size(0)
    device = class_index.device
    indices = torch.arange(num_bins, device=device).unsqueeze(0).expand(B, -1)  # [B, num_bins]
    class_index = class_index.view(-1).float()
    dist = (indices - class_index.unsqueeze(1)) % num_bins
    dist = torch.minimum(dist, num_bins - dist)
    mask = dist <= smoothing_radius
    weights = torch.exp(- (dist ** 2) / (2 * sigma ** 2)) * mask
    weights = weights / weights.sum(dim=1, keepdim=True)

    return weights

def scale_amp(x, amp_mean, amp_std, scale_factor=5.0):
    nor_x = (x - amp_mean) / amp_std
    scale_x = nor_x * scale_factor
    return scale_x

def denormalize_amp(x, amp_mean, amp_std, scale_factor=5.0):
    unscale_x = x / scale_factor
    x = unscale_x * amp_std + amp_mean
    return x
