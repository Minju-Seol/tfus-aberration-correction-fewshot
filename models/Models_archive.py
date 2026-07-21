import torch
import torch.nn as nn
import math
import numpy as np
import argparse
import os
import torch.nn.functional as F

'''
Paper    : Few-shot Phase-Amplitude Aberration Correction for Phased Array Transducer in Real-time for Transcranial Focused Ultrasound
Authors  : Minju Seol, Minjee Seo, Seonaeng Cho, Kyungho Yoon
Venue    : MICCAI 2026 DT4H Workshop (Accepted)

Phase and Amplitude Prediction models.

'''

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data', help='Path to inference_data.pt')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------
# Data Loading
# ----------------------------------------------------------------------------
data = torch.load(os.path.join(args.data_dir, 'inference_data.pt'))

skull     = data['skull'].to(device)
target    = data['target'].to(device)
amp_list   = data['amp'].to(device)  
td_list   = data['td'].to(device)    
ph_list   = data['phase'].to(device)

print(f"skull: {skull.shape}, target: {target.shape}, amp_list: {amp_list.shape}, td_list: {td_list.shape}")

target_vxl = target * 1e3 / 0.5 + torch.tensor([100, 100, 180]).to(device)  
td_vxl     = td_list[:, 0, :, :] * 1e3 / 0.5 + torch.tensor([100, 100, 180]).to(device)

N_SKULLS = skull.shape[0]   
N_TD     = td_vxl.shape[1]  
N_TARGET = target_vxl.shape[1]

PATCH_HWD    = (40, 40, 80)  
PATCH_ANCHOR = (20, 20, 4)   

amp_array = np.array(amp_list.cpu())  

amp_mean = amp_array.mean()
amp_std  = amp_array.std()

scale_factor=5.0
def scale_amp(x, scale_factor=5.0):
    nor_x = (x - amp_mean) / amp_std
    scale_x = nor_x * scale_factor  
    return scale_x

def denormalize_amp(x, scale_factor=5.0):
    unscale_x = x / scale_factor
    x = unscale_x * amp_std + amp_mean
    return x
amp_list= scale_amp(amp_list, scale_factor=scale_factor)

print("=" * 60)
print(f"Amplitude Scaled: Scale factor = {scale_factor}, mean={amp_list.mean():.4f}, std={amp_list.std():.4f}")


num_bins  = 32
bin_edges = torch.linspace(-np.pi, np.pi, num_bins + 1).to(device)
target_class = torch.bucketize(ph_list, bin_edges, right=True) - 1
target_class = torch.clamp(target_class, 0, num_bins - 1)

PATCH_HWD    = (40, 40, 80)  
PATCH_ANCHOR = (20, 20, 4)   

class FourierFeatures(nn.Module):
    def __init__(self, in_dim, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.freq_bands = 2 ** torch.arange(0, num_frequencies).float()

    def forward(self, x):
        x = x.unsqueeze(-1) * self.freq_bands.to(x.device) * math.pi * 2
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1).view(x.shape[0], -1)


class PhaseClassifier(nn.Module):
    def __init__(self, dim, num_bins=num_bins, p=0.3):
        super().__init__()
        self.in_ln = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 256), nn.GELU(), nn.Dropout(p),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(p),
            nn.Linear(128, 64), nn.GELU(),
        )
        self.out = nn.Linear(64, num_bins)

        k       = torch.arange(num_bins).float()
        centers = -math.pi + (2*math.pi)*(k + 0.5)/num_bins
        self.register_buffer('sin_c', centers.sin())
        self.register_buffer('cos_c', centers.cos())

    def forward(self, x, temperature: float = 1.0):
        h      = self.mlp(self.in_ln(x))
        logits = self.out(h)
        probs  = F.softmax(logits / temperature, dim=-1)
        s = torch.sum(probs * self.sin_c, dim=-1, keepdim=True)
        c = torch.sum(probs * self.cos_c, dim=-1, keepdim=True)
        return torch.atan2(s, c), logits

class Skull3dCNN(nn.Module):
    def __init__(self, out_dim=16, pool_size=(2, 2, 4)):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1,  8,  kernel_size=3, padding=1), nn.GELU(), nn.MaxPool3d(2),
            nn.Conv3d(8,  16, kernel_size=3, padding=1), nn.GELU(), nn.MaxPool3d(2),
            nn.Conv3d(16, 8, kernel_size=5, padding=2), nn.GELU(), nn.MaxPool3d(2),
        )
        self.pool = nn.AdaptiveAvgPool3d(pool_size)
        flat_dim  = 8 * pool_size[0] * pool_size[1] * pool_size[2]  
        self.fc   = nn.Linear(flat_dim, out_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x)          
        x = x.view(x.size(0), -1) 
        return self.fc(x)         

class PhaseModel(nn.Module):
    def __init__(self, num_fourier_freqs=16, num_bins=num_bins, n_td=N_TD):
        super().__init__()
        self.ff         = FourierFeatures(6, num_frequencies=num_fourier_freqs)
        self.phase_head = PhaseClassifier(262, num_bins=num_bins)
        self.skullcnn   = Skull3dCNN(out_dim=64)
        self.register_buffer(
            "coord_offset",
            torch.tensor([0.132, 0.132, 0.092], dtype=torch.float32)
        )
        self.norm_geo   = nn.LayerNorm(192)
        self.norm_dist  = nn.LayerNorm(5)
        self.norm_tof   = nn.LayerNorm(1)
        self.norm_skull = nn.LayerNorm(64)

    def encode_skull_in_chunks(self, skull_patch, chunk_size=32):
        feats = []
        for i in range(0, skull_patch.size(0), chunk_size):
            feats.append(self.skullcnn(skull_patch[i:i + chunk_size]))
        return torch.cat(feats, dim=0)

    def forward(self, target_xyz, trans_xyz, out_inter, exit_c1, exit_t, in_inter, skull_patch):
        offset = self.coord_offset.to(target_xyz.device)

        td_mm        = trans_xyz  * 0.001 - offset
        tgt_mm       = target_xyz * 0.001 - offset

        ff_input  = torch.cat([td_mm, tgt_mm], dim=-1) 

        out_inter = out_inter.float()
        in_inter  = in_inter.float()

        dist_out  = torch.norm(out_inter - trans_xyz,  dim=-1, keepdim=True) * 0.001
        dist_c1   = torch.norm(out_inter - exit_c1,   dim=-1, keepdim=True) * 0.001
        dist_t    = torch.norm(exit_c1 - exit_t,  dim=-1, keepdim=True) * 0.001
        dist_c2   = torch.norm(exit_t - in_inter,   dim=-1, keepdim=True) * 0.001
        dist_in   = torch.norm(in_inter  - target_xyz, dim=-1, keepdim=True) * 0.001

        tof = (dist_out / 1500) + ((dist_c1 + dist_c2) / 2384) + (dist_t / 2140) + (dist_in / 1500)
        tof = tof*1000

        enc_geo   = self.ff(ff_input)

        dist_feat = torch.cat([dist_out, dist_c1, dist_t, dist_c2, dist_in], dim=-1)
        skull_3d  = self.encode_skull_in_chunks(skull_patch)
        enc_geo   = self.norm_geo(enc_geo)
        dist_feat = self.norm_dist(dist_feat)
        tof       = (tof - tof.mean()) / (tof.std() + 1e-8)
        # print(tof)
        skull_3d  = self.norm_skull(skull_3d)

        geo_feat = torch.cat([enc_geo, dist_feat, tof, skull_3d], dim=-1)
        pred_phase, logits = self.phase_head(geo_feat)

        pred_phase = torch.atan2(torch.sin(pred_phase), torch.cos(pred_phase))
        return pred_phase, logits

class AmpRegressor(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.in_ln = nn.LayerNorm(in_dim)
        self.model = nn.Sequential(
            nn.Linear(in_dim, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(64, 16), nn.GELU(),
        )
        self.out = nn.Linear(16, 1)
    def forward(self, x):
        x = self.in_ln(x)
        x = self.model(x)
        return self.out(x) 

class AmpModel(nn.Module):
    def __init__(self, num_fourier_freqs=16, n_td=N_TD):
        super().__init__()
        self.ff         = FourierFeatures(6, num_frequencies=num_fourier_freqs)
        self.amp_head = AmpRegressor(262)
        self.skullcnn   = Skull3dCNN(out_dim=64)
        self.register_buffer(
            "coord_offset",
            torch.tensor([0.132, 0.132, 0.092], dtype=torch.float32)
        )
        self.norm_geo   = nn.LayerNorm(192)
        self.norm_dist  = nn.LayerNorm(5)
        self.norm_tof   = nn.LayerNorm(1)
        self.norm_skull = nn.LayerNorm(64)

    def forward(self, target_xyz, trans_xyz, out_inter, exit_c1, exit_t, in_inter, skull_patch):
        offset = self.coord_offset.to(target_xyz.device)

        td_mm        = trans_xyz  * 0.001 - offset
        tgt_mm       = target_xyz * 0.001 - offset

        ff_input  = torch.cat([td_mm, tgt_mm], dim=-1) 

        out_inter = out_inter.float()
        in_inter  = in_inter.float()

        dist_out  = torch.norm(out_inter - trans_xyz,  dim=-1, keepdim=True) * 0.001
        dist_c1   = torch.norm(out_inter - exit_c1,   dim=-1, keepdim=True) * 0.001
        dist_t    = torch.norm(exit_c1 - exit_t,  dim=-1, keepdim=True) * 0.001
        dist_c2   = torch.norm(exit_t - in_inter,   dim=-1, keepdim=True) * 0.001
        dist_in   = torch.norm(in_inter  - target_xyz, dim=-1, keepdim=True) * 0.001

        tof = (dist_out / 1500) + ((dist_c1+dist_c2) / 2384) + (dist_t / 2140) + (dist_in / 1500)
        tof = tof*1000
        enc_geo   = self.ff(ff_input)        
        dist_feat = torch.cat([dist_out, dist_c1, dist_t, dist_c2, dist_in], dim=-1) 
        skull_3d  = self.skullcnn(skull_patch)                         
        enc_geo   = self.norm_geo(enc_geo)
        dist_feat = self.norm_dist(dist_feat)
        tof       = (tof - tof.mean()) / (tof.std() + 1e-8)
        skull_3d  = self.norm_skull(skull_3d)

        geo_feat = torch.cat([enc_geo, dist_feat, tof, skull_3d], dim=-1)  
        pred_amp = self.amp_head(geo_feat)

        return pred_amp
