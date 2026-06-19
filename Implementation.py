# ================================================
# Anonymous submission for review
# ================================================
'''
Paper    : Few-shot Phase-Amplitude Aberration Correction for Phased Array Transducer in Real-time for Transcranial Focused Ultrasound
Authors  : Anonymous

Inference script for phase prediction model.

This script loads a pre-trained (fine-tuned) Phase/Amplitude model checkpoint 
and performs prediction on an unseen skull, then evaluates prediction accuracy 
against ground-truth values.

'''

import torch
import numpy as np
import time
import argparse
import os

from models.Models_archive import PhaseModel, AmpModel
from utils import find_intersections_4, scale_amp, denormalize_amp

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data', help='Path to inference_data.pt')
parser.add_argument('--result_dir', type=str, default='./results', help='Path to save results')
args = parser.parse_args()

num_bins=32
N_TD=96

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------
# Data Loading
# ----------------------------------------------------------------------------

data = torch.load(os.path.join(args.data_dir, 'inference_data.pt'))

skull     = data['skull'].to(device)                           
target    = data['target'].to(device)                          
ph_list   = data['phase'].to(device)                           
amp_list  = data['amp'].to(device)                             
td_list   = data['td'].to(device)                              

target_vxl = target * 1e3 / 0.5 + torch.tensor([100, 100, 180]).to(device)  
td_vxl     = td_list[:, 0, :, :] * 1e3 / 0.5 + torch.tensor([100, 100, 180]).to(device)  

N_SKULLS = skull.shape[0]  
N_TD     = td_vxl.shape[1] 
N_TARGET = target_vxl.shape[1] 

num_bins  = 32
bin_edges = torch.linspace(-np.pi, np.pi, num_bins + 1).to(device)
target_class = torch.bucketize(ph_list, bin_edges, right=True) - 1
target_class = torch.clamp(target_class, 0, num_bins - 1)  

amp_array = np.array(amp_list.cpu())

amp_mean = amp_array.mean()
amp_std  = amp_array.std()
print(amp_mean, amp_array.max(), amp_array.min())
scale_factor=5.0

amp_list= scale_amp(amp_list, amp_mean, amp_std, scale_factor=scale_factor)
 
PATCH_HWD    = (40, 40, 80)
PATCH_ANCHOR = (20, 20, 4)

def circular_mae(pred, gt):
    diff = torch.atan2(torch.sin(pred - gt), torch.cos(pred - gt))
    return diff.abs().mean().item()

phase_model = PhaseModel(num_fourier_freqs=16).to(device)
amp_model = AmpModel(num_fourier_freqs=16).to(device)

fine_tuning_pts = torch.tensor([83, 53, 70, 45, 44, 39, 22, 80, 10, 0], device = device)
test_pts = [i for i in range(N_TARGET) if i not in fine_tuning_pts.cpu().numpy()]


for skull_idx in range(N_SKULLS):
    skull_vol = skull[skull_idx]
    ckpt_ph  = torch.load(os.path.join(args.model_dir, f'skull{skull_idx}', 'fine_model', f'model_skull{skull_idx}.pt'))
    ckpt_amp = torch.load(os.path.join(args.model_dir, f'skull{skull_idx}', 'fine_model', f'model_skull{skull_idx}.pt'))
    skull_vol = skull[skull_idx].to(device)
    skull_vol = torch.tensor(skull_vol, device=device, dtype=torch.float32)

    if isinstance(ckpt_ph, dict) and 'model_state_dict' in ckpt_ph:
        state_dict_ph = ckpt_ph['model_state_dict']
    else:
        state_dict_ph = ckpt_ph
    phase_model.load_state_dict(state_dict_ph) 
    if isinstance(ckpt_amp, dict) and 'model_state_dict' in ckpt_amp:
        state_dict_amp = ckpt_amp['model_state_dict']
    else:
        state_dict_amp = ckpt_amp
    amp_model.load_state_dict(state_dict_amp)
    
    phase_model.to(device)
    amp_model.to(device)

    phase_model.eval()
    amp_model.eval()
    results = {"skull_idx": skull_idx, "target_idx": [], "pred_phase": [], "gt_phase": [], "inference_time":[], "pred_amp": [], "gt_amp": []}
    Cmae = 0
    Mae = 0
    Inf_time = 0
    for targ in test_pts:
        target_pos = target_vxl[skull_idx][targ]
        target_pos = target_pos.unsqueeze(0).expand(N_TD, -1) 
        td_pos     = td_vxl[skull_idx]

        with torch.no_grad():
            entry, exit_c1, exit_t, exit_p = find_intersections_4(skull_vol, td_pos, target_pos)
        entry  = torch.round(entry)
        exit_p = torch.round(exit_p)
        exit_c1 = torch.round(exit_c1)
        exit_t = torch.round(exit_t)

        B = N_TD
        patches = torch.empty((B, *PATCH_HWD), device=device)
        for b in range(B):
            center = (entry[b] + exit_p[b]) / 2
            center = torch.round(center).long()
            half = torch.tensor(PATCH_HWD, device=device).long() // 2
            start = (center - half).long()
            end = start + torch.tensor(PATCH_HWD, device=device).long()
            
            patches[b] = skull_vol[start[0]:end[0], 
                                   start[1]:end[1], 
                                   start[2]:end[2]]
        patch = patches.unsqueeze(1) 
        
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            pred_phase, logits = phase_model(target_pos, td_pos,
                                             entry, exit_c1, exit_t, exit_p,
                                             patch)
            pred_amp = amp_model(target_pos, td_pos,
                                             entry, exit_c1, exit_t, exit_p,
                                             patch)
        
        torch.cuda.synchronize()
        end_time = time.time()

        gt_phase = ph_list[skull_idx][targ]
        gt_amp   = amp_list[skull_idx][targ]
        cmae = circular_mae(pred_phase, gt_phase)

        pred_amp = denormalize_amp(pred_amp, amp_mean, amp_std, scale_factor=scale_factor)
        gt_amp = denormalize_amp(gt_amp, amp_mean, amp_std, scale_factor=scale_factor)
        mae = torch.abs(pred_amp - gt_amp).mean().item()
        inf_time = end_time-start_time
        
        results['target_idx'].append(targ)
        results['pred_phase'].append(pred_phase.detach().cpu())
        results['gt_phase'].append(gt_phase.cpu())
        results['pred_amp'].append(pred_amp.detach().cpu())
        results['gt_amp'].append(gt_amp.cpu())
        results['inference_time'].append(inf_time)
        Cmae += cmae
        Mae += mae
        Inf_time += inf_time
        del patch, patches, pred_phase, logits
        torch.cuda.empty_cache()
    print("-"*50)
    print(f"Skull {skull_idx}")
    print(f"Phase Error    : {Cmae/len(test_pts):.4f} rad")
    print(f"Amplitude Error: {Mae/len(test_pts):.4f} Pa")
    print(f"Inference Time : {Inf_time/len(test_pts):.4f} sec")
    torch.save(results, os.path.join(args.result_dir, f'final_inference_skull{skull_idx}.pt'))

print("-"*50)
