# ================================================
# Anonymous submission for review
# ================================================

'''
Paper    : Few-shot Phase-Amplitude Aberration Correction for Phased Array Transducer in Real-time for Transcranial Focused Ultrasound
Authors  : Anonymous

Training script for phase prediction model.
Few-shot fine-tuning script on the held-out test skull is also included.

This script trains a neural network to predict acoustic **amplitude** 
from skull CT patches and transducer/target geometry 
using leave-one-out (LOO) cross-validation.

Input:
    - target position                                   (target_b)
    - transducer element position                       (td_b)
    - intersections (water -> cortical bone             (entry)
                     cortical bone -> trabecular bone,  (exit_c1)
                     trabecular bone -> cortical bone,  (exit_t)
                     cortical bone -> soft tissue       (eixt_p)
    - skull patch                                       (patch)
Output:
    - predicted amplitude values                        (preds)

'''

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import time
import gc
from datetime import datetime
import argparse
from utils import find_intersections_4, scale_amp, denormalize_amp
from models.Models_archive import AmpModel
from dataset import AmpDataset

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data', help='Path to inference_data.pt')
parser.add_argument('--result_dir', type=str, default='./results', help='Path to save results')
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
amp_list= scale_amp(amp_list, amp_mean, amp_std, scale_factor=scale_factor)

print("=" * 60)
print(f"Amplitude Scaled: Scale factor = {scale_factor}, mean={amp_list.mean():.4f}, std={amp_list.std():.4f}")


# ----------------------------------------------------------------------------
# LOO training
# ----------------------------------------------------------------------------
phase_loss_fn = nn.KLDivLoss(reduction='batchmean')
epochs        = 30
loo_results   = []

for test_skull_idx in range(N_SKULLS):
    fold_start   = time.time()
    train_skulls = [i for i in range(N_SKULLS) if i != test_skull_idx]

    # result_dir    = f'./[3]/Results/amplitude models/skull{test_skull_idx}'
    result_dir = os.path.join(args.result_dir, f'skull{test_skull_idx}')
    base_model_dir = os.path.join(result_dir, 'base_model')
    fine_model_dir = os.path.join(result_dir, 'fine_model')
    os.makedirs(base_model_dir, exist_ok=True)
    os.makedirs(fine_model_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"LOO Fold {test_skull_idx+1}/{N_SKULLS}  "
          f"| train skulls: {train_skulls}  | test skull: {test_skull_idx}")
    print(f"{'='*60}")

    train_dataset = AmpDataset(
        skull           = skull,
        nbr_skull_list  = train_skulls,
        target_vxl      = target_vxl,
        td_vxl          = td_vxl,
        amp_list         = amp_list,
        td_idx_list     = list(range(N_TD)),   
        target_idx_list = list(range(N_TARGET)),
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              pin_memory=False, drop_last=False)
    print(f"Train samples: {len(train_dataset)} | "
          f"batches/epoch: {len(train_loader)}")

    model     = AmpModel(num_fourier_freqs=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    for epoch in range(epochs):
        model.train()
        total_loss = total_mae = 0.0

        for target_b, td_b, gt_amp, skull_idx_b, td_idx_b, _ in train_loader:
            target_b   = target_b.to(device)
            td_b       = td_b.to(device)
            gt_amp     = gt_amp.to(device)
            td_idx_b   = td_idx_b.to(device)
            skull_idx_b = skull_idx_b.to(device)
            skull_b    = skull[skull_idx_b]
            with torch.no_grad():
                entry, exit_c1, exit_t, exit_p = find_intersections_4(skull_b, td_b, target_b)
            entry  = torch.round(entry)
            exit_p = torch.round(exit_p)
            exit_c1 = torch.round(exit_c1)
            exit_t = torch.round(exit_t)

            B = skull_idx_b.shape[0]
            patches = torch.empty((B, *PATCH_HWD), device=device)
            for b in range(B):
                s_idx = skull_idx_b[b].item()
                skull_vol_b = skull[s_idx] 
                
                center = (entry[b] + exit_p[b]) / 2
                center = torch.round(center).long()
                half = torch.tensor(PATCH_HWD, device=device).long() // 2
                start = (center - half).long()
                end = start + torch.tensor(PATCH_HWD, device=device).long()
                
                patches[b] = skull_vol_b[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
            patch = patches.unsqueeze(1) 

            optimizer.zero_grad()
            preds = model(target_b, td_b, entry, exit_c1, exit_t, exit_p, patch)

            diff       = torch.abs(preds - gt_amp)
            mse        = F.mse_loss(preds, gt_amp)
            huber      = F.huber_loss(preds, gt_amp)
            loss = huber
            mae        = torch.abs(denormalize_amp(preds, amp_mean, amp_std, scale_factor=5.0) - denormalize_amp(gt_amp, amp_mean, amp_std, scale_factor=5.0))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss  += loss.item()
            total_mae   += mae.mean().item()

        scheduler.step()

        n = len(train_loader)
        print(f"  [{epoch+1:3d}/{epochs}] "
              f"loss {total_loss/n:.4f} | "
              f"MAE {total_mae/n:.4f}")
        
    torch.save(model.state_dict(), os.path.join(base_model_dir, f'model_skull{test_skull_idx}.pt'))

    np.random.seed(42)
    n_finetuning_pts=10
    fine_target_idx = np.random.choice(np.arange(N_TARGET),size=n_finetuning_pts, replace=False)
    test_target_idx = [i for i in range(N_TARGET) if i not in fine_target_idx]
    
    fine_dataset = AmpDataset(
        skull           = skull,
        nbr_skull_list  = [test_skull_idx],
        target_vxl      = target_vxl,
        td_vxl          = td_vxl,
        amp_list         = amp_list,
        td_idx_list     = list(range(N_TD)),
        target_idx_list = fine_target_idx,
    )
    fine_loader = DataLoader(fine_dataset, batch_size=32, shuffle=True)

    for p in model.skullcnn.parameters():
        p.requires_grad = False
    fine_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=1e-4
    )
    
    fine_epochs = 40
    start_fine = time.time()
    print(f"\n{'='*60}")
    print(f"Fine-tuning ({n_finetuning_pts} targets)")        
    print(f"{'='*60}\n")
    for epoch in range(fine_epochs):
        model.train()
        total_loss = total_mae = total_kl = total_circ = 0.0

        for target_b, td_b, gt_amp, skull_idx_b, td_idx_b, _ in fine_loader:
            target_b   = target_b.to(device)
            td_b       = td_b.to(device)
            gt_amp     = gt_amp.to(device)
            td_idx_b   = td_idx_b.to(device)
            skull_idx_b = skull_idx_b.to(device)
            skull_b    = skull[skull_idx_b]

            with torch.no_grad():
                entry, exit_c1, exit_t, exit_p = find_intersections_4(skull_b, td_b, target_b)
            entry  = torch.round(entry)
            exit_p = torch.round(exit_p)
            exit_c1 = torch.round(exit_c1)
            exit_t = torch.round(exit_t)

            B = skull_idx_b.shape[0]
            patches = torch.empty((B, *PATCH_HWD), device=device)
            for b in range(B):
                s_idx = skull_idx_b[b].item()
                skull_vol_b = skull[s_idx] 
                
                center = (entry[b] + exit_p[b]) / 2
                center = torch.round(center).long()
                half = torch.tensor(PATCH_HWD, device=device).long() // 2
                start = (center - half).long()
                end = start + torch.tensor(PATCH_HWD, device=device).long()
                
                patches[b] = skull_vol_b[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
            patch = patches.unsqueeze(1) 
            fine_optimizer.zero_grad()
            preds = model(target_b, td_b, entry, exit_c1, exit_t, exit_p, patch)


            diff       = torch.abs(preds - gt_amp)
            huber      = F.huber_loss(preds, gt_amp)
            mae        = torch.abs(denormalize_amp(preds, amp_mean, amp_std, scale_factor=5.0) - denormalize_amp(gt_amp, amp_mean, amp_std, scale_factor=5.0))
            loss = huber
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            fine_optimizer.step()

            total_loss  += loss.item()
            total_mae   += mae.mean().item()

        n = len(fine_loader)
        print(f"  [{epoch+1:3d}/{fine_epochs}] "
              f"loss {total_loss/n:.4f} | "
              f"MAE {total_mae/n:.4f} Pa")
    print(f"Finetuning time: {time.time() - start_fine:.4f} s")
    torch.save(model.state_dict(), os.path.join(fine_model_dir, f'model_skull{test_skull_idx}.pt'))

    model.eval()
    test_dataset = AmpDataset(
        skull           = skull,
        nbr_skull_list  = [test_skull_idx],
        target_vxl      = target_vxl,
        td_vxl          = td_vxl,
        amp_list         = amp_list,
        td_idx_list     = list(range(N_TD)),
        target_idx_list = test_target_idx,
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             pin_memory=False)

    all_mae     = []
    all_huber   = []
    all_records = []
    total_time    = 0.0
    total_samples = 0
    with torch.no_grad():
        for target_b, td_b, gt_amp, skull_idx_b, td_idx_b, target_idx_b in test_loader:
            target_b    = target_b.to(device)
            td_b        = td_b.to(device)
            gt_amp      = gt_amp.to(device)
            td_idx_b    = td_idx_b.to(device)
            skull_idx_b = skull_idx_b.to(device)
            skull_b     = skull[skull_idx_b] 

            entry, exit_c1, exit_t, exit_p = find_intersections_4(
                skull_b, td_b, target_b
            )
            entry   = torch.round(entry)
            exit_c1 = torch.round(exit_c1)
            exit_t  = torch.round(exit_t)
            exit_p  = torch.round(exit_p)

            B = skull_idx_b.shape[0]
            patches = torch.empty((B, *PATCH_HWD), device=device)
            for b in range(B):
                s_idx       = skull_idx_b[b].item()
                skull_vol_b = skull[s_idx]
                center = torch.round((entry[b] + exit_p[b]) / 2).long()
                half   = torch.tensor(PATCH_HWD, device=device).long() // 2
                st     = (center - half).long()
                en     = st + torch.tensor(PATCH_HWD, device=device).long()
                patches[b] = skull_vol_b[st[0]:en[0], st[1]:en[1], st[2]:en[2]]

            patch = patches.unsqueeze(1) 

            wall_clock = datetime.now()
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            preds = model(target_b, td_b, entry, exit_c1, exit_t, exit_p, patch)

            torch.cuda.synchronize()
            t1 = time.perf_counter()

            batch_inf_time      = t1 - t0
            per_sample_inf_time = batch_inf_time / B
            total_time    += batch_inf_time
            total_samples += B

            diff       = torch.abs(preds - gt_amp)
            mse        = F.mse_loss(preds, gt_amp)
            huber      = F.huber_loss(preds, gt_amp)
            loss = huber
            mae        = torch.abs(denormalize_amp(preds, amp_mean, amp_std, scale_factor=5.0) - denormalize_amp(gt_amp, amp_mean, amp_std, scale_factor=5.0))
            preds_cpu    = denormalize_amp(preds).cpu()
            gt_amp_cpu   = denormalize_amp(gt_amp).cpu()
            target_cpu   = target_b.cpu()
            for i in range(B):
                all_records.append({
                    'skull_idx':  skull_idx_b[i].item(),
                    'target_idx': target_idx_b[i].item(),
                    'target':     target_cpu[i], 
                    'pred_amp': preds_cpu[i].item(),
                    'gt_amp':   gt_amp_cpu[i].item(),
                    'inf_time':   per_sample_inf_time,
                    'wall_clock': wall_clock,
                })
            all_mae.append(mae)
            all_huber.append(huber)
    avg_time_per_sample = total_time / total_samples
    print(f"\n  >> Average inference time per sample: {avg_time_per_sample:.6f} s")
    mae_pa = torch.stack(all_mae).mean().item()
    huber_pa = torch.stack(all_huber).mean().item()
    torch.save(all_records, os.path.join(result_dir, f'val_records_skull{test_skull_idx}.pt'))
    loo_results.append({'test_skull': test_skull_idx, 'mae_pa': all_mae, 'huber_pa': all_huber, 'records': all_records})
    print(f"\n  >> Test skull {test_skull_idx}: Amplitude MAE = {mae_pa:.4f} , Huber Loss = {huber_pa:.4f} Pa  | "
          f"fold time {time.time()-fold_start:.0f}s")


    del train_loader, test_loader, train_dataset, test_dataset, model, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()

print(f"\n{'='*60}")
print("LOO CV Summary")
print(f"{'='*60}")
for r in loo_results:
    print(f"  Skull {r['test_skull']:2d}: {r['mae_pa']:.4f} Pa  ")
mean_mae = np.mean([r['mae_pa'] for r in loo_results])
print(f"\n  Mean Amplitude MAE: {mean_mae:.4f} Pa  ")
