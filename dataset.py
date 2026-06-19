# ================================================
# Anonymous submission for review
# ================================================

'''
Paper    : Few-shot Phase-Amplitude Aberration Correction for Phased Array Transducer in Real-time for Transcranial Focused Ultrasound
Authors  : Anonymous

Dataset classes for amplitude and phase prediction models.
Builds (skull, target, transducer-element) index triplets and 
returns the corresponding geometry and ground-truth labels for 
each sample.

Classes:
    - AmpDataset   : returns ground-truth amplitude per sample
    - PhaseDataset : returns ground-truth phase + soft class label per sample

Input (per dataset):
    - skull volumes                                       (skull)
    - list of skull indices to include                    (nbr_skull_list)
    - target positions (voxel space)                      (target_vxl)
    - transducer element positions (voxel space)          (td_vxl)
    - ground-truth amplitude / phase values               (amp_list / ph_list)
    - soft class labels (PhaseDataset only)               (target_class)
    - transducer element index list                       (td_idx_list)
    - target index list                                   (target_idx_list)

Output (per __getitem__ call):
    - AmpDataset   : target, td, gt_amp, skull_idx, td_idx, target_idx
    - PhaseDataset : target, td, gt_phase, soft_label, skull_idx, td_idx, target_idx

'''

import torch

# ----------------------------------------------------------------------------
# Amplitude Dataset
# ----------------------------------------------------------------------------
class AmpDataset(torch.utils.data.Dataset):
    def __init__(self, skull, nbr_skull_list, target_vxl, td_vxl,
                 amp_list, td_idx_list, target_idx_list):
        self.index_list = []
        for skull_idx in nbr_skull_list:
            for target_idx in target_idx_list:
                for td_idx in td_idx_list:
                    self.index_list.append((skull_idx, target_idx, td_idx))
        self.skull        = skull
        self.target_vxl   = target_vxl
        self.td_vxl       = td_vxl
        self.amp_list      = amp_list

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        skull_idx, target_idx, td_idx = self.index_list[idx]
        target     = self.target_vxl[skull_idx, target_idx]
        td         = self.td_vxl[skull_idx, td_idx]        
        gt_amp   = self.amp_list[skull_idx, target_idx, td_idx]
        return target, td, gt_amp, skull_idx, td_idx, target_idx

# ----------------------------------------------------------------------------
# Phase Dataset
# ----------------------------------------------------------------------------
class PhaseDataset(torch.utils.data.Dataset):
    def __init__(self, skull, nbr_skull_list, target_vxl, td_vxl,
                 ph_list, target_class, td_idx_list, target_idx_list):
        self.index_list = [] 
        for skull_idx in nbr_skull_list:
            for target_idx in target_idx_list:
                for td_idx in td_idx_list:
                    self.index_list.append((skull_idx, target_idx, td_idx))
        self.skull        = skull
        self.target_vxl   = target_vxl
        self.td_vxl       = td_vxl
        self.ph_list      = ph_list
        self.target_class = target_class

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        skull_idx, target_idx, td_idx = self.index_list[idx]
        target     = self.target_vxl[skull_idx, target_idx]            # [3]
        td         = self.td_vxl[skull_idx, td_idx]                    # [3]
        gt_phase   = self.ph_list[skull_idx, target_idx, td_idx]       # [1]
        soft_label = self.target_class[skull_idx, target_idx, td_idx]  # [1]
        return target, td, gt_phase, soft_label, skull_idx, td_idx, target_idx
