import os
from typing import Dict
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import h5py
from itertools import islice

from DiffusionGuidence.DiffusionCondition import GaussianDiffusionSampler, GaussianDiffusionTrainer
from DiffusionGuidence.ModelCondition import UNet
from Scheduler import GradualWarmupScheduler

from torch.cuda.amp import GradScaler as GradScaler
from torch.cuda.amp import autocast

from monai.metrics import MultiScaleSSIMMetric
from monai.metrics.regression import SSIMMetric
import pandas as pd

class SandstonesDatasetHDF5(Dataset):
    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path
        with h5py.File(self.hdf5_path, "r") as hdf5_file:
            self.keys = sorted(hdf5_file.keys(), key=lambda x: int(x.split('_')[1]))[:1400]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        with h5py.File(self.hdf5_path, "r") as hdf5_file:
            key = self.keys[index]
            data = hdf5_file[key]["data"][:]  # 3D data 
            label = hdf5_file[key]["label"][:]  # randomly selected three planes + features, a total of 12 channels
        
        data = data.astype(np.float32)
        label = label.astype(np.float32)

        return data, label
    
class EvalDatasetHDF5(Dataset):
    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path
        with h5py.File(self.hdf5_path, "r") as hdf5_file:
            self.keys = sorted(hdf5_file.keys(), key=lambda x: int(x.split('_')[1]))[1400:]


    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        with h5py.File(self.hdf5_path, "r") as hdf5_file:
            key = self.keys[index]
            data = hdf5_file[key]["data"][:]  
            label = hdf5_file[key]["label"][:] 
        
        data = data.astype(np.float32)
        label = label.astype(np.float32)

        return data, label
    

def train(modelConfig: Dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SandstonesDatasetHDF5(modelConfig["data_path"])

    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True,
        num_workers=8, drop_last=True, pin_memory=True
    )

    net_model = UNet(
        T=modelConfig["T"], ch=modelConfig["channel"],
        ch_mult=modelConfig["channel_mult"],
        num_res_blocks=modelConfig["num_res_blocks"],
        dropout=modelConfig["dropout"]
    ).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        net_model = torch.nn.DataParallel(net_model)

    if modelConfig["training_load_weight"] is not None:
        ckpt_path = os.path.join(modelConfig["save_dir"], modelConfig["training_load_weight"])
        ckpt = torch.load(ckpt_path, map_location=device)

        if any(key.startswith("module.") for key in ckpt.keys()):
            ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}

        if isinstance(net_model, torch.nn.DataParallel):
            ckpt = {"module." + k: v for k, v in ckpt.items()}

        try:
            net_model.load_state_dict(ckpt, strict=True)
            print("Model weight loaded successfully.")
        except RuntimeError as e:
            print("Error loading model weights. Please check the model structure and weights file.")
            print(e)

    model_for_optim = net_model.module if isinstance(net_model, torch.nn.DataParallel) else net_model
    decay_params = list()

    for module in model_for_optim.modules():
        if module.__class__.__name__ == "AdditionBlock":
            decay_params.extend(list(module.parameters()))
    
    all_params = list(model_for_optim.parameters())
    decay_params_set = set(decay_params)
    other_params = [p for p in all_params if p not in decay_params_set]

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "lr": modelConfig["lr"]*2},
        {"params": other_params, "lr": modelConfig["lr"]}
    ], weight_decay=1e-4)

    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1
    )

    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"],
        warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler
    )

    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]
    ).to(device)

    scaler = GradScaler()
    epoch_losses = []

    for e in range(modelConfig["epoch"]):
        epoch_loss = 0
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for i,(images,labels) in enumerate(tqdmDataLoader):
                x_0 = images.to(device).type(torch.float32) # 3D image
                labels = labels.to(device) # 3 2D images + additional 6 features for label

                if torch.isnan(x_0).any():
                    raise ValueError(f"Detected NaN in input images at epoch {e}, batch {i}.")
                if torch.isnan(labels).any():
                    raise ValueError(f"Detected NaN in input labels at epoch {e}, batch {i}.")

            # (batch_size, D, H, W) -> (batch_size, 1, D, H, W)
                x_0 = torch.unsqueeze(x_0, dim=1)
                optimizer.zero_grad()

                with autocast():
                    loss = trainer(x_0, labels).sum() / x_0.shape[0] ** 3.  # batch_size^3
                
                if torch.isnan(loss):
                    raise ValueError(f"Detected NaN in loss at epoch {e}, batch {i}.")

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(net_model.parameters(), modelConfig["grad_clip"])
                scaler.step(optimizer)
                scaler.update()

                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR (Decay Params)": optimizer.state_dict()['param_groups'][0]["lr"],  
                    "LR (Other Params)": optimizer.state_dict()['param_groups'][1]["lr"]   

                }) 
                epoch_loss += loss.item()
        warmUpScheduler.step()
            
        if e % 2 == 0 or e == modelConfig["epoch"] - 1:
            save_path = os.path.join(modelConfig["save_dir"], f'BUG_ckpt_{e}.pt')
            torch.save(net_model.state_dict(), save_path)
            print(f"Model saved at epoch {e}.")

            decay_values = []
            for module in model_for_optim.modules():
                if module.__class__.__name__ == "AdditionBlock":
                    decay_values.append((
                        module.decay_rate_d.item(),
                        module.decay_rate_h.item(),
                        module.decay_rate_w.item()))
                    
            log_file = os.path.join(modelConfig["save_dir"], "decay_params_log.txt")

            with open(log_file, "a") as f:
                f.write(f"Epoch {e}:\n")
                for idx, (decay_d, decay_h, decay_w) in enumerate(decay_values):
                    f.write(f"Block {idx} - decay_d={decay_d:.6f}, decay_h={decay_h:.6f}, decay_w={decay_w:.6f}\n")
                f.write("\n") 

        epoch_losses.append(epoch_loss / (i + 1))  
 

ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0,reduction="none")
ms_ssim_metric = MultiScaleSSIMMetric(spatial_dims=3, data_range=1.0,kernel_size=3,reduction="none")

def eval(modelConfig: Dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        model = UNet(
            T=modelConfig["T"], ch=modelConfig["channel"],
            ch_mult=modelConfig["channel_mult"],
            num_res_blocks=modelConfig["num_res_blocks"],
            dropout=modelConfig["dropout"]
        ).to(device)
 
        ckpt = torch.load(os.path.join(modelConfig["save_dir"], modelConfig["test_load_weight"]), map_location=device)
        
        if "module." in list(ckpt.keys())[0]:  
            ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}

        model.load_state_dict(ckpt)
        model.eval()

        sampler = GaussianDiffusionSampler(model, modelConfig["beta_1"], modelConfig["beta_T"],modelConfig["T"]
        ).to(device)

        dataset = EvalDatasetHDF5(modelConfig["test_data_path"])
        dataloader = DataLoader(dataset, batch_size=modelConfig["batch_size"], shuffle=False, num_workers=8)

        dir = modelConfig["output_dir"]
        start_batch = 0
        for batch_idx, (data, labels) in islice(enumerate(dataloader), start_batch, None):
            print(f"Processing batch {batch_idx}...", flush=True)
            data = data.to(device).type(torch.float32)
            labels = labels.to(device).type(torch.float32) 

            noisyImage = torch.randn(size=[modelConfig["batch_size"], 1, 100,100,100], device=device)
            sampledImgs = sampler(noisyImage, labels)

            save_samples = sampledImgs.cpu().numpy()
            np.save(dir + f"generated_batch_{batch_idx}_samples.npy", save_samples)

            data = torch.unsqueeze(data, dim=1)
            np.save(dir + f"original_batch_{batch_idx}_samples.npy", data.cpu().numpy())
            
            ssim_score = ssim_metric(sampledImgs, data).cpu().numpy()
            msssim_score = ms_ssim_metric(sampledImgs, data).cpu().numpy()

            ssim_score_flat = ssim_score.flatten()
            msssim_score_flat = msssim_score.flatten()

            ssim_ = pd.DataFrame({"SSIM": ssim_score_flat})
            msssim_ = pd.DataFrame({"MSSSIM": msssim_score_flat})

            ssim_.to_csv(os.path.join(dir, "ssim_scores.csv"), mode="a", header=False, index=False)
            msssim_.to_csv(os.path.join(dir, "msssim_scores.csv"), mode="a", header=False, index=False)





