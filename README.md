# AB‑CDM: Attenuation‑Based Conditional Diffusion Model

AB‑CDM is a conditional denoising diffusion probabilistic model (DDPM) for **stochastic reconstruction of 3D porous media** from limited 2D surface observations. It integrates **multi‑view conditional embeddings**, **axis‑specific exponential attenuation**, and a **3D U‑Net** backbone to generate high‑fidelity internal pore structures.

![Workflow](./Images/Workflow.jpg)

---

## Repository overview

This repository provides a PyTorch implementation of AB‑CDM. The main components are:

- **Train.py** – training entry point (defines `modelConfig` and calls `train()`).
- **Eval.py** – inference entry point (loads a checkpoint and calls `eval()`).
- **TrainCondition.py** – dataset loading, training loop, checkpointing, and evaluation metrics (SSIM / MS‑SSIM).
- **ModelCondition.py** – AB‑CDM network (time embedding, conditional embedding, attenuation‑based guidance, 3D U‑Net).
- **DiffusionCondition.py** – Gaussian diffusion trainer and sampler (forward noising + reverse denoising).
- **Scheduler.py** – learning‑rate schedule (warmup + cosine annealing).

---

## Data preparation

### 1) Core splitting and downsampling (`Preprocess.ipynb`)

**Input.** Raw rock cores are stored as `.raw` files with a resolution of **1000³ voxels**.

**Patch extraction.** Each 1000³ core is split into overlapping **500³** sub‑volumes using a **stride of 100**, producing **216** patches per core (6×6×6).

**Downsampling.** Each 500³ patch is downsampled to **100³** using **5×5×5 block averaging**, yielding a continuous volume (values in [0, 1]) that preserves porosity under coarse graining.

### 2) Surface conditioning construction (faces + features)

For each 100³ patch, AB‑CDM randomly selects **three mutually orthogonal faces** from the six cube faces. For each selected face, the pipeline constructs a **3‑channel surface tensor**:

1. **Raw surface** (binary surface slice)
2. **Porosity** of the surface
3. **Specific surface area (SSA)** of the surface

In addition, each selected face is encoded by a **face ID** (`1–6`).  
The repository supports **eight surface‑combination patterns** used for augmentation (see figure below), producing **8 (data, label)** pairs per patch (216×8 = **1728** samples per core).

![Surface-combination patterns](./Images/Conditional_Input.png)

> **Note on label layout.** In the training code, a label packs **three faces**, each as `[3 channels + 1 face‑ID]`, resulting in **12 channels** total.

---

## Network architecture

AB‑CDM follows a 3D U‑Net augmented with **attenuation‑based conditional guidance**:

1. **Time‑step embedding** – sinusoidal embedding projected to a learnable space for diffusion step *t*.
2. **Conditional embedding** – a compact 2D CNN encoder that embeds each surface (image + features).
3. **Addition block (attenuation guidance)** – injects 2D embeddings into 3D feature maps. The embedded surface features are broadcast along the corresponding axis and fused with 3D features using **learnable exponential decay** in both forward and reverse directions.
4. **Residual blocks (optional attention)** – 3D convolutions + time embedding + conditional fusion, with GroupNorm and Swish.
5. **Downsampling / upsampling** – hierarchical feature learning with skip connections.

![Network architecture and conditional diffusion process](./Images/Network_architecture.jpg)

---

## Training

Run:

```bash
python Train.py
```

Key hyper‑parameters in `modelConfig`:

- `data_path` – path to the HDF5 dataset (**1400** samples used for training).
- `save_dir` – directory for checkpoints and logs.
- `epoch` – number of epochs (default: **300**).
- `batch_size` – batch size (default: **4**).
- `T` – diffusion steps (default: **500**).
- `channel`, `channel_mult`, `num_res_blocks`, `dropout` – 3D U‑Net settings.
- `lr` – base learning rate; **decay‑parameter groups** in the Addition Blocks use **2× lr**.
- `beta_1`, `beta_T` – linear noise schedule endpoints.
- `multiplier` – warmup peak multiplier (default: **2.5**).
- `grad_clip` – gradient clipping (default: **1.0**).

During training, the optimizer is **AdamW**, and the scheduler uses **warmup (first 10% epochs) → cosine annealing**. Mixed‑precision training is enabled, and checkpoints are saved periodically along with a log of learned decay parameters.

---

## Evaluation

Run:

```bash
python Eval.py
```

`Eval.py` loads a trained checkpoint and samples 3D volumes from Gaussian noise using `GaussianDiffusionSampler`, conditioned on the provided surface labels.

- The test split uses **328** samples.
- Generated volumes and their ground truth are saved as `.npy`.
- **SSIM** and **MS‑SSIM** are computed per sample and appended to CSV files in the output directory.

---

## Post‑processing and analysis

### Porosity (`Porosity.ipynb`)

This notebook computes **volumetric porosity** for both original and reconstructed volumes. It supports repeated stochastic sampling; in the default workflow, it performs **20 independent sampling runs** for all **328** test samples and reports the mean porosity per volume.

### Two‑point correlation function (`two_point_correlation.ipynb`)

This notebook evaluates spatial statistics via the **solid‑phase two‑point correlation function** \(S_2(r)\). For each reconstructed volume, \(S_2(r)\) is computed and then averaged over **20 sampling runs**, enabling direct comparison with ground truth across distances.

---

## Results

Qualitative examples on **Berea Upper Gray Sandstone** and **Claystone** show that AB‑CDM preserves pore connectivity and morphology consistent with ground truth, with strong agreement in voxel‑wise metrics ($R²$ / SSIM / MS‑SSIM) and physical/statistical measures (porosity, \(S_2(r)\), permeability).

![Reconstruction examples and statistics](./Images/Reconstruction.jpg)

The two‑point correlation curves for both datasets show that reconstructions closely overlap the reference across distances:

![Two‑point correlation function comparison](./Images/Two-point_correlation_functions.jpg)

---

## Resources and pre‑trained models

To reproduce the results, download the `.h5` dataset and set `data_path` in `Train.py`. For inference, set `test_load_weight` in `Eval.py` to the checkpoint filename.

---

## Acknowledgements

The overall code structure was inspired by the open‑source DDPM implementation from `zoubohao/DenoisingDiffusionProbabilityModel-ddpm-`. Thanks to the original authors for sharing their codebase.
