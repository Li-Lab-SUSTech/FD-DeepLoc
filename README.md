# FD-DeepLoc
Field dependent deep learning enables high-throughput whole-cell 3D super-resolution imaging
## Abstract
Single-molecule localization microscopy (SMLM) in a typical wide-field setup has been widely used for investigating sub-cellular structures with super resolution. However, field-dependent aberrations restrict the field of view (FOV) to only few tens of micrometers. Here, we present a deep learning method for precise localization of spatially variant point emitters (FD-DeepLoc) over a large FOV covering the full chip of a modern sCMOS camera. Using a graphic processing unit (GPU) based vectorial PSF fitter, we can fast and accurately fit the spatially variant point spread function (PSF) of a high numerical aperture (NA) objective in the entire FOV. Combined with deformable mirror based optimal PSF engineering, we demonstrate high-accuracy 3D SMLM over a volume of ~180 × 180 × 5 μm3, allowing us to image mitochondria and nuclear pore complex in the entire cells in a single imaging cycle without hardware scanning - a 100-fold increase in throughput compared to the state-of-the-art.

![traverse_large_FOV_NPC_ast3_compressed2](https://user-images.githubusercontent.com/67769465/174561057-1745a2c5-fe0c-416c-ada7-a6f2e1902207.gif)
![Supplementary Movie 5 large FOV DOF NPC(edfig 9)](https://user-images.githubusercontent.com/67769465/168954401-9c4d006f-9431-433c-9d74-d28011dc8146.gif)
![Supplementary Movie 3 large FOV DOF mito(fig5)](https://user-images.githubusercontent.com/67769465/168954626-3c10257f-6f4b-49d4-aa70-6c608f609b18.gif)
![singletom20compressed](https://user-images.githubusercontent.com/67769465/168968411-34e482a1-2241-48d4-be09-48f3d43612c9.gif)


#  Requirements
* FD-DeepLoc was tested on a workstation equipped with Windows 10 system, 128 GB of memory, an Intel(R) Core(TM) i9-11900K, 3.50GHz CPU, and an NVidia GeForce RTX 3080 GPU with 10 GB of video memory. To use FD-DeepLoc yourself, a computer with CPU memory ≥ 32GB and GPU memory ≥ 8GB is recommended since FD-DeepLoc would process large-FOV SMLM images (usually > 500GB).
* CUDA (>=10.1, https://developer.nvidia.com/cuda-toolkit-archive) is required for fast GPU-based PSF simulation and PSF fitting.
* For field-dependent aberration map calibration, we tested on Matlab 2020b with CUDA 10.1 on a Windows 10 system.
* The deep learning part of FD-DeepLoc is based on Python and Pytorch. We recommend conda (https://anaconda.org) to manage the environment and provide a `fd_deeploc_env.yaml` file under the folder `FD-DeepLoc/Field Dependent PSF Learning` to build the conda environment.

# Installation
1. Download this repository (or clone it using git).
2. Open Anaconda Prompt and change the current directory to `FD-DeepLoc/Field Dependent PSF Learning`.
3. Use the command `conda env create -f fd_deeploc_env.yaml` to build the FD-DeepLoc environment, it may take several minutes.
4. Activate the environment using the command `conda activate fd_deeploc`, then check the demo using the command `jupyter notebook`.
5. The example bead stacks files for field-dependent aberration map calibration and network analysis example data can be found at: https://zenodo.org/record/6547064#.YoRrPKhByUk

# How to run
A detailed tutorial file `FD-DeepLoc tutorial.pdf` is provided under the main directory, which illustrates the procedures for 
1. Field-dependent aberration map calibration.
2. Field-dependent deep-learning localization network (including training and inference examples)

# Contact
For any questions / comments about this software, please contact [Li Lab](https://faculty.sustech.edu.cn/liym2019/en/).

# Copyright
Copyright (c) 2022 Li Lab, Southern University of Science and Technology, Shenzhen.
