# FD-DeepLoc
This code comes with the paper: "Field dependent deep learning enables high-throughput whole-cell 3D super-resolution imaging".
  
![traverse_large_FOV_NPC_ast2_compressed2](https://user-images.githubusercontent.com/67769465/168802700-71ba4e5d-b57a-45b0-a069-c27e110e487e.gif)
![singletom20compressed](https://user-images.githubusercontent.com/67769465/168948622-6fac4509-407b-4886-932c-91acc73113ff.gif)
![Supplementary Movie 5 large FOV DOF NPC(edfig 9)](https://user-images.githubusercontent.com/67769465/168954401-9c4d006f-9431-433c-9d74-d28011dc8146.gif)
![Supplementary Movie 3 large FOV DOF mito(fig5)](https://user-images.githubusercontent.com/67769465/168954626-3c10257f-6f4b-49d4-aa70-6c608f609b18.gif)

#  Requirements
* FD-DeepLoc was tested on a workstation equipped with Windows 10 system, 128 GB of memory, an Intel(R) Core(TM) i9-11900K, 3.50GHz CPU, and an NVidia GeForce RTX 3080 GPU with 10 GB of video memory. To use FD-DeepLoc yourself, a computer with CPU memory ≥ 32GB and GPU memory ≥ 8GB is recommended since FD-DeepLoc would process large-FOV SMLM images (usually > 500GB).
* CUDA (>=10.1, https://developer.nvidia.com/cuda-toolkit-archive) is required for fast GPU-based PSF simulation and PSF fitting.
* For field-dependent aberration map calibration, we tested on Matlab 2020b with CUDA 10.1 on a Windows 10 system.
* The deep learning part of FD-DeepLoc is based on Python and Pytorch. We recommend conda (https://anaconda.org) to manage the environment and provide a `fd_deeploc_env.yaml` file under the folder `FD-DeepLoc/Field Dependent PSF Learning` to build the conda environment.
# How to run
A tutorial file `FD-DeepLoc tutorial.pdf` is provided under the main directory, which illustrates the procedures for 
1. Field-dependent aberration map calibration.
2. Field-dependent deep-learning localization network (including training and inference examples)

# Contact
For any questions / comments about this software, please contact [Li Lab](https://faculty.sustech.edu.cn/liym2019/en/).

# Copyright
Copyright (c) 2021 Li Lab, Southern University of Science and Technology, Shenzhen.
