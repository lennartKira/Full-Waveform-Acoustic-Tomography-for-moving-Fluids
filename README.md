# Full-Waveform-Acoustic-Tomography-in-Fluids
This is a Python Code for conducting synthetic Tests on Full-Waveform Acoustic Tomography in moving Fluids.

To get started, download the repository and create the conda environment "FWI_env". Do this by navigating to the location of the folder using the command window and typing
'conda env create -f FWI_env.yml'

Alternatively create the environment 'FWI_env' yourself ('conda env create --name FWI_env python=3.12') and install the following packages:
-numpy
-matplotlib
-scipy
-torchaudio and torchvision (for GPU computing we recommend installing the torch version fitting your unit's cuda capabilities - see https://pytorch.org/get-started/locally/)
-ipykernel
-h5py
-(jupyter)

Then activate the environment with
'conda activate FWI_env'.

When completed, use Jupyter Notebook (or an alternative) to run the Notebooks in the **Folder "Target Models"** with the environment active. By running this notebook, you will create the true model data.

Afterwards navigate into the **Folder "./Code/Notebooks"** and run the Notebooks insde:

**'00_Generate Synthetic Waveforms'**\
A synthetic forward wavefield is simulated and recorded at defined receiver positions.
Loads a target model from the folder 'Target Models' (Wiens, Precession, BuoyancyFree) and computes the synthetic observed waveforms.
Optionally the snapshots can be saved to be animated later.


**'01_Display Synthetic Wavefields'**\
This Notebook loads snapshots of the synthetic wavefields and displays them as a video.
Additionally the waveforms can be accessed and visualized.

---
#### Note: If your machine has no cuda capability, the simulations will run on the cpu. High resolution simulations may be prohibitively expensive in this case! It is not recommended to proceed and compute sensitivity kernels for low resolution models, since we have not assessed the effect of grid dispersion on the results*.
---

**'02_Sensitivity Kernels Example'**\
Using the true data and a prior model, this routine computes the Gradients of the Misfit (Sensitivity Kernels) and displays them for demonstrative purposes.
The true data loaded here should be synthetic data generated by using a specific source-receiver pair - not by a complete transducer array.

**'03_Full-Waveform Inversion_Simultaneous Aquisition'**\
This routine uses the true Waveform Data, where all sources are fired simultaneously within one aquisition, and performs the inversion. The Output files are
- 'Data_m{k}_SimAqu' including the reconstruction of the k-th iteration, as well as the estimated waveforms
- 'Gradient_m{k}_SimAqu' including the gradient of the k-th iteration, as well as the adjoint source
- 'Iteration-{k}_simAqu' including the information on previous gradients and misfits, as well as the misfit history and the widths sigma_k of the filters applied to the gradients.

**'04_Evaluation'**\
This notebook can be used to load the reconstructions of the FWI and SRI.
By doing so, the evolution of the misfit and the choice of the widths of the filters can be reviewed.
Furthermore, this routine is used to average the FWI results onto the straight-ray grid and compute the reconstruction errors - and store them for display.

----
*If you lack computation power and are interested in experimenting with the kernels and full-waveform reconstructions using the Notebooks 02 and 04, please contact us. We have prepared high resolution data to access with these Notebooks.
