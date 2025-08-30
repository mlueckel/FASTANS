# FASTANS - Fast Targeted Functional Network Stimulation
<img src="https://github.com/user-attachments/assets/c11ab74d-8d21-4092-a262-328c50185f49" width="600" />

**FASTANS** is an accelerated, Python-based implementation of the *Targeted Functional Network Stimulation (TANS)* approach described in Lynch et al. [1,2]. The github repository with the original Matlab code can be found here: https://github.com/cjl2007/Targeted-Functional-Network-Stimulation

<img src="https://github.com/user-attachments/assets/72fab372-8161-4ff5-903e-76eaaab6ba0c" width="25" /> While the original implementation usually takes several hours to run [2], FASTANS allows optimization of TMS coil placements **within minutes**, without any particular need for high performance computing or parallelization.

[1] *Lynch, C. J., Elbau, I. G., Ng, T. H., Wolk, D., Zhu, S., Ayaz, A., ... & Liston, C. (2022). Automated optimization of TMS coil placement for personalized functional network engagement. Neuron, 110(20), 3263-3277.*

[2] *Lynch, C. J., Elbau, I. G., Zhu, S., Ayaz, A., Bukhari, H., Power, J. D., & Liston, C. (2023). Precision mapping and transcranial magnetic stimulation of individual-specific functional brain networks in humans. STAR protocols, 4(1), 102118.*


# Installation
1. Download/clone this code repository.
2. In *FASTANS_TMS_optimization_pipeline.py* and *FASTANS.py* (within the *code* folder), edit the *FASTANS_installation_folderpath* variable according to your download location of the FASTANS code repository.
3. Additionally: In *FASTANS_TMS_optimization_pipeline.py*, edit the *simnibs_installation_path* variable according to the location of your SimNIBS 4.5 installation - this is needed to find the correct TMS coil (.ccd) files provided by SimNIBS (usually stored in */SimNIBSInstallationFolder/resources/coil_models/Drakaki_BrainStim_2022/*).

**Software dependencies:**
- Connectome Workbench (https://humanconnectome.org/software/get-connectome-workbench) - needs to be available in $PATH.
- SimNIBS **4.5** (https://simnibs.github.io/simnibs/build/html/installation/simnibs_installer.html).

Recommended way of installing SimNIBS 4.5 (Linux):
1. Download and install the Miniconda Python 3 distribution (https://www.anaconda.com/docs/getting-started/miniconda/install#linux-terminal-installer).
2. Download the SimNIBS Linux environment file (https://github.com/simnibs/simnibs/blob/v4.5.0/environment_linux.yml).
3. Run in a terminal window:
   ```bash
   export PATH="$HOME/miniconda3/bin:$PATH" # This part can change depending on your miniconda installation
   conda env create -f ~/Downloads/environment_linux.yml # This part can change depending on your download location of the SimNIBS Linux environment file
   conda activate simnibs_env
   pip install https://github.com/simnibs/simnibs/releases/download/v4.5.0/simnibs-4.5.0-cp311-cp311-linux_x86_64.whl
   ```
4. Wihtin the SimNIBS conda environment, install a Python IDE, e.g., Spyder:
   ```bash
   pip install spyder
   ```
5. (Optional) To setup the menu icons, file associations, the MATLAB library and add SimNIBS to the system path, run the postinstall_simnibs script:
   ```bash
   mkdir $HOME/SimNIBS
   postinstall_simnibs --setup-links -d $HOME/SimNIBS
   ```
Note: These instructions are largely based on the installation instructions given on the SimNIBS website: https://simnibs.github.io/simnibs/build/html/installation/conda.html


# Example use
```
#=============================================================================
# FASTANS configuration
#=============================================================================

# Name of output folder
output_foldername = 'Frontoparietal'

# Full path to output folder
output_folderpath = os.path.join('/.../FASTANS/resources/example_data/FASTANS', output_foldername)

# Path to subject-specific SimNIBS m2m folder
m2m_folderpath = '/.../FASTANS/resources/example_data/m2m_FASTANS_example'

# Functional network map (.dlabel.nii/.dscalar.nii/.dtseries.nii file; 32k_fs_LR space)
FCmap_filepath = '/.../FASTANS/resources/example_data/PFM/PFM_Lynch2024priors.dlabel.nii'

# Subject midthickness surfaces (.surf.gii files; 32k_fs_LR space)
surface_midthickness_left_filepath = '/.../FASTANS/resources/example_data/data/anat/midthickness_surface_left.32k_fs_LR.surf.gii'
surface_midthickness_right_filepath = '/.../FASTANS/resources/example_data/data/anat/midthickness_surface_right.32k_fs_LR.surf.gii'

# Sulcal depth map (.dscalar.nii file; 32k_fs_LR space)
sulcal_depth_filepath = '/.../FASTANS/resources/example_data/data/anat/sulcal_depth.32k_fs_LR.dscalar.nii'

# Type of FC map ('metric' or 'parcellation') — this pipeline expects 'parcellation'.
FCmap_type = 'parcellation'

# Parcellation label IDs: targets and avoidance
target_ids = [9] # 9 = Frontoparietal
avoidance_ids = [13,14,1,2,3,4] # 13 = Salience, 14 = Cingulo-opercular, 1-4 = Default Mode (sub)networks

# Percentiles used to define E-field "hotspots" (higher = smaller, more focal)
hotspot_percentiles = np.arange(99.0, 99.9, 0.1)

# Search space restricting stimulation to left PFC (choose variant as needed)
search_space_filepath = os.path.join(FASTANS_installation_folderpath, 'resources', 'search_spaces', 'SearchSpace_PFC_L.dscalar.nii')
# Alternative search spaces:
# search_space_filepath = '/.../SearchSpace_PFC_L_noPCG.dscalar.nii'
# search_space_filepath = '/.../SearchSpace_PFC_L_noPCG+DMPFC.dscalar.nii'
# search_space_filepath = '/.../SearchSpace_PFC_L_noPCG+DMPFC+IFG.dscalar.nii'

# TMS coil model (SimNIBS naming); choose matching to actual hardware
coil_model = 'MagVenture_Cool-B65'
```
