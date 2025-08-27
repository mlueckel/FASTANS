# FASTANS - Fast Targeted Functional Network Stimulation
<img src="https://github.com/user-attachments/assets/c11ab74d-8d21-4092-a262-328c50185f49" width="600" />

**FASTANS** is an accelerated, Python-based implementation of the *Targeted Functional Network Stimulation (TANS)* approach described in Lynch et al. [1,2]. The github repository with the original Matlab code can be found here: https://github.com/cjl2007/Targeted-Functional-Network-Stimulation

[1] *Lynch, C. J., Elbau, I. G., Ng, T. H., Wolk, D., Zhu, S., Ayaz, A., ... & Liston, C. (2022). Automated optimization of TMS coil placement for personalized functional network engagement. Neuron, 110(20), 3263-3277.*

[2] *Lynch, C. J., Elbau, I. G., Zhu, S., Ayaz, A., Bukhari, H., Power, J. D., & Liston, C. (2023). Precision mapping and transcranial magnetic stimulation of individual-specific functional brain networks in humans. STAR protocols, 4(1), 102118.*


# Installation

**Software dependencies:**
- Connectome Workbench (https://humanconnectome.org/software/get-connectome-workbench)
- SimNIBS **4.5** (https://simnibs.github.io/simnibs/build/html/installation/simnibs_installer.html)

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
4. (Optional) To setup the menu icons, file associations, the MATLAB library and add SimNIBS to the system path, run the postinstall_simnibs script:
   ```bash
   mkdir $HOME/SimNIBS
   postinstall_simnibs --setup-links -d $HOME/SimNIBS
   ```
Note: These instruction are based on the installation intructions given on the SimNIBS website: https://simnibs.github.io/simnibs/build/html/installation/conda.html
