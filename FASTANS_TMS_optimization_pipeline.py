#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FASTANS pipeline for optimizaing TMS coil placement(s)
=======================================================

This script configures and runs a two-stage FASTANS workflow to identify and
simulate TMS coil placements that maximize the induced E-field "hotspot" within
the selected functional networks while minimizing it in avoidance networks.

High-level steps
----------------
1) Define all paths, parameters and software locations.
2) Extract target and avoidance parcels from a parcellation.
3) Constrain the targets by (a) a PFC search space and (b) sulcal crowns and
   keep the largest connected cluster ("TargetPatch").
4) Build a coarse search grid of coil placements and run accelerated
   simulations.
5) Score coil placements by hotspot overlap with target/avoidance networks and
   keep the best one(s).
6) Build a fine grid around the best coarse placement, re-simulate, and
   rescore to obtain the final best placements.
7) Run full SimNIBS simulations for the final best placements and export:
   - Localite XML markers for multiple coil–scalp distances
   - dtseries of E-field per distance for quick inspection

Requirements
------------
- Valid SimNIBS m2m subject directory (``*.msh`` present)
- HCP 32k_fs_LR surfaces and parcellations
- Connectome Workbench available in PATH
- SimNIBS version 4.5

Author: Maximilian Lueckel, mlueckel@uni-mainz.de
"""
import os
import numpy as np

#=============================================================================
# Specify software locations
#=============================================================================

FASTANS_installation_folderpath = '/media/maximilian/e4713b47-344e-4ac6-85dd-b6769e0cbfa81/FASTANS'
simnibs_installation_path = '/home/maximilian/SimNIBS-4.5'

#=============================================================================
# FASTANS configuration
#=============================================================================

# Name of output folder
output_foldername = 'output_foldername'

# Full path to output folder
output_folderpath = os.path.join('/path/to/output/folder', output_foldername)

# Path to subject-specific SimNIBS m2m folder
m2m_folderpath = ''

# Functional network map (.dlabel.nii/.dscalar.nii/.dtseries.nii file; 32k_fs_LR space)
FCmap_filepath = '/path/to/m2m_folder'

# Subject midthickness surfaces (.surf.gii files; 32k_fs_LR space)
surface_midthickness_left_filepath = '/path/to/left_midthickness_surface.surf.gii'
surface_midthickness_right_filepath = '/path/to/right_midthickness_surface.surf.gii'

# Sulcal depth map (.dscalar.nii file; 32k_fs_LR space)
sulcal_depth_filepath = '/path/to/suclal_depth.dscalar.nii'

# Type of FC map ('metric' or 'parcellation') — this pipeline expects 'parcellation'.
FCmap_type = 'parcellation'

# Parcellation label IDs: targets and avoidance
target_ids = [16]       # 16 = Somatomotor_Hand
avoidance_ids = [17,18] # 17 = Somatomotor_Face, 18 = Somatomotor_Foot

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

#=============================================================================
# Simulation options
#=============================================================================

# Coil–scalp distance in mm (baseline; final stage also evaluates d..d+9 mm)
coil_scalp_distance = 1

# Induced-current slew rate (A/s)
didt = 1 * 1e6

# Number of best placements to keep at each stage
n_placements = 1

# Coil model file
coil_filepath = os.path.join(simnibs_installation_path, 'resources', 'coil_models', 'Drakaki_BrainStim_2022', coil_model + '.ccd')

#=============================================================================
# Load FASTANS
#=============================================================================

import sys
sys.path.append(os.path.join(FASTANS_installation_folderpath, 'code'))
os.chdir(os.path.join(FASTANS_installation_folderpath, 'code'))
import FASTANS

#=============================================================================
# Pipeline
#=============================================================================

# Prepare output directory
os.makedirs(output_folderpath, exist_ok=True)

# -- (1) Extract target & avoidance regions ----------------------------------
FASTANS.extract_parcel(FCmap_filepath, target_ids,    os.path.join(output_folderpath, 'TargetRegions.dlabel.nii'))
FASTANS.extract_parcel(FCmap_filepath, avoidance_ids, os.path.join(output_folderpath, 'AvoidanceRegions.dlabel.nii'))

# -- (2) Constrain target by (a) search space and (b) sulcal crown ------------
# (a) Search space masking + keep largest cluster
FASTANS.mask_cifti(os.path.join(output_folderpath, 'TargetRegions.dlabel.nii'),
                   'SearchSpace',
                   search_space_filepath,
                   'binary')
FASTANS.cifti_extract_largest_cluster(os.path.join(output_folderpath, 'TargetRegions_SearchSpace.dlabel.nii'),
                                      surface_midthickness_left_filepath,
                                      surface_midthickness_right_filepath)

# (b) Sulcal crown (metric) threshold + keep largest cluster
FASTANS.mask_cifti(os.path.join(output_folderpath, 'TargetRegions_SearchSpace.dlabel.nii'),
                   'SulcalCrown',
                   sulcal_depth_filepath,
                   'metric',
                   mask_threshold=0.5)
FASTANS.cifti_extract_largest_cluster(os.path.join(output_folderpath, 'TargetRegions_SearchSpace_SulcalCrown.dlabel.nii'),
                                      surface_midthickness_left_filepath,
                                      surface_midthickness_right_filepath)

# -- (3) Coarse grid generation over the target patch -------------------------
target_coordinates = FASTANS.extract_target_coordinates(os.path.join(output_folderpath, 'TargetRegions_SearchSpace_TargetPatch.dlabel.nii'),
                                                        surface_midthickness_left_filepath,
                                                        surface_midthickness_right_filepath)

search_grid_coarse = FASTANS.generate_search_grid(m2m_folderpath,
                                                  os.path.join(output_folderpath, 'SimNIBS', 'SearchGrid', 'Step1_coarse'),
                                                  target_coordinates,
                                                  coil_scalp_distance,
                                                  35,  # radius
                                                  10,  # resolution
                                                  30,  # angle resolution
                                                  [-90, 60])

# -- (4) Accelerated simulations over the cortex ------------------------------
simulation_results_cortex = FASTANS.simnibs_accelerated_simulations_cortex(search_grid_coarse,
                                                                           coil_filepath,
                                                                           didt,
                                                                           m2m_folderpath,
                                                                           os.path.join(output_folderpath, 'SimNIBS', 'SearchGrid', 'Step1_coarse'),
                                                                           surface_midthickness_left_filepath,
                                                                           surface_midthickness_right_filepath)
# (you can reload later instead of re-running)
search_grid_coarse, simulation_results_cortex = FASTANS.load_simulation_results(os.path.join(output_folderpath, 'SimNIBS', 'SearchGrid', 'Step1_coarse', 'simulation_results.pickle'))

# -- (5) Rank placements by hotspot overlap ----------------------------------
best_coil_placements = FASTANS.extract_best_coil_placements_hotspot(simulation_results_cortex,
                                                                    search_grid_coarse,
                                                                    hotspot_percentiles,
                                                                    FCmap_filepath,
                                                                    target_ids,
                                                                    avoidance_ids,
                                                                    n_placements,
                                                                    surface_midthickness_left_filepath,
                                                                    surface_midthickness_right_filepath)

# -- (6) Fine grid around the best coarse placement ---------------------------
target_coordinates = best_coil_placements[0][0:3, 3]

search_grid_fine = FASTANS.generate_search_grid(m2m_folderpath,
                                                os.path.join(output_folderpath, 'SimNIBS', 'SearchGrid', 'Step2_fine'),
                                                target_coordinates,
                                                coil_scalp_distance,
                                                15,  # radius
                                                5,   # resolution
                                                10,  # angle resolution
                                                [-90, 80])

# Repeat accelerated sims and ranking on the fine grid
simulation_results_cortex = FASTANS.simnibs_accelerated_simulations_cortex(search_grid_fine,
                                                                           coil_filepath,
                                                                           didt,
                                                                           m2m_folderpath,
                                                                           os.path.join(output_folderpath, 'SimNIBS', 'SearchGrid', 'Step2_fine'),
                                                                           surface_midthickness_left_filepath,
                                                                           surface_midthickness_right_filepath)

search_grid_fine, simulation_results_cortex = FASTANS.load_simulation_results(os.path.join(output_folderpath, 'SimNIBS', 'SearchGrid', 'Step2_fine', 'simulation_results.pickle'))

best_coil_placements = FASTANS.extract_best_coil_placements_hotspot(simulation_results_cortex,
                                                                    search_grid_fine,
                                                                    hotspot_percentiles,
                                                                    FCmap_filepath,
                                                                    target_ids,
                                                                    avoidance_ids,
                                                                    n_placements,
                                                                    surface_midthickness_left_filepath,
                                                                    surface_midthickness_right_filepath)

# -- (7) Final FEM runs and exports -------------------------------------------
FASTANS.run_final_simulation(output_foldername,
                             best_coil_placements,
                             m2m_folderpath,
                             coil_filepath,
                             coil_scalp_distance,
                             didt,
                             os.path.join(output_folderpath, 'SimNIBS', 'Simulations'),
                             surface_midthickness_left_filepath,
                             surface_midthickness_right_filepath)
