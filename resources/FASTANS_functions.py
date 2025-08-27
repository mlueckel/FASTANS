#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 15:19:08 2025

@author: maximilian
"""

#==============================================================================
# DEPENDENCIES
#
# - Connectome Workbench (wb_command)
# - Freesurfer (mris_convert, mris_combine)
#
#
# - "resources" folder
#    (here: /media/maximilian/e4713b47-344e-4ac6-85dd-b6769e0cbfa81/FASTANS/resources)
#
#==============================================================================

resources_folder = '/media/maximilian/e4713b47-344e-4ac6-85dd-b6769e0cbfa81/FASTANS/resources'

#==============================================================================
# Helper functions
#==============================================================================

def get_indices(lst, targets):
    return [index for index, element in enumerate(lst) if element in targets]



def load_cifti_values(cifti_filepath):
    
    import os
    import numpy as np
    import nibabel as nib
    
    cifti_template_cortex_filepath = os.path.join(resources_folder, 'ones_CORTEX.dscalar.nii')
    
    ### load template data
    cifti_template_cortex = nib.load(cifti_template_cortex_filepath)
    cifti_template_cortex_values = cifti_template_cortex.get_fdata()[0]
    
    ### load cifti (values)
    cifti = nib.load(cifti_filepath)
    cifti_brainmodelaxis = cifti.header.get_axis(1)
    
    cifti_values = cifti.get_fdata()[0]
    cifti_values = cifti_values[cifti_brainmodelaxis.surface_mask]
    
    if len(cifti_values) == 59412:
        cifti_values_tmp = np.zeros(64984)
        cifti_values_tmp[cifti_template_cortex_values == 1] = cifti_values
        cifti_values = cifti_values_tmp

    return cifti_values



def array_to_cifti_dscalar(array, cifti_template_filepath, output_filepath):
    
    import os
    import numpy as np
    
    output_path = os.path.split(output_filepath)[0]
    output_filename = os.path.split(output_filepath)[1].replace('.dscalar.nii', '')
    
    os.makedirs(output_path, exist_ok=True)
    
    np.savetxt(os.path.join(output_path, output_filename + '.txt'), array)
    
    os.system('wb_command -cifti-convert -from-text {} {} {}'.format(
        os.path.join(output_path, output_filename + '.txt'),
        cifti_template_filepath,
        output_filepath))
    
    os.system('rm ' + os.path.join(output_path, output_filename + '.txt'))
    


def cifti_dscalar_to_dlabel(dscalar_filepath, dlabel_template_filepath, output_type=None):
    
    import os
    
    dscalar_path = os.path.split(dscalar_filepath)[0]
    dscalar_filename = os.path.split(dscalar_filepath)[1].replace('.dscalar.nii', '')
    
    dlabel_template_path = os.path.split(dlabel_template_filepath)[0]
    dlabel_template_filename = os.path.split(dlabel_template_filepath)[1].replace('.dlabel.nii', '')
    
    if output_type == 'binary':
        
        os.system('wb_command -cifti-label-import {} {} {}'.format(
            dscalar_filepath,
            os.path.join(resources_folder, 'ROILabelsTempalte.txt'),
            dscalar_path + '/' + dscalar_filename + '.dlabel.nii'))
          
    elif not output_type:
        
        if os.path.split(dlabel_template_filepath)[1][-10:] == 'dlabel.nii':
        
            os.system('wb_command -cifti-label-export-table {} 1 {}'.format(
                dlabel_template_filepath,
                dlabel_template_path + '/' + dlabel_template_filename + '_NetworkLabels.txt'))
            
            os.system('wb_command -cifti-label-import {} {} {}'.format(
                dscalar_filepath,
                dlabel_template_path + '/' + dlabel_template_filename + '_NetworkLabels.txt',
                dscalar_path + '/' + dscalar_filename + '.dlabel.nii'))
            
            os.system('rm ' + dlabel_template_path + '/' + dlabel_template_filename + '_NetworkLabels.txt')
            
        elif os.path.split(dlabel_template_filepath)[1][-3:] == 'txt':
    
            os.system('wb_command -cifti-label-import {} {} {}'.format(
                dscalar_filepath,
                dlabel_template_filepath,
                dscalar_path + '/' + dscalar_filename + '.dlabel.nii'))
            
            

def extract_parcel(parcellation_filepath, parcel_id_list, output_filepath):
    
    import os
    import numpy as np

    cifti_template_cortex_filepath = os.path.join(resources_folder, 'ones_CORTEX.dscalar.nii')
    
    output_path = os.path.split(output_filepath)[0]
    output_filename = os.path.split(output_filepath)[1].replace('.dlabel.nii', '')
    
    parcellation_values = load_cifti_values(parcellation_filepath)
    parcel_indices = get_indices(parcellation_values, parcel_id_list)
    
    parcel_values = np.zeros([64984, 1]).flatten()
    parcel_values[parcel_indices] = parcellation_values[parcel_indices]
    array_to_cifti_dscalar(parcel_values, cifti_template_cortex_filepath, os.path.join(output_path, output_filename + '.dscalar.nii'))
    cifti_dscalar_to_dlabel(os.path.join(output_path, output_filename + '.dscalar.nii'), parcellation_filepath)
    os.system('rm ' + os.path.join(output_path, output_filename + '.dscalar.nii'))
    
    parcel_values_binary = np.zeros([64984, 1]).flatten()
    parcel_values_binary[parcel_indices] = 1
    array_to_cifti_dscalar(parcel_values_binary, cifti_template_cortex_filepath, os.path.join(output_path, output_filename + '_binary.dscalar.nii'))
    cifti_dscalar_to_dlabel(os.path.join(output_path, output_filename + '_binary.dscalar.nii'), parcellation_filepath, output_type='binary')
    os.system('rm ' + os.path.join(output_path, output_filename + '_binary.dscalar.nii'))

    
    
def mask_cifti(cifti_filepath, suffix, mask_filepath, mask_type, mask_threshold=None):
    
    import os
    
    cifti_template_cortex_filepath = os.path.join(resources_folder, 'ones_CORTEX.dscalar.nii')
    
    cifti_path = os.path.split(cifti_filepath)[0]
    cifti_filename = os.path.split(cifti_filepath)[1].replace('.dlabel.nii', '')
    cifti_filename = os.path.split(cifti_filename)[1].replace('.dscalar.nii', '')    
    
    cifti_values = load_cifti_values(cifti_filepath)
    mask_values = load_cifti_values(mask_filepath)
    
    if mask_type == 'binary':
        
        cifti_values_masked = cifti_values.copy()
        cifti_values_masked[mask_values == 0] = 0
        
        array_to_cifti_dscalar(cifti_values_masked, cifti_template_cortex_filepath, os.path.join(cifti_path, cifti_filename + '_' + suffix + '.dscalar.nii'))
        cifti_dscalar_to_dlabel(os.path.join(cifti_path, cifti_filename + '_' + suffix + '.dscalar.nii'), cifti_filepath)
        
        os.system('rm '+ os.path.join(cifti_path, cifti_filename + '_' + suffix + '.dscalar.nii'))
        
    elif mask_type == "metric":
        
        cifti_values_masked = cifti_values.copy()
        cifti_values_masked[mask_values < mask_threshold] = 0
        
        array_to_cifti_dscalar(cifti_values_masked, cifti_template_cortex_filepath, os.path.join(cifti_path, cifti_filename + '_' + suffix + '.dscalar.nii'))
        cifti_dscalar_to_dlabel(os.path.join(cifti_path, cifti_filename + '_' + suffix + '.dscalar.nii'), cifti_filepath)
        
        os.system('rm '+ os.path.join(cifti_path, cifti_filename + '_' + suffix + '.dscalar.nii'))


def cifti_extract_largest_cluster(cifti_filepath, surface_midthickness_left_filepath, surface_midthickness_right_filepath):
    
    import os
    
    cifti_path = os.path.split(cifti_filepath)[0]
    cifti_filename = os.path.split(cifti_filepath)[1].replace('.dlabel.nii', '')
    cifti_filename = os.path.split(cifti_filename)[1].replace('.dscalar.nii', '')    
    
    os.system('wb_command -cifti-find-clusters {} 0 0 0 0 COLUMN {} -left-surface {} -right-surface {} -size-ratio 100 100'.format(
        cifti_filepath,
        os.path.join(cifti_path, cifti_filename + '_TargetPatch.dscalar.nii'),
        surface_midthickness_left_filepath,
        surface_midthickness_right_filepath))
    
    cifti_dscalar_to_dlabel(os.path.join(cifti_path, cifti_filename + '_TargetPatch.dscalar.nii'), os.path.join(resources_folder, 'ROILabelsTempalte.txt'))
    
    os.system('rm ' + os.path.join(cifti_path, cifti_filename + '_TargetPatch.dscalar.nii'))
    


def extract_target_coordinates(cifti_filepath, surface_midthickness_left_filepath, surface_midthickness_right_filepath):
    
    import os
    import numpy as np
    import nibabel as nib
    
    # load cifti values
    cifti_values = load_cifti_values(cifti_filepath)
    
    # get paths
    surface_midthickness_left_path = os.path.split(surface_midthickness_left_filepath)[0]
    surface_midthickness_right_path = os.path.split(surface_midthickness_right_filepath)[0]
    
    # get coordinates of surface vertices    
    # left hemisphere
    os.system('wb_command -surface-coordinates-to-metric {} {}'.format(
        surface_midthickness_left_filepath,
        surface_midthickness_left_path + '/midthickness_surface_coordinates_left.func.gii'))
    
    # right hemisphere
    os.system('wb_command -surface-coordinates-to-metric {} {}'.format(
        surface_midthickness_right_filepath,
        surface_midthickness_right_path + '/midthickness_surface_coordinates_right.func.gii'))
    
    # load surface coordinates
    coordinates_midthickness_surface_left = nib.load(surface_midthickness_left_path + '/midthickness_surface_coordinates_left.func.gii')
    coordinates_midthickness_surface_left = np.transpose(np.asarray([coordinates_midthickness_surface_left.darrays[0].data, coordinates_midthickness_surface_left.darrays[1].data, coordinates_midthickness_surface_left.darrays[2].data]))
    coordinates_midthickness_surface_right = nib.load(surface_midthickness_right_path + '/midthickness_surface_coordinates_right.func.gii')
    coordinates_midthickness_surface_right = np.transpose(np.asarray([coordinates_midthickness_surface_right.darrays[0].data, coordinates_midthickness_surface_right.darrays[1].data, coordinates_midthickness_surface_right.darrays[2].data]))
    
    coordinates_midthickness_surface = np.concatenate([coordinates_midthickness_surface_left, coordinates_midthickness_surface_right])
    
    # extract coordinates within target patch
    coordinates_cifti = coordinates_midthickness_surface[cifti_values == 1]
    
    # compute center of target patch coordinates as their mean value
    coordinates_cifti_center = np.mean(coordinates_cifti, axis=0)
    
    return coordinates_cifti_center



def generate_search_grid(m2m_path, output_folderpath, target_coordinates, coil_scalp_distance, grid_radius, grid_resolution, angle_resolution, angle_limits, target_radius):
    
    import os
    import shutil
    import glob
    import simnibs
    from simnibs import optimization
    import numpy as np
    
    grid_output_folder = os.path.join(output_folderpath, 'SearchGrid_CoilPlacements')
    
    if os.path.exists(grid_output_folder):
        shutil.rmtree(grid_output_folder)
    
    os.makedirs(grid_output_folder, exist_ok=True)
    
    # Load brain mesh
    mesh_filepath = glob.glob(m2m_path + '/*.msh')[0]
    mesh = simnibs.read_msh(mesh_filepath)
    
    # Generate small target region in mesh
    target_region = optimization.tms_optimization.define_target_region(mesh = mesh,
                                                                        target_position = target_coordinates,
                                                                        target_radius = target_radius,
                                                                        tags = [2])
    
    # Generate grid of coil positions and orientations
    search_grid = optimization.tms_optimization.get_opt_grid(mesh = mesh,
                                                             pos = target_coordinates,
                                                             #handle_direction_ref = reference_pos,
                                                             distance  = coil_scalp_distance,
                                                             radius = grid_radius,
                                                             resolution_pos = grid_resolution,
                                                             resolution_angle = angle_resolution,
                                                             angle_limits  = angle_limits)
    
    # Write mesh with target and grid
    output_mesh_filepath = os.path.join(grid_output_folder, 'SearchGrid.msh')
    target_mesh = np.zeros(mesh.elm.nr)
    target_mesh[target_region - 1] = 1
    mesh.add_element_field(target_mesh, 'TargetSphere')
    mesh.write(output_mesh_filepath)
    v = mesh.view(visible_tags=[2], 
                  visible_fields='all')
    v.View[0].CustomMax = 1
    v.View[0].CustomMin = 0
    mesh.elmdata = []
    optimization.tms_optimization.plot_matsimnibs_list(search_grid, 
                                                    np.ones(len(search_grid)),
                                                    "Search grid for coil placement optimization",
                                                    os.path.join(grid_output_folder, 'CoilPositions.geo'))
    v.add_merge(os.path.join(grid_output_folder, 'CoilPositions.geo'))
    v.add_view(CustomMax=1, CustomMin=1,
               VectorType=4, CenterGlyphs=0,
               Visible=1, ColormapNumber=0)
    v.write_opt(output_mesh_filepath)
    
    return search_grid



def combine_label_files(gii_left, gii_right):
    
    import os
    import nibabel as nib
    import numpy as np
    
    gii_filepath = os.path.split(gii_left)[0]
    gii_filename = os.path.split(gii_left)[1].replace('.L.label.gii', '')
    
    # load label.gii files
    left_gii = nib.load(gii_left)
    right_gii = nib.load(gii_right)
    
    # extract data arrays
    left_data = left_gii.darrays[0].data
    right_data = right_gii.darrays[0].data
    
    # combine values across hemispheres
    combined_data = np.concatenate([left_data, right_data])
    
    # create giti structure
    combined_gii = nib.gifti.GiftiImage()
    combined_gii.add_gifti_data_array(nib.gifti.GiftiDataArray(data=combined_data, intent='NIFTI_INTENT_LABEL'))
    
    # use lable table if available
    if left_gii.labeltable is not None:
        combined_gii.labeltable = left_gii.labeltable
    
    # save new label.gii file
    nib.save(combined_gii, gii_filepath + '/' + gii_filename + '.LR.label.gii')    
    
    
    
def create_simnibs_roi(cifti_filepath, roi_id, roi_name, output_folderpath, m2m_path, surface_midthickness_left_filepath, surface_midthickness_right_filepath):
    
    import os
    import glob
    import simnibs
    from nipype.interfaces.freesurfer import MRIsConvert, MRIsCombine
    
    
    os.makedirs(os.path.join(output_folderpath, 'SimNIBS_ROIs', roi_name), exist_ok=True)
    
    
    # Combine midthickness surface (+ rename when starting with lh.)
    combine = MRIsCombine(in_files = [surface_midthickness_left_filepath, surface_midthickness_right_filepath],
                          out_file = os.path.split(surface_midthickness_left_filepath)[0] + '/' + os.path.split(surface_midthickness_left_filepath)[1].replace('.L.', '.LR.'))
    combine.run()


    # Separate dlabel.nii file into left and right label.gii files
    os.system('wb_command -cifti-separate {} COLUMN -label CORTEX_LEFT {}'.format(
        cifti_filepath,
        os.path.split(cifti_filepath)[0] + '/' + os.path.split(cifti_filepath)[1].replace('.dlabel.nii', '.L.label.gii')))

    os.system('wb_command -cifti-separate {} COLUMN -label CORTEX_RIGHT {}'.format(
        cifti_filepath,
        os.path.split(cifti_filepath)[0] + '/' + os.path.split(cifti_filepath)[1].replace('.dlabel.nii', '.R.label.gii')))


    # Combine left and right label.gii files
    gii_left = os.path.split(cifti_filepath)[0] + '/' + os.path.split(cifti_filepath)[1].replace('.dlabel.nii', '.L.label.gii')
    gii_right = os.path.split(cifti_filepath)[0] + '/' + os.path.split(cifti_filepath)[1].replace('.dlabel.nii', '.R.label.gii')
    combine_label_files(gii_left, gii_right)


    # Convert combined label.gii file to .annot files
    gii_left_right = os.path.split(cifti_filepath)[0] + '/' + os.path.split(cifti_filepath)[1].replace('.dlabel.nii', '.LR.label.gii')
    annot_left_right = os.path.split(cifti_filepath)[0] + '/' + os.path.split(cifti_filepath)[1].replace('.dlabel.nii', '.LR.annot')
    os.system('mris_convert --annot {} {} {}'.format(
        gii_left_right,
        surface_midthickness_left_filepath.replace('.L.', '.LR.'),
        annot_left_right))


    # Create SimNIBS ROI
    head_mesh = simnibs.read_msh(glob.glob(m2m_path + '/*.msh')[0])

    roi_settings_dict = {'method': 'surface',
                         'subpath': m2m_path,
                         'surface_type': 'custom',
                         'surface_path': surface_midthickness_left_filepath.replace('.L.', '.LR.'),
                         'mask_space': ['subject'],    
                         'mask_path': [annot_left_right],
                         'mask_value': [roi_id]}
    
    roi = simnibs.utils.region_of_interest.RegionOfInterest(roi_settings_dict)
    roi.write_visualization(os.path.join(output_folderpath, 'SimNIBS_ROIs', roi_name), 'simnibs_roi_' + roi_name)
    
    
    # Remove intermediate files
    os.system('rm ' + gii_left)
    os.system('rm ' + gii_right)
    os.system('rm ' + gii_left_right)
    os.system('rm ' + annot_left_right)
        
    return roi



def simnibs_accelerated_simulations(search_grid, simnibs_roi_list, m2m_path, coil_filepath, didt):
    
    import os
    import glob
    import numpy as np
    import simnibs
    from simnibs.simulation.onlinefem import OnlineFEM, FemTargetPointCloud
    
    
    head_mesh = simnibs.read_msh(glob.glob(m2m_path + '/*.msh')[0])
    
    
    simnibs_roi_FemTargetPointCloud_list = []
    for simnibs_roi in simnibs_roi_list:
        simnibs_roi_nodes = simnibs_roi.get_nodes()
        simnibs_roi_FemTargetPointCloud_list.append(FemTargetPointCloud(mesh = head_mesh, center = simnibs_roi_nodes))

    
    # Create OnlineFEM structure
    ofem = OnlineFEM(mesh = glob.glob(m2m_path + '/*.msh')[0],
                     method = "TMS",
                     fn_coil = coil_filepath,
                     roi = simnibs_roi_FemTargetPointCloud_list,
                     useElements = False,
                     dataType = [0]*len(simnibs_roi_FemTargetPointCloud_list))
    
    # Compute E-field for coil positions in search grid
    e_list = []

    for i in np.arange(len(search_grid)):
        
        print('Simulating coil position ' + str(i+1) + ' of ' + str(len(search_grid)))
        
        e = ofem.update_field(matsimnibs = search_grid[i],
                              didt = didt)
        
        e_list.append(e)
        
        
    return e_list
    

    
