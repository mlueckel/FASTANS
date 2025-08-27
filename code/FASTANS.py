#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FASTANS — Fast Targeted Functional Network Stimulation
===============================================================================

A collection of functions to interact with Connectome Workbenchand SimNIBS 4.5
in order to:
- Read and write CIFTI/GIFTI surface data
- Generate/search TMS coil placement grids
- Run accelerated electric field simulations
- Score coil placements against functional network targets/avoidance masks
- Export final placements
- Perform a fast precision functional mapping (PFM) from rs-fMRI
  timeseries using published priors

External command-line dependencies
----------------------------------
- Connectome Workbench: ``wb_command``
- SimNIBS version 4.5

File formats
------------
- CIFTI-2: ``.dscalar.nii``, ``.dlabel.nii``, ``.dtseries.nii``
- GIFTI: ``.surf.gii``, ``.label.gii``
- SimNIBS meshes: ``.msh``

The majority of functions are thin orchestrators around those tools and therefore
rely on the presence of the binaries in ``$PATH`` and on valid input files.

Author: Maximilian Lueckel, mlueckel@uni-mainz.de
"""
import os

#==============================================================================
# PATHS
#==============================================================================

FASTANS_installation_folderpath = '/media/maximilian/e4713b47-344e-4ac6-85dd-b6769e0cbfa81/FASTANS'
resources_folderpath = os.path.join(FASTANS_installation_folderpath, 'resources')

#==============================================================================
# Helper functions
#==============================================================================

def get_indices(lst, targets):
    """
    Return indices of elements in *lst* that are contained in *targets*.

    Parameters
    ----------
    lst : sequence
        Sequence of values (e.g., network labels per vertex).
    targets : set or sequence
        Set/sequence of values to select.

    Returns
    -------
    list[int]
        Indices i where ``lst[i]`` is in *targets*.
    """
    return [index for index, element in enumerate(lst) if element in targets]


def load_cifti_values(cifti_filepath):
    """
    Load values from a CIFTI file and return a cortex-length (64984) vector or
    a 2D array with columns aligned to the HCP 32k_fs_LR cortex blueprint.

    Notes
    -----
    - If the incoming CIFTI contains only cortical data (59412 vertices), this
      function pads it to 64984 using the cortex template mask.
    - Works for ``.dlabel.nii``, ``.dscalar.nii`` and ``.dtseries.nii``.

    Parameters
    ----------
    cifti_filepath : str
        Path to a CIFTI-2 file.

    Returns
    -------
    numpy.ndarray
        1D array (64984,) for a single map or 2D array (T, 64984) for series.
    """
    import os
    import numpy as np
    import nibabel as nib

    cifti_template_cortex_filepath = os.path.join(resources_folderpath, 'templates', 'CORTEX.32k_fs_LR.dscalar.nii')

    # Load cortex template mask (0/1 per surface vertex; length 64984).
    cifti_template_cortex = nib.load(cifti_template_cortex_filepath)
    cifti_template_cortex_values = cifti_template_cortex.get_fdata()[0]

    # Load data and pick the brainmodel axis mask to extract cortical vertices.
    cifti = nib.load(cifti_filepath)
    cifti_brainmodelaxis = cifti.header.get_axis(1)

    cifti_values = cifti.get_fdata()
    cifti_values = cifti_values[:, cifti_brainmodelaxis.surface_mask]

    if cifti_values.shape[0] == 1:
        # Single map -> flatten to 1D
        cifti_values = cifti_values[0]

        if len(cifti_values) == 59412:
            # Pad to 64984 by placing values at cortical indices in the template.
            cifti_values_tmp = np.zeros(64984)
            cifti_values_tmp[cifti_template_cortex_values == 1] = cifti_values
            cifti_values = cifti_values_tmp

    elif cifti_values.shape[0] > 1:
        # Multi-map (e.g., dtseries). Align each row to cortex blueprint length.
        if cifti_values.shape[1] == 59412:
            cifti_values_tmp = np.zeros([cifti_values.shape[0], 64984])
            for i in np.arange(cifti_values.shape[0]):
                cifti_values_tmp[i, cifti_template_cortex_values == 1] = cifti_values[i, :]
            cifti_values = cifti_values_tmp

    return cifti_values


def get_surface_vertex_areas(surface_filepath):
    """
    Compute per-vertex surface areas for a midthickness surface using Workbench.

    This is a light wrapper around the Workbench command
    ``wb_command -surface-vertex-areas`` that writes a temporary ``*.shape.gii``
    and reads it back via nibabel.

    Parameters
    ----------
    surface_filepath : str
        Path to a ``*.surf.gii`` (32k_fs_LR) surface file.

    Returns
    -------
    numpy.ndarray
        1D array of per-vertex areas (same length as surface vertices).
    """
    import os
    import nibabel as nib

    surface_path = os.path.split(surface_filepath)[0]
    surface_filename = os.path.split(surface_filepath)[1].replace('.surf.gii', '')

    os.system('wb_command -surface-vertex-areas {} {}'.format(
        surface_filepath,
        os.path.join(surface_path, surface_filename + '_va.shape.gii')))

    surface_vertex_area_values = nib.load(os.path.join(surface_path, surface_filename + '_va.shape.gii'))
    surface_vertex_area_values = surface_vertex_area_values.darrays[0].data

    return surface_vertex_area_values


def array_to_cifti_dscalar(array, cifti_template_filepath, output_filepath):
    """
    Save a 1D array to a ``.dscalar.nii`` by converting through an intermediate text file.

    Parameters
    ----------
    array : array_like
        Vector of length 64984 (cortex blueprint).
    cifti_template_filepath : str
        CIFTI file providing the target header/axes (e.g., CORTEX.32k_fs_LR).
    output_filepath : str
        Destination ``*.dscalar.nii`` path (directories will be created).

    Side Effects
    ------------
    Uses ``wb_command -cifti-convert -from-text`` and removes the temporary text.
    """
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


def array_to_cifti_dtseries(array, output_filepath):
    """
    Save a 2D array as a ``.dtseries.nii`` with the HCP blueprint mapping.

    Parameters
    ----------
    array : array_like, shape (T, 64984) or (64984, T)
        Time-by-vertex data. Internally written via ``-create-scalar-series``
        and then remapped to ``dtseries`` using a template.
    output_filepath : str
        Destination ``*.dtseries.nii`` path.

    Notes
    -----
    This function writes an intermediate ``.txt`` and ``.dscalar.nii``.
    """
    import os
    import numpy as np

    output_folderpath = os.path.split(output_filepath)[0]
    output_filename = os.path.split(output_filepath)[1].replace('.dtseries.nii', '')

    np.savetxt(os.path.join(output_folderpath, output_filename + '.txt'), array)

    os.system('wb_command -cifti-create-scalar-series {} {} -transpose'.format(
        os.path.join(output_folderpath, output_filename + '.txt'),
        os.path.join(output_folderpath, output_filename + '.dscalar.nii')))

    os.system('wb_command -cifti-change-mapping {} COLUMN {} -from-cifti {} COLUMN'.format(
        os.path.join(output_folderpath, output_filename + '.dscalar.nii'),
        output_filepath,
        os.path.join(resources_folderpath, 'templates', 'HCP_blueprint.dtseries.nii')))

    os.system('rm ' + os.path.join(output_folderpath, output_filename + '.txt'))
    os.system('rm ' + os.path.join(output_folderpath, output_filename + '.dscalar.nii'))


def cifti_dscalar_to_dlabel(dscalar_filepath, dlabel_template_filepath, output_type=None):
    """
    Convert a ``.dscalar.nii`` to a ``.dlabel.nii`` by importing a label table.

    Parameters
    ----------
    dscalar_filepath : str
        Source scalar map.
    dlabel_template_filepath : str
        Either (a) a ``.dlabel.nii`` whose label-table is exported and reused,
        or (b) a plain ``.txt`` label table compatible with Workbench.
    output_type : {'binary', None}, optional
        If ``'binary'``, use a binary ROI label template.

    Side Effects
    ------------
    Calls multiple ``wb_command`` operations and deletes temporary artifacts.
    """
    import os

    dscalar_path = os.path.split(dscalar_filepath)[0]
    dscalar_filename = os.path.split(dscalar_filepath)[1].replace('.dscalar.nii', '')

    dlabel_template_path = os.path.split(dlabel_template_filepath)[0]
    dlabel_template_filename = os.path.split(dlabel_template_filepath)[1].replace('.dlabel.nii', '')

    if output_type == 'binary':
        os.system('wb_command -cifti-label-import {} {} {}'.format(
            dscalar_filepath,
            os.path.join(resources_folderpath, 'templates', 'ROILabelsTemplate.txt'),
            dscalar_path + '/' + dscalar_filename + '.dlabel.nii'))
    elif not output_type:
        if os.path.split(dlabel_template_filepath)[1][-10:] == 'dlabel.nii':
            # Export label table from template dlabel and import onto dscalar.
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
            
            
def combine_gifti_surfaces(surface_left_filepath, surface_right_filepath, output_filepath):
    """
    Combine two *.surf.gii (LH/RH) into a single surface.
    """
    import numpy as np
    import nibabel as nib

    L = nib.load(surface_left_filepath)
    R = nib.load(surface_right_filepath)
    vL, fL = L.darrays[0].data, L.darrays[1].data
    vR, fR = R.darrays[0].data, R.darrays[1].data

    v = np.vstack([vL, vR])
    fR_off = (fR + vL.shape[0]).astype(np.int32)
    f = np.vstack([fL.astype(np.int32), fR_off])

    img = nib.gifti.GiftiImage()
    img.add_gifti_data_array(nib.gifti.GiftiDataArray(v, intent='NIFTI_INTENT_POINTSET'))
    img.add_gifti_data_array(nib.gifti.GiftiDataArray(f, intent='NIFTI_INTENT_TRIANGLE'))
    nib.save(img, output_filepath)
            
            
def gifti_label_to_fs_annot(gifti_label_path, surface_gifti_path, annot_out_path):
    """
    Convert a GIFTI label map (per-vertex ints) into a FreeSurfer .annot file,
    without calling external binaries. Preserves the original integer label IDs
    (e.g., 13/14 for Salience/CO), so downstream ROI selection by ID still works.
    """
    import numpy as np
    import nibabel as nib
    from nibabel.freesurfer import io as fsio

    # Load labels from GIFTI
    lab_img = nib.load(gifti_label_path)
    labels = lab_img.darrays[0].data.astype(np.int32)  # per-vertex label IDs

    # Optional: verify vertex count against the surface
    surf = nib.load(surface_gifti_path)
    n_vertices = surf.darrays[0].data.shape[0]
    if labels.shape[0] != n_vertices:
        raise ValueError(
            f"Label length ({labels.shape[0]}) != surface vertices ({n_vertices})"
        )

    # Collect label IDs we actually use
    used_ids = np.unique(labels).astype(int)

    # Build FS color table rows: [R, G, B, T, label_id], where T = 255 - alpha
    rows = []
    names = []
    id_seen = set()

    def _rgba255(rgba):
        # GIFTI stores floats 0..1; sometimes ints 0..255.
        r, g, b, a = rgba if rgba is not None else (0.0, 0.0, 0.0, 0.0)
        def _to255(x): return int(round(x * 255)) if (0.0 <= x <= 1.0) else int(round(x))
        R, G, B, A = map(_to255, (r, g, b, a))
        T = 255 - A
        return R, G, B, T

    # 1) Use the labeltable if present
    lt = getattr(lab_img, "labeltable", None)
    if lt is not None and getattr(lt, "labels", None):
        for lab in lt.labels:
            key = int(lab.key)
            R, G, B, T = _rgba255(lab.rgba)
            name = str(lab.label) if lab.label is not None else f"label_{key}"
            rows.append([R, G, B, T, key])
            names.append(name)
            id_seen.add(key)

    # 2) Ensure every used id has a row (in case labeltable was incomplete)
    for key in used_ids:
        if key in id_seen:
            continue
        # Background gets transparent black; others opaque white by default
        if key == 0:
            rows.append([0, 0, 0, 255, 0])
            names.append("???")
        else:
            rows.append([255, 255, 255, 0, key])
            names.append(f"label_{key}")
        id_seen.add(key)

    # Sort rows by label_id for deterministic output
    rows_sorted = sorted(rows, key=lambda r: r[4])
    names_sorted = [n for _, n in sorted(zip([r[4] for r in rows], names), key=lambda x: x[0])]
    ctab = np.asarray(rows_sorted, dtype=np.int32)

    # Write FS .annot (RGBT + label_id); keep our label IDs by setting fill_ctab=False
    fsio.write_annot(annot_out_path, labels=labels, ctab=ctab, names=names_sorted, fill_ctab=False)


def extract_parcel(parcellation_filepath, parcel_id_list, output_filepath):
    """
    Extract a subset of parcels from a parcellation and save as both label and
    (binary) ROI maps.

    Parameters
    ----------
    parcellation_filepath : str
        Path to an input ``.dlabel.nii`` parcellation (integer IDs).
    parcel_id_list : list[int]
        Label IDs to include.
    output_filepath : str
        Output path with ``.dlabel.nii`` suffix (without ``_binary``).

    Outputs
    -------
    - ``<output>.dlabel.nii`` with original IDs for the selected parcels
    - ``<output>_binary.dlabel.nii`` with 1s at selected parcels, 0 elsewhere
    """
    import os
    import numpy as np

    cifti_template_cortex_filepath = os.path.join(resources_folderpath, 'templates', 'CORTEX.32k_fs_LR.dscalar.nii')

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
    """
    Mask a CIFTI map with either a binary ROI or a metric threshold.

    Parameters
    ----------
    cifti_filepath : str
        Input map (``.dlabel.nii`` or ``.dscalar.nii``).
    suffix : str
        Suffix appended to the output filename (before extension).
    mask_filepath : str
        CIFTI file providing the mask values.
    mask_type : {'binary', 'metric'}
        - 'binary' : zero out vertices where mask == 0
        - 'metric' : zero out where mask < *mask_threshold*
    mask_threshold : float, optional
        Threshold for metric masking; ignored for binary masking.

    Outputs
    -------
    Writes ``<input>_<suffix>.dlabel.nii`` and removes intermediate scalar.
    """
    import os

    cifti_template_cortex_filepath = os.path.join(resources_folderpath, 'templates', 'CORTEX.32k_fs_LR.dscalar.nii')

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

        os.system('rm ' + os.path.join(cifti_path, cifti_filename + '_' + suffix + '.dscalar.nii'))

    elif mask_type == "metric":
        cifti_values_masked = cifti_values.copy()
        cifti_values_masked[mask_values < mask_threshold] = 0

        array_to_cifti_dscalar(cifti_values_masked, cifti_template_cortex_filepath, os.path.join(cifti_path, cifti_filename + '_' + suffix + '.dscalar.nii'))
        cifti_dscalar_to_dlabel(os.path.join(cifti_path, cifti_filename + '_' + suffix + '.dscalar.nii'), cifti_filepath)

        os.system('rm ' + os.path.join(cifti_path, cifti_filename + '_' + suffix + '.dscalar.nii'))


def cifti_extract_largest_cluster(cifti_filepath, surface_midthickness_left_filepath, surface_midthickness_right_filepath):
    """
    Find spatial clusters in a CIFTI map and keep only the largest one.

    Workbench is used to find clusters on the cortical surface. The result
    is stored as ``<input>_TargetPatch.dlabel.nii``.

    Parameters
    ----------
    cifti_filepath : str
        Path to input ``.dlabel.nii`` or scalar map.
    surface_midthickness_left_filepath, surface_midthickness_right_filepath : str
        32k_fs_LR midthickness surfaces used by Workbench.
    """
    import os

    cifti_path = os.path.split(cifti_filepath)[0]
    cifti_filename = os.path.split(cifti_filepath)[1].replace('.dlabel.nii', '')
    cifti_filename = os.path.split(cifti_filename)[1].replace('.dscalar.nii', '')

    os.system('wb_command -cifti-find-clusters {} 0 0 0 0 COLUMN {} -left-surface {} -right-surface {} -size-ratio 100 100'.format(
        cifti_filepath,
        os.path.join(cifti_path, cifti_filename + '_TargetPatch.dscalar.nii'),
        surface_midthickness_left_filepath,
        surface_midthickness_right_filepath))

    cifti_dscalar_to_dlabel(os.path.join(cifti_path, cifti_filename + '_TargetPatch.dscalar.nii'), os.path.join(resources_folderpath, 'templates', 'ROILabelsTemplate.txt'))

    os.system('rm ' + os.path.join(cifti_path, cifti_filename + '_TargetPatch.dscalar.nii'))


def extract_target_coordinates(cifti_filepath, surface_midthickness_left_filepath, surface_midthickness_right_filepath):
    """
    Compute the geometric center (mean coordinates) of the vertices within a
    binary target patch defined by a CIFTI map.

    Parameters
    ----------
    cifti_filepath : str
        ``.dlabel.nii`` with ones marking the target patch.
    surface_midthickness_left_filepath, surface_midthickness_right_filepath : str
        32k_fs_LR midthickness surfaces.

    Returns
    -------
    numpy.ndarray, shape (3,)
        XYZ coordinates (in surface space) of the center of the patch.
    """
    import os
    import numpy as np
    import nibabel as nib

    # Load binary mask values.
    cifti_values = load_cifti_values(cifti_filepath)

    # Prepare output locations for temporary coordinate metrics.
    surface_midthickness_left_path = os.path.split(surface_midthickness_left_filepath)[0]
    surface_midthickness_right_path = os.path.split(surface_midthickness_right_filepath)[0]

    # Left/right vertex coordinates -> func.gii (one array per axis).
    os.system('wb_command -surface-coordinates-to-metric {} {}'.format(
        surface_midthickness_left_filepath,
        surface_midthickness_left_path + '/midthickness_surface_coordinates_left.func.gii'))
    os.system('wb_command -surface-coordinates-to-metric {} {}'.format(
        surface_midthickness_right_filepath,
        surface_midthickness_right_path + '/midthickness_surface_coordinates_right.func.gii'))

    # Load coordinates and stack (LH then RH) to match HCP ordering.
    coordinates_midthickness_surface_left = nib.load(surface_midthickness_left_path + '/midthickness_surface_coordinates_left.func.gii')
    coordinates_midthickness_surface_left = np.transpose(np.asarray([coordinates_midthickness_surface_left.darrays[0].data, coordinates_midthickness_surface_left.darrays[1].data, coordinates_midthickness_surface_left.darrays[2].data]))
    coordinates_midthickness_surface_right = nib.load(surface_midthickness_right_path + '/midthickness_surface_coordinates_right.func.gii')
    coordinates_midthickness_surface_right = np.transpose(np.asarray([coordinates_midthickness_surface_right.darrays[0].data, coordinates_midthickness_surface_right.darrays[1].data, coordinates_midthickness_surface_right.darrays[2].data]))

    coordinates_midthickness_surface = np.concatenate([coordinates_midthickness_surface_left, coordinates_midthickness_surface_right])

    # Subselect vertices within the target patch (value==1) and average.
    coordinates_cifti = coordinates_midthickness_surface[cifti_values == 1]
    coordinates_cifti_center = np.mean(coordinates_cifti, axis=0)

    return coordinates_cifti_center


def generate_search_grid(m2m_path, output_folderpath, target_coordinates, coil_scalp_distance, grid_radius, grid_resolution, angle_resolution, angle_limits):
    """
    Generate a grid of candidate TMS coil placements around *target_coordinates*.

    The grid is created using SimNIBS' optimization utilities and written as a
    ``SearchGrid.msh`` and accompanying ``CoilPositions.geo`` visualization.

    Parameters
    ----------
    m2m_path : str
        Subject-specific SimNIBS ``m2m_*`` folder.
    output_folderpath : str
        Destination directory for grid artifacts.
    target_coordinates : array_like, shape (3,)
        XYZ coordinate of the target center (in SimNIBS coordinates).
    coil_scalp_distance : float
        Coil-to-scalp distance in millimeters.
    grid_radius : float
        Radius (mm) of the lateral search around the target.
    grid_resolution : float
        Spacing (mm) between candidate positions.
    angle_resolution : float
        Step (deg) between tested coil orientations.
    angle_limits : list[float, float]
        Min/max angles (deg) relative to reference orientation.

    Returns
    -------
    list[simnibs.MatSimnibs]
        List of candidate coil placements to be fed into OnlineFEM.
    """
    import os
    import shutil
    import glob
    import simnibs
    from simnibs import optimization
    import numpy as np

    grid_output_folder = output_folderpath
    os.makedirs(grid_output_folder, exist_ok=True)

    # Load brain mesh
    mesh_filepath = glob.glob(m2m_path + '/*.msh')[0]
    mesh = simnibs.read_msh(mesh_filepath)

    # Generate a small spherical target for visualization.
    target_region = optimization.tms_optimization.define_target_region(mesh=mesh,
                                                                       target_position=target_coordinates,
                                                                       target_radius=5,
                                                                       tags=[2])

    # Create grid of coil positions and orientations.
    search_grid = optimization.tms_optimization.get_opt_grid(mesh=mesh,
                                                             pos=target_coordinates,
                                                             distance=coil_scalp_distance,
                                                             radius=grid_radius,
                                                             resolution_pos=grid_resolution,
                                                             resolution_angle=angle_resolution,
                                                             angle_limits=angle_limits)

    # Write mesh with target and grid for visual inspection (Gmsh).
    output_mesh_filepath = os.path.join(grid_output_folder, 'SearchGrid.msh')
    target_mesh = np.zeros(mesh.elm.nr)
    target_mesh[target_region - 1] = 1
    mesh.add_element_field(target_mesh, 'TargetSphere')
    mesh.write(output_mesh_filepath)
    v = mesh.view(visible_tags=[5], visible_fields='all')
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
    """
    Combine two ``*.label.gii`` (LH/RH) files into a single LR label file.

    Parameters
    ----------
    gii_left, gii_right : str
        Paths to left/right hemisphere label files.

    Output
    ------
    ``<left>`` with suffix replaced to ``.LR.label.gii`` in the same directory.
    """
    import os
    import nibabel as nib
    import numpy as np

    gii_filepath = os.path.split(gii_left)[0]
    gii_filename = os.path.split(gii_left)[1].replace('.L.label.gii', '')

    left_gii = nib.load(gii_left)
    right_gii = nib.load(gii_right)

    left_data = left_gii.darrays[0].data
    right_data = right_gii.darrays[0].data

    combined_data = np.concatenate([left_data, right_data])

    combined_gii = nib.gifti.GiftiImage()
    combined_gii.add_gifti_data_array(nib.gifti.GiftiDataArray(data=combined_data, intent='NIFTI_INTENT_LABEL'))

    if left_gii.labeltable is not None:
        combined_gii.labeltable = left_gii.labeltable

    nib.save(combined_gii, gii_filepath + '/' + gii_filename + '.LR.label.gii')


def create_simnibs_roi(cifti_filepath, roi_id, roi_name, output_folderpath, m2m_path, surface_midthickness_left_filepath, surface_midthickness_right_filepath):
    """
    Convert a CIFTI label map into a SimNIBS RegionOfInterest and save a
    visualization inside ``SimNIBS_ROIs/<roi_name>``.

    Steps
    -----
    1) Split CIFTI ``.dlabel.nii`` into LH/RH GIFTI label maps.
    2) Combine the two into a single LR label map aligned to a combined surface.
    3) Convert LR label map into FreeSurfer ``.annot``.
    4) Create a SimNIBS ROI in subject space referencing the ``.annot`` file.

    Parameters
    ----------
    cifti_filepath : str
        Path to ``.dlabel.nii`` containing *roi_id* as a label.
    roi_id : int
        Label value to keep in the ROI.
    roi_name : str
        Name used for output folder and file prefixes.
    output_folderpath : str
        Output root directory.
    m2m_path : str
        Subject m2m path (SimNIBS).
    surface_midthickness_left_filepath, surface_midthickness_right_filepath : str
        Paths to subject-specific midthickness surfaces.

    Returns
    -------
    simnibs.utils.region_of_interest.RegionOfInterest
        The constructed ROI object.
    """
    import os
    import glob
    import simnibs

    # Combine midthickness surfaces.
    combine_gifti_surfaces(surface_midthickness_left_filepath,
                           surface_midthickness_right_filepath,
                           os.path.join(os.path.split(surface_midthickness_left_filepath)[0], 'combined_surface.surf.gii'))

    # Split dlabel into hemisphere label.gii files.
    os.system('wb_command -cifti-separate {} COLUMN -label CORTEX_LEFT {}'.format(
        cifti_filepath,
        os.path.split(cifti_filepath)[0] + '/' + os.path.split(cifti_filepath)[1].replace('.dlabel.nii', '.L.label.gii')))
    os.system('wb_command -cifti-separate {} COLUMN -label CORTEX_RIGHT {}'.format(
        cifti_filepath,
        os.path.split(cifti_filepath)[0] + '/' + os.path.split(cifti_filepath)[1].replace('.dlabel.nii', '.R.label.gii')))

    # Merge LH/RH labels into one LR label file.
    gii_left = os.path.split(cifti_filepath)[0] + '/' + os.path.split(cifti_filepath)[1].replace('.dlabel.nii', '.L.label.gii')
    gii_right = os.path.split(cifti_filepath)[0] + '/' + os.path.split(cifti_filepath)[1].replace('.dlabel.nii', '.R.label.gii')
    combine_label_files(gii_left, gii_right)

    # Convert LR label to FreeSurfer annot on the combined surface.
    gii_left_right = os.path.split(cifti_filepath)[0] + '/' + os.path.split(cifti_filepath)[1].replace('.dlabel.nii', '.LR.label.gii')
    annot_left_right = os.path.split(cifti_filepath)[0] + '/' + os.path.split(cifti_filepath)[1].replace('.dlabel.nii', '.LR.annot')
    
    gifti_label_to_fs_annot(gii_left_right,
                            os.path.join(os.path.split(surface_midthickness_left_filepath)[0], 'combined_surface.surf.gii'),
                            annot_left_right)
    

    # Create SimNIBS ROI referencing the subject's surfaces/annot.
    roi_settings_dict = {'method': 'surface',
                         'subpath': m2m_path,
                         'surface_type': 'custom',
                         'surface_path': os.path.join(os.path.split(surface_midthickness_left_filepath)[0], 'combined_surface.surf.gii'),
                         'mask_space': ['subject'],
                         'mask_path': [annot_left_right],
                         'mask_value': [roi_id]}

    roi = simnibs.utils.region_of_interest.RegionOfInterest(roi_settings_dict)
    
    # os.makedirs(os.path.join(output_folderpath, 'SimNIBS_ROIs', roi_name), exist_ok=True)
    # roi.write_visualization(os.path.join(output_folderpath, 'SimNIBS_ROIs', roi_name), 'simnibs_roi_' + roi_name)

    # Cleanup temporary intermediates.
    os.system('rm ' + gii_left)
    os.system('rm ' + gii_right)
    os.system('rm ' + gii_left_right)
    os.system('rm ' + annot_left_right)

    return roi


def run_simnibs_ofem(ofem, search_grid, position, didt):
    """
    Run an OnlineFEM update for a single coil *position* in *search_grid*.

    Parameters
    ----------
    ofem : simnibs.simulation.onlinefem.OnlineFEM
        Pre-configured OnlineFEM object.
    search_grid : list
        List of MatSimnibs coil placements.
    position : int
        Index in *search_grid*.
    didt : float
        Coil current change rate.

    Returns
    -------
    (MatSimnibs, list)
        Tuple of the selected placement and the OnlineFEM return structure.
    """
    from simnibs.simulation.onlinefem import OnlineFEM

    print('Simulating coil position ' + str(position+1) + ' of ' + str(len(search_grid)))

    e = ofem.update_field(matsimnibs=search_grid[position], didt=didt)

    return search_grid[position], e


def simnibs_accelerated_simulations_roi(search_grid, simnibs_roi_list, m2m_path, coil_filepath, didt, n_procs):
    """
    Compute E-field values over one or more SimNIBS ROIs for all placements in a grid.

    Parameters
    ----------
    search_grid : list
        Candidate placements returned by :func:`generate_search_grid`.
    simnibs_roi_list : list[simnibs.utils.region_of_interest.RegionOfInterest]
        One or more ROIs to score.
    m2m_path : str
        Subject m2m path (to load the head mesh).
    coil_filepath : str
        SimNIBS ``.ccd`` coil model file.
    didt : float
        Coil current change rate.
    n_procs : int
        (Currently unused; loop is serial for deterministic logging.)

    Returns
    -------
    list
        OnlineFEM output list (per placement).
    """
    import glob
    import numpy as np
    import simnibs
    from simnibs.simulation.onlinefem import OnlineFEM, FemTargetPointCloud
    from joblib import Parallel, delayed
    import time

    head_mesh = simnibs.read_msh(glob.glob(m2m_path + '/*.msh')[0])

    simnibs_roi_FemTargetPointCloud_list = []
    for simnibs_roi in simnibs_roi_list:
        simnibs_roi_nodes = simnibs_roi.get_nodes()
        simnibs_roi_FemTargetPointCloud_list.append(FemTargetPointCloud(mesh=head_mesh, center=simnibs_roi_nodes))

    ofem = OnlineFEM(mesh=glob.glob(m2m_path + '/*.msh')[0],
                     method="TMS",
                     fn_coil=coil_filepath,
                     roi=simnibs_roi_FemTargetPointCloud_list,
                     useElements=False,
                     dataType=[0]*len(simnibs_roi_FemTargetPointCloud_list))

    simulation_results = []
    start_time = time.time()
    for i in np.arange(len(search_grid)):
        print('Simulating coil position ' + str(i+1) + ' of ' + str(len(search_grid)))
        e = ofem.update_field(matsimnibs=search_grid[i], didt=didt)
        simulation_results.append(e)
    print("--- Simulations took %s seconds ---" % (time.time() - start_time))

    return simulation_results


def simnibs_accelerated_simulations_cortex(search_grid, coil_filepath, didt, m2m_path, output_folderpath, surface_midthickness_left_filepath, surface_midthickness_right_filepath):
    """
    Compute cortical E-field magnitudes for all placements and save as a dtseries.

    For each coil placement, the field over the entire cortex is sampled at
    surface nodes (via a FemTargetPointCloud) and then aligned to a 64984-long
    HCP cortex vector. Results are serialized to:
    - ``simulation_results.dtseries.nii`` (time = placement index)
    - ``simulation_results.pickle`` (tuple of *search_grid* and Numpy array)

    Parameters
    ----------
    search_grid : list
        Candidate placements (MatSimnibs).
    coil_filepath : str
        SimNIBS ``.ccd`` model.
    didt : float
        Coil current change rate.
    m2m_path : str
        Subject m2m folder.
    output_folderpath : str
        Where outputs will be written.
    surface_midthickness_left_filepath, surface_midthickness_right_filepath : str
        Subject midthickness surfaces (only for ROI creation).

    Returns
    -------
    numpy.ndarray, shape (N_positions, 64984)
        Electric field magnitude per vertex for each placement.
    """
    import os
    import glob
    import pickle
    import numpy as np
    import nibabel as nib
    import simnibs
    from simnibs.simulation.onlinefem import OnlineFEM, FemTargetPointCloud
    import time

    head_mesh = simnibs.read_msh(glob.glob(m2m_path + '/*.msh')[0])

    cifti_template_cortex_filepath = os.path.join(resources_folderpath, 'templates', 'CORTEX.32k_fs_LR.dscalar.nii')
    cifti_template_cortex = nib.load(cifti_template_cortex_filepath)
    cifti_template_cortex_values = cifti_template_cortex.get_fdata()[0]

    # ROI over whole cortex (sampling points).
    simnibs_roi_cortex = create_simnibs_roi(os.path.join(resources_folderpath, 'templates', 'CORTEX.32k_fs_LR.dlabel.nii'),
                                            1,
                                            'cortex',
                                            output_folderpath,
                                            m2m_path,
                                            surface_midthickness_left_filepath,
                                            surface_midthickness_right_filepath)

    simnibs_roi_cortex_nodes = simnibs_roi_cortex.get_nodes()
    simnibs_roi_cortex_FemTargetPointCloud = FemTargetPointCloud(mesh=head_mesh, center=simnibs_roi_cortex_nodes)

    ofem = OnlineFEM(mesh=head_mesh,
                     method="TMS",
                     fn_coil=coil_filepath,
                     roi=simnibs_roi_cortex_FemTargetPointCloud,
                     useElements=False,
                     dataType=[0])

    simulation_results = []
    start_time = time.time()
    for i in np.arange(len(search_grid)):
        print('Simulating coil position ' + str(i+1) + ' of ' + str(len(search_grid)))
        e = ofem.update_field(matsimnibs=search_grid[i], didt=didt)
        e = np.asarray(e[0][0]).flatten()
        e_final = cifti_template_cortex_values.copy()
        e_final[cifti_template_cortex_values == 1] = e
        simulation_results.append(e_final)
    print("--- Simulations took %s seconds ---" % (time.time() - start_time))

    simulation_results = np.asarray(simulation_results)

    # Persist as a dtseries for quick inspection in Workbench.
    import numpy as _np
    _np.savetxt(os.path.join(output_folderpath, 'simulation_results.txt'), simulation_results)
    os.system('wb_command -cifti-create-scalar-series {} {} -transpose'.format(
        os.path.join(output_folderpath, 'simulation_results.txt'),
        os.path.join(output_folderpath, 'simulation_results.dscalar.nii')))
    os.system('wb_command -cifti-change-mapping {} COLUMN {} -from-cifti {} COLUMN'.format(
        os.path.join(output_folderpath, 'simulation_results.dscalar.nii'),
        os.path.join(output_folderpath, 'simulation_results.dtseries.nii'),
        os.path.join(resources_folderpath, 'templates', 'HCP_blueprint.dtseries.nii')))
    os.system('rm ' + os.path.join(output_folderpath, 'simulation_results.txt'))
    os.system('rm ' + os.path.join(output_folderpath, 'simulation_results.dscalar.nii'))

    # Also pickle the grid + array to reload later.
    results = [search_grid, simulation_results]
    with open(os.path.join(output_folderpath, 'simulation_results.pickle'), "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    return simulation_results


def extract_best_coil_placements_magnitude(roi_filepath_list, simulation_results, search_grid, n_placements):
    """
    Rank coil placements by total E-field in target ROI while penalizing
    non-target cortex, and return the top ``n_placements`` placements.

    Two modes:
    - Single entry in ``roi_filepath_list``: treat non-target as complement.
    - Two entries: ``[target_roi, avoidance_roi]`` explicitly provided.

    Scoring
    -------
    ``score = 2/3 * z(E_target) + 1/3 * (1 - z(E_avoidance))``

    Parameters
    ----------
    roi_filepath_list : list[str]
        One or two CIFTI paths.
    simulation_results : numpy.ndarray, shape (N_positions, 64984)
        Output from :func:`simnibs_accelerated_simulations_cortex`.
    search_grid : list
        Placements corresponding to rows in *simulation_results*.
    n_placements : int
        Number of top placements to return.

    Returns
    -------
    list
        Top placements (MatSimnibs objects) ordered by score.
    """
    import numpy as np

    if len(roi_filepath_list) == 1:
        roi_target_values = load_cifti_values(roi_filepath_list[0])
        roi_target_values[roi_target_values != 0] = 1  # binarize

        roi_avoidance_values = roi_target_values.copy()
        roi_avoidance_values[roi_target_values != 1] = 1
        roi_avoidance_values[roi_target_values == 1] = 0

        e_roi_target = np.matmul(simulation_results, roi_target_values)
        e_roi_target_norm = (e_roi_target - min(e_roi_target)) / (max(e_roi_target) - min(e_roi_target))

        e_roi_avoidance = np.matmul(simulation_results, roi_avoidance_values)
        e_roi_avoidance_norm = (e_roi_avoidance - min(e_roi_avoidance)) / (max(e_roi_avoidance) - min(e_roi_avoidance))
        e_roi_avoidance_norm_inv = 1 - e_roi_avoidance_norm

        e_final = 2/3*e_roi_target_norm + 1/3*e_roi_avoidance_norm_inv

    if len(roi_filepath_list) == 2:
        roi_target_values = load_cifti_values(roi_filepath_list[0])
        roi_target_values[roi_target_values != 0] = 1

        roi_avoidance_values = load_cifti_values(roi_filepath_list[1])
        roi_avoidance_values[roi_avoidance_values != 0] = 1

        e_roi_target = np.matmul(simulation_results, roi_target_values)
        e_roi_target_norm = (e_roi_target - min(e_roi_target)) / (max(e_roi_target) - min(e_roi_target))

        e_roi_avoidance = np.matmul(simulation_results, roi_avoidance_values)
        e_roi_avoidance_norm = (e_roi_avoidance - min(e_roi_avoidance)) / (max(e_roi_avoidance) - min(e_roi_avoidance))
        e_roi_avoidance_norm_inv = 1 - e_roi_avoidance_norm

        e_final = 2/3*e_roi_target_norm + 1/3*e_roi_avoidance_norm_inv

    best_coil_placements_indices = (-np.array(e_final)).argsort()[:n_placements]
    best_coil_placements = [search_grid[i] for i in best_coil_placements_indices]

    return best_coil_placements


def extract_best_coil_placements_hotspot(simulation_results, search_grid, hotspot_percentiles, FCmap_filepath, target_id_list, avoidance_id_list, n_placements, surface_midthickness_left_filepath, surface_midthickness_right_filepath):
    """
    Rank placements by how much of the high-E-field *hotspot* falls within
    target vs. avoidance networks defined by a parcellation.

    For each placement and each percentile p in *hotspot_percentiles*:
    - threshold the cortex E-field at p%
    - compute the fraction of hotspot area inside *target* labels and inside
      *avoidance* labels (or the complement if ``avoidance_id_list`` empty)
    - average fractions across percentiles
    - score = ``2/3 * mean(target_frac) - 1/3 * mean(avoidance_frac)``

    Parameters
    ----------
    simulation_results : numpy.ndarray, shape (N_positions, 64984)
        Per-vertex E-field magnitudes.
    search_grid : list
        Coil placements.
    hotspot_percentiles : array_like
        Percentiles (e.g., 99.0..99.8) used to define hotspots.
    FCmap_filepath : str
        Parcellation CIFTI where integers encode network IDs.
    target_id_list : list[int]
        Label IDs considered "target".
    avoidance_id_list : list[int] or []
        Optional label IDs considered "avoidance"; if empty, uses complement.
    n_placements : int
        How many top placements to return.
    surface_midthickness_left_filepath, surface_midthickness_right_filepath : str
        For area computations (vertex area per hemisphere).

    Returns
    -------
    list
        Top placements maximizing hotspot mass in targets while minimizing in avoidance.
    """
    import numpy as np

    FCmap_values = load_cifti_values(FCmap_filepath)

    vertex_area_values_LH = get_surface_vertex_areas(surface_midthickness_left_filepath)
    vertex_area_values_RH = get_surface_vertex_areas(surface_midthickness_right_filepath)
    vertex_area_values = np.concatenate([vertex_area_values_LH, vertex_area_values_RH])

    target_indices = get_indices(FCmap_values, target_id_list)
    FCmap_values_target = np.zeros(len(FCmap_values))
    FCmap_values_target[target_indices] = 1

    if not avoidance_id_list:
        FCmap_values_avoidance = np.zeros(len(FCmap_values))
        FCmap_values_avoidance[np.setdiff1d(np.arange(len(FCmap_values_avoidance)), target_indices)] = 1
    else:
        avoidance_indices = get_indices(FCmap_values, avoidance_id_list)
        FCmap_values_avoidance = np.zeros(len(FCmap_values))
        FCmap_values_avoidance[avoidance_indices] = 1

    simulation_hotspots = np.zeros([simulation_results.shape[0], simulation_results.shape[1], len(hotspot_percentiles)])
    FCmap_hotspots = np.zeros([simulation_results.shape[0], simulation_results.shape[1], len(hotspot_percentiles)])

    vertex_area_hotspots = np.zeros([simulation_results.shape[0], 2, len(hotspot_percentiles)])

    for i in np.arange(simulation_results.shape[0]):
        for j in np.arange(len(hotspot_percentiles)):
            hotspot_tmp = simulation_results[i, :].copy()
            percentile_value = np.percentile(hotspot_tmp, hotspot_percentiles[j])
            hotspot_tmp[hotspot_tmp < percentile_value] = 0
            hotspot_tmp[hotspot_tmp >= percentile_value] = 1

            simulation_hotspots[i, :, j] = hotspot_tmp
            FCmap_hotspots[i, :, j] = np.multiply(FCmap_values, hotspot_tmp)

            vertex_area_hotspot = np.sum(vertex_area_values[hotspot_tmp == 1])
            vertex_area_target = np.sum(vertex_area_values[(hotspot_tmp == 1) & (FCmap_values_target == 1)])
            vertex_area_avoidance = np.sum(vertex_area_values[(hotspot_tmp == 1) & (FCmap_values_avoidance == 1)])

            vertex_area_target_perc = vertex_area_target / vertex_area_hotspot
            vertex_area_avoidance_perc = vertex_area_avoidance / vertex_area_hotspot

            vertex_area_hotspots[i, 0, j] = vertex_area_target_perc
            vertex_area_hotspots[i, 1, j] = vertex_area_avoidance_perc

    vertex_area_hotpots_average = np.zeros([simulation_results.shape[0], 2])
    vertex_area_hotpots_average[:, 0] = np.asarray([np.mean(vertex_area_hotspots[i, 0, :]) for i in np.arange(simulation_results.shape[0])])
    vertex_area_hotpots_average[:, 1] = np.asarray([np.mean(vertex_area_hotspots[i, 1, :]) for i in np.arange(simulation_results.shape[0])])

    vertex_area_hotpots_average_norm = np.zeros([simulation_results.shape[0], 2])
    vertex_area_hotpots_average_norm[:, 0] = (vertex_area_hotpots_average[:, 0] - np.min(vertex_area_hotpots_average[:, 0])) / (np.max(vertex_area_hotpots_average[:, 0]) - np.min(vertex_area_hotpots_average[:, 0]))
    vertex_area_hotpots_average_norm[:, 1] = (vertex_area_hotpots_average[:, 1] - np.min(vertex_area_hotpots_average[:, 1])) / (np.max(vertex_area_hotpots_average[:, 1]) - np.min(vertex_area_hotpots_average[:, 1]))

    vertex_area_hotpots_average_norm_final = 2/3 * vertex_area_hotpots_average_norm[:, 0] - 1/3 * vertex_area_hotpots_average_norm[:, 1]

    best_coil_placements_indices = (-np.array(vertex_area_hotpots_average_norm_final)).argsort()[:n_placements]
    best_coil_placements = [search_grid[i] for i in best_coil_placements_indices]

    return best_coil_placements


def run_final_simulations_accelerated(coil_placements_list, coil_filepath, didt, m2m_path, output_folderpath, surface_midthickness_left_filepath, surface_midthickness_right_filepath):
    """
    Compute high-resolution final E-fields for one or more fixed placements,
    writing each result as a ``Efield.dscalar.nii`` under a separate folder.

    Parameters
    ----------
    coil_placements_list : list
        Placements (MatSimnibs) to evaluate.
    coil_filepath : str
        SimNIBS ``.ccd`` model path.
    didt : float
        Coil current change rate.
    m2m_path : str
        Subject m2m path.
    output_folderpath : str
        Root output directory.
    surface_midthickness_left_filepath, surface_midthickness_right_filepath : str
        Subject surfaces for ROI creation.
    """
    import os
    import glob
    import numpy as np
    import nibabel as nib
    import simnibs
    from simnibs.simulation.onlinefem import OnlineFEM, FemTargetPointCloud

    os.makedirs(os.path.join(output_folderpath, 'SimNIBS_simulations'), exist_ok=True)

    head_mesh = simnibs.read_msh(glob.glob(m2m_path + '/*.msh')[0])

    cifti_template_cortex_filepath = os.path.join(resources_folderpath, 'templates', 'CORTEX.32k_fs_LR.dscalar.nii')
    cifti_template_cortex = nib.load(cifti_template_cortex_filepath)
    cifti_template_cortex_values = cifti_template_cortex.get_fdata()[0]

    simnibs_roi_cortex = create_simnibs_roi(os.path.join(resources_folderpath, 'templates', 'CORTEX.32k_fs_LR.dlabel.nii'),
                                            1,
                                            'cortex',
                                            output_folderpath,
                                            m2m_path,
                                            surface_midthickness_left_filepath,
                                            surface_midthickness_right_filepath)

    simnibs_roi_cortex_nodes = simnibs_roi_cortex.get_nodes()
    simnibs_roi_cortex_FemTargetPointCloud = FemTargetPointCloud(mesh=head_mesh, center=simnibs_roi_cortex_nodes)

    ofem = OnlineFEM(mesh=head_mesh,
                     method="TMS",
                     fn_coil=coil_filepath,
                     roi=simnibs_roi_cortex_FemTargetPointCloud,
                     useElements=False,
                     dataType=[0])

    for i in np.arange(len(coil_placements_list)):
        e = ofem.update_field(matsimnibs=coil_placements_list[i], didt=didt)
        e = np.asarray(e[0][0]).flatten()

        e_final = cifti_template_cortex_values.copy()
        e_final[cifti_template_cortex_values == 1] = e

        os.makedirs(os.path.join(output_folderpath, 'SimNIBS_simulations', 'CoilPlacement' + str(i)), exist_ok=True)

        array_to_cifti_dscalar(e_final,
                               cifti_template_cortex_filepath,
                               os.path.join(output_folderpath, 'SimNIBS_simulations', 'CoilPlacement' + str(i), 'Efield.dscalar.nii'))


def load_simulation_results(simulation_results_filepath):
    """
    Reload a ``simulation_results.pickle`` into ``(search_grid, results_array)``.

    Parameters
    ----------
    simulation_results_filepath : str
        Path to the pickle created by :func:`simnibs_accelerated_simulations_cortex`.

    Returns
    -------
    tuple
        (search_grid, simulation_results_array)
    """
    import pickle

    with open(simulation_results_filepath, 'rb') as read_obj:
        simulation_results = pickle.load(read_obj)

    search_grid = simulation_results[0]
    simulation_results = simulation_results[1]

    return search_grid, simulation_results


def convert_simnibs_to_localite(TMScoilPositionMatrix, PositionName, PositionIndex, OutputFilepath):
    """
    Append two Localite-compatible InstrumentMarker XML elements (normal and
    inverted orientation) for a given SimNIBS coil placement matrix.

    The conversion applies coordinate-system rotations to match Localite's
    conventions for MR-compatible coils.

    Parameters
    ----------
    TMScoilPositionMatrix : numpy.ndarray, shape (4, 4)
        Homogeneous transformation (SimNIBS matsimnibs).
    PositionName : str
        Human-readable name for the marker (e.g., condition or distance).
    PositionIndex : int
        Index used in the XML attributes (will also write an _inv element with index+1).
    OutputFilepath : str
        Path to an XML-like text file (content is appended).
    """
    import numpy as np

    # Predefined rotations
    z_rot = np.array([[round(np.cos(np.pi)), round(np.sin(np.pi)), 0.0, 0.0],
                      [round(-np.sin(np.pi)), round(np.cos(np.pi)), 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])

    y_rot = np.array([[round(np.cos(0.5*np.pi)), 0.0, round(-np.sin(0.5*np.pi)), 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [round(np.sin(0.5*np.pi)), 0.0, round(np.cos(0.5*np.pi)), 0.0],
                      [0.0, 0.0, 0.0, 1.0]])

    full_rot = np.matmul(z_rot, y_rot)

    x_rot = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, round(np.cos(np.pi)), round(np.sin(np.pi)), 0.0],
                      [0.0, round(-np.sin(np.pi)), round(np.cos(np.pi)), 0.0],
                      [0.0, 0.0, 0.0, 1.0]])

    # Transform into Localite coordinates (normal + inverted x-rotation variant).
    Position_tp = np.transpose(TMScoilPositionMatrix)
    Position_Localite_tp = np.matmul(full_rot, Position_tp)
    Position_Localite = np.transpose(Position_Localite_tp)

    Position_Localite_inv_tp = np.matmul(x_rot, Position_Localite_tp)
    Position_Localite_inv = np.transpose(Position_Localite_inv_tp)

    # Write two InstrumentMarker XML blocks.
    xml = '    <Element index="' + str(PositionIndex) + '" selected="true" type="InstrumentMarker">\n\
            <InstrumentMarker additionalInformation="" alwaysVisible="false"\n\
                color="#0000ff" description="' + PositionName + '" locked="false" set="true" uid="' + str(PositionIndex) + '">\n\
                <Matrix4D data00="' + str(round(Position_Localite[0,0],6)) + '" data01="' + str(round(Position_Localite[0,1],6)) + '" data02="' + str(round(Position_Localite[0,2],6)) + '"\n\
                    data03="' + str(round(Position_Localite[0,3],6)) + '" data10="' + str(round(Position_Localite[1,0],6)) + '" data11="' + str(round(Position_Localite[1,1],6)) + '"\n\
                    data12="' + str(round(Position_Localite[1,2],6)) + '" data13="' + str(round(Position_Localite[1,3],6)) + '" data20="' + str(round(Position_Localite[2,0],6)) + '"\n\
                    data21="' + str(round(Position_Localite[2,1],6)) + '" data22="' + str(round(Position_Localite[2,2],6)) + '" data23="' + str(round(Position_Localite[2,3],6)) + '" data30="0.0"\n\
                    data31="0.0" data32="0.0" data33="1.0"/>\n\
            </InstrumentMarker>\n\
        </Element>\n\
            <Element index="' + str(PositionIndex+1) + '" selected="true" type="InstrumentMarker">\n\
            <InstrumentMarker additionalInformation="" alwaysVisible="false"\n\
                color="#0000ff" description="' + PositionName + '_inv" locked="false" set="true" uid="' + str(PositionIndex+1) + '">\n\
                <Matrix4D data00="' + str(round(Position_Localite_inv[0,0],6)) + '" data01="' + str(round(Position_Localite_inv[0,1],6)) + '" data02="' + str(round(Position_Localite_inv[0,2],6)) + '"\n\
                    data03="' + str(round(Position_Localite_inv[0,3],6)) + '" data10="' + str(round(Position_Localite_inv[1,0],6)) + '" data11="' + str(round(Position_Localite_inv[1,1],6)) + '"\n\
                    data12="' + str(round(Position_Localite_inv[1,2],6)) + '" data13="' + str(round(Position_Localite_inv[1,3],6)) + '" data20="' + str(round(Position_Localite_inv[2,0],6)) + '"\n\
                    data21="' + str(round(Position_Localite_inv[2,1],6)) + '" data22="' + str(round(Position_Localite_inv[2,2],6)) + '" data23="' + str(round(Position_Localite_inv[2,3],6)) + '" data30="0.0"\n\
                    data31="0.0" data32="0.0" data33="1.0"/>\n\
            </InstrumentMarker>\n\
        </Element>\n\
    '

    with open(OutputFilepath, 'a') as f:
        f.write(xml)


def run_final_simulation(coil_placement_name, coil_placement_list, m2m_path, coil_filepath, coil_scalp_distance, didt, output_folderpath, surface_midthickness_left_filepath, surface_midthickness_right_filepath):
    """
    Run full SimNIBS FEM simulations for the best placement(s), export placements
    at distances {d, d+1, ..., d+9} mm (along coil normal), write Localite XML
    markers for each, and compute a compact dtseries per distance.

    Parameters
    ----------
    coil_placement_name : str
        Name prefix used for outputs.
    coil_placement_list : list
        Best placements (MatSimnibs) to evaluate.
    m2m_path : str
        Subject m2m directory.
    coil_filepath : str
        ``.ccd`` model path.
    coil_scalp_distance : float
        Base coil–scalp distance (mm).
    didt : float
        Current change rate.
    output_folderpath : str
        Output root folder.
    surface_midthickness_left_filepath, surface_midthickness_right_filepath : str
        Subject-specific midthickness surfaces for mapping.

    Outputs
    -------
    - ``SimNIBS/Simulations/CoilPlacement*/CoilPlacement.txt`` placement matrices
    - Localite XML snippets with normal and inverted orientations
    - ``simulation_results.dtseries.nii`` per placement-distance set
    """
    import os
    import glob
    import shutil
    import numpy as np
    from simnibs import sim_struct, run_simnibs

    os.makedirs(output_folderpath, exist_ok=True)

    for i in np.arange(len(coil_placement_list)):

        simulation_output_folderpath = os.path.join(output_folderpath, 'CoilPlacement' + str(i+1))

        if os.path.exists(simulation_output_folderpath):
            shutil.rmtree(simulation_output_folderpath)

        os.makedirs(simulation_output_folderpath)

        coil_placement = coil_placement_list[i]

        s = sim_struct.SESSION()
        s.fnamehead = glob.glob(m2m_path + '/*.msh')[0]
        s.pathfem = simulation_output_folderpath
        s.fields = 'e'
        s.open_in_gmsh = True
        s.map_to_surf = False
        s.map_to_fsavg = False
        s.map_to_vol = False
        s.map_to_mni = False
        s.subpath = m2m_path
        tmslist = s.add_tmslist()
        tmslist.fnamecoil = coil_filepath
        pos = tmslist.add_position()
        pos.didt = didt
        pos.matsimnibs = coil_placement
        run_simnibs(s)

        # Persist the exact placement used.
        np.savetxt(simulation_output_folderpath + '/CoilPlacement.txt', coil_placement)

        coil_placement_distances = []

        for j in np.arange(10):
            coil_placement_tmp = coil_placement.copy()
            coil_placement_tmp[0:3, 3] = coil_placement_tmp[0:3, 3] - j * coil_placement_tmp[0:3, 2]
            coil_placement_distances.append(coil_placement_tmp)

            # Localite XML markers for d, d+1, ..., d+9 mm
            convert_simnibs_to_localite(coil_placement_tmp,
                                        coil_placement_name + '_' + str((coil_scalp_distance + j)) + 'mm',
                                        (0 + j*2),
                                        simulation_output_folderpath + '/' + coil_placement_name + '_OptimalPosition' + str(i+1) + '_Localite_XML.txt')

        # For each distance, compute cortex E-field and output as dtseries.
        simnibs_accelerated_simulations_cortex(coil_placement_distances,
                                               coil_filepath,
                                               didt,
                                               m2m_path,
                                               simulation_output_folderpath,
                                               surface_midthickness_left_filepath,
                                               surface_midthickness_right_filepath)


def fast_pfm(timeseries_filepath, parcellation, output_folderpath, surface_midthickness_left_filepath, surface_midthickness_right_filepath):
    """
    Fast precision functional mapping (PFM) using published priors.

    Given a vertexwise (dtseries) timeseries, project it into prior network
    timecourses, z-score both spaces, compute weights (correlations), and assign
    each vertex to its most likely network. Post-process with cluster cleanup
    and dilation to remove holes and re-serialize as a clean ``.dlabel.nii``.

    Parameters
    ----------
    timeseries_filepath : str
        Input CIFTI dtseries path (surface-aligned; 32k_fs_LR).
    parcellation : {'Lynch2024', 'Hermosillo2024'}
        Choice of priors/labels.
    output_folderpath : str
        Output directory.
    surface_midthickness_left_filepath, surface_midthickness_right_filepath : str
        Surfaces used by Workbench for cluster/dilation ops.

    Outputs
    -------
    - ``PFM_<parcellation>priors.dlabel.nii`` (final network map)
    - Intermediate files are removed.
    """
    import os
    import numpy as np
    import pickle

    if parcellation == 'Lynch2024':
        priors_filepath = os.path.join(FASTANS_installation_folderpath, 'resources', 'PFM', 'priors', 'Lynch2024', 'Lynch2024_priors.pickle')
        labels_filepath = os.path.join(FASTANS_installation_folderpath, 'resources', 'PFM', 'priors', 'Lynch2024', 'Lynch2024_LabelList.txt')
    elif parcellation == 'Hermosillo2024':
        priors_filepath = os.path.join(FASTANS_installation_folderpath, 'resources', 'PFM', 'priors', 'Hermosillo2024', 'Hermosillo2024_priors.pickle')
        labels_filepath = os.path.join(FASTANS_installation_folderpath, 'resources', 'PFM', 'priors', 'Hermosillo2024', 'Hermosillo2024_LabelList.txt')

    cifti_template_cortex = load_cifti_values(os.path.join(resources_folderpath, 'templates', 'CORTEX.32k_fs_LR.dscalar.nii'))

    with open(priors_filepath, 'rb') as read_obj:
        priors = pickle.load(read_obj)

    timeseries = load_cifti_values(timeseries_filepath)
    timeseries_means = timeseries.mean(axis=0, keepdims=True)
    timeseries_stds = timeseries.std(axis=0, keepdims=True)
    timeseries_standardized = (timeseries - timeseries_means) / timeseries_stds

    prior_timeseries = np.matmul(priors, np.transpose(timeseries))
    prior_timeseries_means = prior_timeseries.mean(axis=1, keepdims=True)
    prior_timeseries_stds = prior_timeseries.std(axis=1, keepdims=True)
    prior_timeseries_standardized = (prior_timeseries - prior_timeseries_means) / prior_timeseries_stds

    network_weights = np.matmul(prior_timeseries_standardized, timeseries_standardized) / (len(timeseries_standardized) - 1)

    network_labels = np.argmax(network_weights, axis=0)
    network_labels = network_labels + 1

    array_to_cifti_dscalar(network_labels,
                           os.path.join(resources_folderpath, 'templates', 'CORTEX.32k_fs_LR.dscalar.nii'),
                           os.path.join(output_folderpath, 'PFM_' + parcellation + 'priors.dscalar.nii'))

    os.system('wb_command -cifti-label-import {} {} {}'.format(
                os.path.join(output_folderpath, 'PFM_' + parcellation + 'priors.dscalar.nii'),
                labels_filepath,
                os.path.join(output_folderpath, 'PFM_' + parcellation + 'priors.dlabel.nii')))

    pfm_data = load_cifti_values(os.path.join(output_folderpath, 'PFM_' + parcellation + 'priors.dlabel.nii'))
    pfm_labels = np.unique(pfm_data)
    pfm_data_new = np.zeros(len(pfm_data))

    for i in pfm_labels:
        pfm_data_tmp = pfm_data.copy()
        pfm_data_tmp[pfm_data != i] = 0

        array_to_cifti_dscalar(pfm_data_tmp,
                               os.path.join(resources_folderpath, 'templates', 'CORTEX.32k_fs_LR.dscalar.nii'),
                               os.path.join(output_folderpath, 'PFM_' + parcellation + 'priors_network' + str(i) + '.dscalar.nii'))

        os.system('wb_command -cifti-find-clusters {} 0 50 0 50 COLUMN {} -left-surface {} -right-surface {} -merged-volume'.format(
            os.path.join(output_folderpath, 'PFM_' + parcellation + 'priors_network' + str(i) + '.dscalar.nii'),
            os.path.join(output_folderpath, 'PFM_' + parcellation + 'priors_network' + str(i) + '.dscalar.nii'),
            surface_midthickness_left_filepath,
            surface_midthickness_right_filepath))

        network_data = load_cifti_values(os.path.join(output_folderpath, 'PFM_' + parcellation + 'priors_network' + str(i) + '.dscalar.nii'))
        network_data[network_data != 0] = i
        pfm_data_new = pfm_data_new + network_data

        os.system('rm ' + os.path.join(output_folderpath, 'PFM_' + parcellation + 'priors_network' + str(i) + '.dscalar.nii'))

    array_to_cifti_dscalar(pfm_data_new,
                           os.path.join(resources_folderpath, 'templates', 'CORTEX.32k_fs_LR.dscalar.nii'),
                           os.path.join(output_folderpath, 'PFM_' + parcellation + 'priors.dscalar.nii'))

    os.system('wb_command -cifti-label-import {} {} {}'.format(
                os.path.join(output_folderpath, 'PFM_' + parcellation + 'priors.dscalar.nii'),
                labels_filepath,
                os.path.join(output_folderpath, 'PFM_' + parcellation + 'priors.dlabel.nii')))

    os.system('wb_command -cifti-dilate {} COLUMN 50 50 -left-surface {} -right-surface {} {} -nearest'.format(
                os.path.join(output_folderpath, 'PFM_' + parcellation + 'priors.dlabel.nii'),
                surface_midthickness_left_filepath,
                surface_midthickness_right_filepath,
                os.path.join(output_folderpath, 'PFM_' + parcellation + 'priors.dlabel.nii')))

    pfm_data_new = load_cifti_values(os.path.join(output_folderpath, 'PFM_' + parcellation + 'priors.dlabel.nii'))
    pfm_data_new[cifti_template_cortex == 0] = 0

    array_to_cifti_dscalar(pfm_data_new,
                           os.path.join(resources_folderpath, 'templates', 'CORTEX.32k_fs_LR.dscalar.nii'),
                           os.path.join(output_folderpath, 'PFM_' + parcellation + 'priors.dscalar.nii'))

    os.system('wb_command -cifti-label-import {} {} {}'.format(
                os.path.join(output_folderpath, 'PFM_' + parcellation + 'priors.dscalar.nii'),
                labels_filepath,
                os.path.join(output_folderpath, 'PFM_' + parcellation + 'priors.dlabel.nii')))

    os.system('rm ' + os.path.join(output_folderpath, 'PFM_' + parcellation + 'priors.dscalar.nii'))
