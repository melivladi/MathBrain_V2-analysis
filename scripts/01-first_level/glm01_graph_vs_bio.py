# #! /usr/bin/env python
# coding: UTF-8

## Run First Level Analysis in native T1w space

#%%
## Imports
import os
from os import getcwd, mkdir, path
import pandas as pd
import nibabel as nib
import numpy as np
import nilearn
import nilearn.glm
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import threshold_img
from nilearn.glm import threshold_stats_img
from nilearn.regions import connected_regions
from nilearn.plotting import plot_design_matrix

#%%
# Define subject and file paths
SUBJ_ID = 201

## Directories
PROJECT_ROOT = "/data/mvladisauskas/MathBrain_V2"
EVENT_PATH = f"{PROJECT_ROOT}/results/00-events"
FUNC_PATH = f"{PROJECT_ROOT}/derivatives/fmriprep/sub-{SUBJ_ID:02}/func"
OUTPUT_DIR = f"{PROJECT_ROOT}/results/01-first_level/native_glm1"

#Files
func_file = os.path.join(FUNC_PATH, f"sub-{SUBJ_ID:02}_task-graphs_dir-ap_space-T1w_desc-preproc_bold.nii.gz")

#%%
## Create a design matrix with events and confounds

# Load event file
task_file = os.path.join(EVENT_PATH, f"sub{SUBJ_ID:02}_graphs_events.tsv")
events = pd.read_csv(task_file, sep="\t")

# Load movement parameters (confounds)
movement_file = path.join(FUNC_PATH, f'sub-{SUBJ_ID:02}_task-graphs_dir-ap_desc-confounds_timeseries.tsv')
movement = pd.read_csv(movement_file, sep="\t")

# 6 motion parameters + csf + white matter + ask a_comp_cor to Bosco
confounds = movement[['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 
                    "a_comp_cor_00", "a_comp_cor_01", "a_comp_cor_02", "a_comp_cor_03", 
                    "a_comp_cor_04", "a_comp_cor_05", "a_comp_cor_06", "a_comp_cor_07",
                    "a_comp_cor_08", "a_comp_cor_09", "csf", "white_matter"
                        ]]

confounds.reset_index(drop=True, inplace=True)

# Get frame times (when each frame, a.k.a. volume, is collected)
TR = 2
frame_times = np.arange(confounds.shape[0])*TR

# Make design matrix
design_matrix = make_first_level_design_matrix(frame_times,
                                                events, 
                                                drift_model='cosine',
                                                high_pass=1/128,
                                                add_regs=confounds,
                                                add_reg_names=[name for name in confounds.columns])


#%%
# Fitting the GLM
print("Fitting GLM")

fmri_glm = FirstLevelModel(t_r=TR,
                            hrf_model="spm",
                            slice_time_ref=0.5,
                            smoothing_fwhm=2.4)


fmri_glm = fmri_glm.fit(func_file, design_matrices=design_matrix)

print("Done!")


#%%
# Define contrasts

# Get the names of all the regressors in the design matrix
regressor_names = design_matrix.columns

# Count regressors in the design matrix
n_regressors = len(regressor_names)

# Create identity matrix for vector operations
identity_matrix = np.eye( n_regressors )

# Create a dictionary with the contrast vector for each regressor
base_contrasts = {name: identity_matrix[i,:] for i, name in enumerate(regressor_names)}

# Define Graphs > Biology contrast
graphs_vs_bio = base_contrasts['math']  - base_contrasts['bio'] 

# Storing the contrasts vectors
contrast_vectors = [graphs_vs_bio]

# Names corresponding to each contrast vector
contrast_names = ["Graphs_versus_Biology"]

#%%
# Compute and save statistical maps

# Voxelwise uncorrected p-value thresholds
p_thresholds = [0.05, 0.01, 0.001]

# Loop over the indices of the contrast list
for contrast_index in range(len(contrast_vectors)):

        # Retrieve the contrast vector at this index
        contrast_weights = contrast_vectors[contrast_index]
        
        # Retrieve its corresponding name
        contrast_name = contrast_names[contrast_index]

        print(f'Current contrast is {contrast_name}')
        print('Computing zmap...')

        # Generate a thresholded statistical map for each p-value
        for p_threshold in p_thresholds:

            # Compute the signed z-statistic map for this contrast
            z_map = fmri_glm.compute_contrast(contrast_weights, output_type='z_score')

            # Use statistical thresholding to the z-map (Nilearns default)
            threshold_result = nilearn.glm.threshold_stats_img(z_map, alpha=p_threshold)

            # Extract the thresholded image from the tuple
            thresholded_z_map = threshold_result[0]

            # Save it
            output_file = os.path.join(OUTPUT_DIR, f"sub-{SUBJ_ID:02}_z-map-{contrast_name}_graphs_T1w-native_p-{p_threshold}.nii.gz")
            nib.save(thresholded_z_map, output_file)


# %%
