#! /usr/bin/env python
# coding: UTF-8

## Run First Level Analysis in the MNI SPACE

## Import
import os
from os import getcwd, mkdir, path

import pandas as pd
import nibabel as nib
import numpy as np

import nilearn
import nilearn.image
import nilearn.plotting
import nilearn.glm
import nilearn.glm.first_level
import nilearn.reporting
from matplotlib import pyplot as plt
import matplotlib as mpl
from nilearn import plotting
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from nilearn.image import threshold_img
from nilearn.glm import threshold_stats_img
from nilearn import datasets
from nilearn import surface
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from nilearn import plotting, image
from matplotlib import cm
from nilearn.regions import connected_regions
#from nilearn.interfaces.bids import save_glm_to_bids
from nilearn.plotting import plot_design_matrix

# Comment if running interactively
import sys
import optparse

#%%
#TO GET SUBJECT NUMBER

#parser = optparse.OptionParser()

#parser.add_option('-s', '--idSubject',
 #                 dest='subjectId',
  #               type='int',
   #              help='Subject id (order of arrival)')


#(options, args) = parser.parse_args()

#if options.subjectId is None:
 #   parser.print_help()
  #  print("\nThe -s argument (subject's number and number of runs) is mandatory")
   # sys.exit(1)

#subject_number  = options.subjectId
subject_number  = 201

print(f"subject number is:", subject_number)


## Directories
EVENT_PATH = "/neurospin/unicog/protocols/Daniela_MathBrain_V2_fmri/Results/pilot_graphs/events"
FUNC_PATH = f"/neurospin/unicog/protocols/MathBrain_Dehaene_2024/Data/MathBrain_7T_V2/Participants/derivatives/fmri_prep/sub-{subject_number:02}/func"
OUTPUT_DIR = "/neurospin/unicog/protocols/Daniela_MathBrain_V2_fmri/Results/pilot_graphs/first_level_analyses/native"

#Files
func_file = os.path.join(FUNC_PATH, f"sub-{subject_number:02}_task-graphs_dir-ap_space-T1w_desc-preproc_bold.nii.gz")
print(func_file)



#%%

## To create a design matrix with events and confounds

# load event file
task_file = os.path.join(EVENT_PATH, f"sub{subject_number:02}_graphs_events.tsv")
events = pd.read_csv(task_file)


# load movement parameters
movement_file = path.join(FUNC_PATH, f'sub-{subject_number:02}_task-graphs_dir-ap_desc-confounds_timeseries.tsv')
movement = pd.read_csv(movement_file, sep='\s+')

# 6 motion parameters + csf + white matter + ask a_comp_cor to Bosco
confounds = movement[['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 
                    "a_comp_cor_00", "a_comp_cor_01", "a_comp_cor_02", "a_comp_cor_03", 
                    "a_comp_cor_04", "a_comp_cor_05", "a_comp_cor_06", "a_comp_cor_07",
                    "a_comp_cor_08", "a_comp_cor_09", "csf", "white_matter"
                        ]]


confounds.reset_index(drop=True, inplace=True)

# get frame times (when each frame, a.k.a. volume, is collected)
TR = 2
frame_times = np.arange(confounds.shape[0])*TR

# make design matrix
design_matrix = make_first_level_design_matrix(frame_times,
                                                events, 
                                                drift_model='cosine',
                                                high_pass=1/128,
                                                add_regs=confounds,
                                                add_reg_names=[name for name in confounds.columns])


#%%
print("Fitting GLM")

fmri_glm = FirstLevelModel(t_r=TR,
                            hrf_model="spm",
                            slice_time_ref=0.5,
                            smoothing_fwhm=2.4)


fmri_glm = fmri_glm.fit(func_file, design_matrices=design_matrix)

print("Done!")


#%%
#Get contrasts 


conds_loca = design_matrix.columns
#print(conds_loca)
mat_size = len(conds_loca)
Id = np.eye( mat_size )

contrasts_Base_loca = {cond : Id[i,:] for i, cond in enumerate(conds_loca)}

Math_bio = contrasts_Base_loca['math']  - contrasts_Base_loca['bio'] 
#Others_math = contrasts_Base_loca['GeneralKnowledge'] + contrasts_Base_loca['ContextualKnowledge'] - contrasts_Base_loca['ArithmeticFacts'] - contrasts_Base_loca['GeometricFacts']

contrastAll_loca = [
        Math_bio,
        ]

titles = [
    'Math_versus_bio',
]


#mean_image = mean_img(func_file)

p_value_list = [0.05, 0.01, 0.001]

for contrast_id in range(0, len(contrastAll_loca)):

        contrast = contrastAll_loca[contrast_id]

        contrast_name = titles[contrast_id]


        print(f'Current contrast is {contrast_name}')

        print('Computing zmap...')

        for value in p_value_list:

            zmap = fmri_glm.compute_contrast(contrast, output_type='z_score')

            #use another function to have p value
            zmap_thresholded_tuple = nilearn.glm.threshold_stats_img(zmap, alpha=value)

            # Extract the thresholded image from the tuple
            zmap_thresholded_img = zmap_thresholded_tuple[0]

            #to save it
            output_file = os.path.join(OUTPUT_DIR, f"sub-{subject_number:02}_z-map-{contrast_name}_graphs_T1w-native_p-{value}.nii.gz")
            nib.save(zmap_thresholded_img, output_file)
