#! /usr/bin/env python
# coding: UTF-8


##This code was provided by Thomas and checks the different events, creating a tsv file
# However, it does not have the differences between left and right button presses

#Run on the server like this: "python xpd_to_tsv_loca.py -s 3"

# %%
# Imports

import os
import nilearn
import sys
import optparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nilearn.plotting import plot_design_matrix


#%%

SUBJ_ID = 201

DATA_PATH = f"/data/derivatives/fmriprep/sub-{SUBJ_ID:02}/func"

EVENTS_PATH = "/neurospin/unicog/protocols/Daniela_MathBrain_V2_fmri/Results/pilot_graphs/events"
                
onsetfile = f"/neurospin/unicog/protocols/MathBrain_Dehaene_2024/Data/MathBrain_7T_V2/output_files_mathbrainbig40_v2/sub{SUBJ_ID:02}/Graphs/sub201/VectorOffset_localizer_sub{SUBJ_ID:02}.xpd"

# Debug print to check paths
print(f"DATA_PATH: {DATA_PATH}")
print(f"onsetfile: {onsetfile}")


# Check if the file exists
# Debug print to check paths

if not os.path.exists(onsetfile):
    raise FileNotFoundError(f"The file {onsetfile} does not exist.")


#%%
TRIAL_TYPE = {
    '01-math': '01-math',
    '02-bio': '02-bio'
}


events_df = pd.read_csv(onsetfile, comment = '#')

# Duration
events_df['duration'] = events_df['end_time'] - events_df['start_time']

# Retrieve pressed button
allEvents = []
for t in events_df.itertuples():
    if t.cond == 'keypressed':
        # to divide the ms into seconds
        allEvents.append([t.start_time/1000, t.duration/1000, f"button_{t.sentence}"])
    elif t.cond == 'endstim': # excludes last event
        continue
    else:
        allEvents.append([t.start_time/1000, t.duration/1000, t.cond])

events_df = pd.DataFrame(allEvents, columns=['onset', 'duration', 'trial_type'])

# events_df_prt = events_df[['start_time', 'end_time', 'duration', 'cond']]

# events_df = events_df[['start_time', 'duration', 'cond']]

# to have the typical file structure

# events_df.rename(columns = {'start_time': 'onset', 
#                             'cond': 'trial_type'}, inplace=True)


# %%
# events_df.to_csv(os.path.join(DATA_PATH, f"sub-{SUBJ_ID:02d}_mathlang_events.csv"), index=False)


# %%
# Write CSV as events tsv
events_df.to_csv(f"{EVENTS_PATH}/sub{SUBJ_ID:02}_graphs_events.tsv", index=False)

## plot to check quality
nilearn.plotting.plot_event(events_df, figsize=(15, 5))
plt.show()

