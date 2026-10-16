#! /usr/bin/env python
# coding: UTF-8

"""
Convert Expyriment recordings (.xpd) into an events.tsv file
for first-level fMRI analysis.

The resulting table contains:
- Sentence events: graph and biology conditions
- Button events: True (V) and False (F) responses

Timing is expressed in seconds relative to the experiment clock.

NOTE: Sentence events currently include the 350 ms red cue.
"""

# %%
# Imports

import os
import nilearn
import pandas as pd
import matplotlib.pyplot as plt
from nilearn.plotting import plot_design_matrix


#%%
# Define subject and file paths

SUBJ_ID = 201

# Absolute project path
PROJECT_ROOT = "/data/mvladisauskas/MathBrain_V2"

# Where the generated events.tsv will be saved
EVENTS_PATH = f"{PROJECT_ROOT}/results/00-events"

# Original Expyriment file containing trial timings
onsetfile = (f"{PROJECT_ROOT}/sourcedata/sub-{SUBJ_ID:02}/"
            f"VectorOffset_localizer_sub{SUBJ_ID:02}.xpd")

# Debug print to check paths
print(f"onsetfile: {onsetfile}")

# Check if the file exists
if not os.path.exists(onsetfile):
    raise FileNotFoundError(f"The file {onsetfile} does not exist.")

#%%
# Load experimental recordings

# The .xpd file contains trial information and response records (also comments with #)
events_df = pd.read_csv(onsetfile, comment = '#')

# Compute the duration of each recorded event in ms (including cue)
events_df['duration'] = events_df['end_time'] - events_df['start_time']

# Convert recorded events into GLM-compatible events
allEvents = []
for t in events_df.itertuples():
    # There are three possible events: response, sentence presentation, and end of the experiment
    if t.cond == 'keypressed':
        # Responses, keep 'sentence' (V or F) and timing in seconds
        allEvents.append([t.start_time/1000, t.duration/1000, f"button_{t.sentence}"])
    elif t.cond == 'endstim':
        # Ignore the special marker at the end of the experiment 
        continue
    else:
        # Sentence presentation, separate condition by hierarchy
         cond = "graph" if t.cond == "math" else "bio"
         trial_type = f"{cond}_h{int(t.hierarchy)}"

    allEvents.append([t.start_time/1000, t.duration/1000, trial_type])

# Final events table
## Each row is one event, described by:
## onset--> time at which it starts, in seconds
## duration--> length of the event, in seconds
## trial_type--> experimental condition or response category
events_df = pd.DataFrame(allEvents, columns=['onset', 'duration', 'trial_type'])

# %%
# Write CSV as events tsv
events_df.to_csv(
    f"{EVENTS_PATH}/sub{SUBJ_ID:02}_graphs_events_glm2.tsv",
    sep="\t",
    index=False
)

# Visualize event timing to check quality
nilearn.plotting.plot_event(events_df, figsize=(15, 5))
plt.show()


# %%
