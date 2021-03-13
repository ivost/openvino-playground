#!/usr/bin/env py
# coding: utf-8

# <a id="top"></a>
# # Social Distancing

# This demo application demonstrates how Person Vehicle Detection algorithm can be utilized for social distancing. To understand how to use the algorithm for general purpose, please view the [Person Vehicle Detection sample](../../samples/person-vehicle-detector/vaspy_person_vehicle_detection_image.ipynb).

# ## Notice

# The algorithms in the video analytic suite are only for evaluation for a limited time. If you are looking for commercial version of the software, please contact Intel sales representative – Click [this link](https://devcloud.intel.com/edge/advanced/licensed_applications/) for more details.
# 
# You may only use the software and any image/video, if any, contained in the software, on the DevCloud. You may not download and use the software or any part thereof outside the DevCloud.

# ## Prerequisites

# This sample requires the following:
# 
# * All files are present and in the following directory structure:
#     * **social_distancing.ipynb** - This Jupyter* Notebook
#     * **social_distancing.py** - Python* code for social distancing demo
#     * **util.py** - Python* code for utility functions for this demo
#     * **building_entrance.mp4** - Input video file
# 
#     
# <br><div class=note><i><b>Note: </b>It is assumed that the server this sample is being run on is on the Intel® DevCloud for the Edge which has Jupyter* Notebook customizations and all the required libraries including VAS 2020R3 already  installed.  If you download or copy to a new server, this sample may not run.</i></div>

# ## Demonstration objectives

# * Social distancing use-case demo using Person Vehicle Detector in Video Analytics Suite.
# * Running inference across CPU, integrated GPU, VPU, FPGA and HDDL-R and comparing throughput and latency

# ## Configuration

# * **input_path**: Path to source video file.
# * **model_path**: Directory path to PVD model.
# * **calibration_points**: Required. List of four 2-dimensional points for calibration. The points represents a square whose side length is `cal_distance`. Order of points should be top left, top right, bottom right, then bottom left(Clockwise). The format should be comma-separated list of coords in image. e.g. "\[\[200,200\],\[400,200\],\[400,400\],\[200,400\]\]".
# * **calibration_distance**: Optional. Distance between 2 calibration points. Unit: meter (default: 1.0)
# * **roi**: Optional. Region of Interest to monitor social distancing. Only people in `roi` will be detected and the distance between them will be calculated. The format shoule be "\[left,top,right,bottom\]" (default: None)
# * **max_display_distance**: Optional. Maximum distance between 2 people for displaying. Unit: meter (default: Max float)
# * **min_safe_distance**: Optional. Minimum safe distance between 2 people in meters. If the distance between 2 people is less than `min_safe_distance`, the color of edge between them turns to red indicating it is not safe. (default: 2.0)

# ## Imports

# We begin by importing all required Python modules
# - [os](https://docs.python.org/3/library/os.html#module-os) - Operating system specific module
# - [sys](https://docs.python.org/3/library/sys.html#module-sys) - System-specific parameters and functions
# - [json](https://docs.python.org/3/library/json.html#module-json) - JSON encoder and decoder
# - [qarpo](https://github.com/ColfaxResearch/qarpo) - Provides utilities for displaying files and managing jobs from within this Jupyter Notebook
# - [job_interface](../../job_interface/__init__.py) - Interactive user interface allowing you to submit jobs to run the demo on multiple edge compute nodes selected by hardware devices
# - [util](./util.py) - Utility functions for displaying calibration points
# 
# Run the cell below to import Python dependencies needed for displaying the results in this notebook 
# 
# <br><div class=tip><b>Tip: </b>Select a cell and then use **Ctrl+Enter** to run that cell.</div>

# In[1]:


import os
import sys
import json

from qarpo.demoutils import videoHTML
from qarpo import liveQstat

module_path = os.path.abspath('../..')
sys.path.append(module_path)
from job_interface import VASInterface

from util import show_calibration_point


# ## Input files

# Display input video

# In[2]:


videoHTML("Input Video",["building_entrance.mp4"])


# ## Run Social Distancing Application on Edge Devices

# ### Generate configuration

# In[3]:


# Required. Path to source video file.
input_video = "building_entrance.mp4"

# Required. Directory path to PVD model
model_path = os.path.join(os.environ["PATH_VAS"], "lib", "intel64")

# Required. 2D points for calibration
calibration_points = [[500, 442], [694, 398], [836, 488], [630, 554]]

# Optional. Distance between 2 calibration points in meter (Default: 1.0)
calibration_distance = 4.5

# Optional. Region of Interest to monitor social distancing (Default: None)
roi = [300, 232, 1000, 706]

# Optional. Maximum distance between 2 people for displaying (Default: Max float)
max_display_distance = 4.0

# Optional. Maximum safe distance between 2 people (Default: 2.0)
min_safe_distance = 2.0


# In[4]:


config = {
    "input": input_video,
    "model": model_path,
    "cal_pts": calibration_points,
    "roi" : roi,
    "cal_distance": calibration_distance,
    "max_display_distance": max_display_distance,
    "min_safe_distance": min_safe_distance
}

# export in JSON
get_ipython().system('mkdir -p config')
config_path = './config/social_distancing.json'
with open(config_path, 'w') as f:
    json.dump(config, f, indent=4)


# Display calibration points.

# In[5]:


show_calibration_point(input_video, calibration_points, calibration_distance, roi=roi, width=12)


# ### Run demo

# Running the next cell will display an interactive user interface allowing you to submit jobs to run the demo on multiple edge compute nodes selected by hardware devices, view the output of each job, and compare performance results across jobs.
# 
# To run a job:
# 1. Select the desired option in the **Target node** list
# 2. Select the desired device in the **Target architecture** list
# 3. Click the **Submit** button
# 
# After the **Submit** button is clicked, a tab will appear for the new job with a label in the format "*status*: *JobID*".  Once the job status appears as "Done", the **Display output** button may be clicked to view the output for the job.
# 
# After one or more jobs are done, the performance results for each job may be plotted by clicking the **Plot results** button.  Results for each job will be potted as bar graphs for **inference time** and **frames per second**.  Lower values mean better performance for **inference time** and higher values mean better performance for **frames per second**. When comparing results, please keep in mind that some architectures are optimized for highest performance, others for low power or other metrics.
# 
# Run the next cell to begin the interactive demo.

# In[6]:


job_interface = VASInterface(algo = 'pvd',
                             filename = "social_distancing.py",
                             args = ["-c", config_path])
job_interface.displayUI()


# Check if the jobs are done

# In[ ]:


liveQstat()


# Remove output files and reset job_interface.
# 
# <br><div class=warn><small><b>Warning</b> Architecture selector widget above will not work after you run the following cell. If you want to run the sample again, please rerun [displayUI cell](#job_interface) again.</small></div>

# In[ ]:


job_interface.clear_output()


# ## Next steps
# 
# - [More Video Analytics Suite Applications](https://devcloud.intel.com/edge/advanced/licensed_applications/) additional sample and demo application
# - [Intel® Distribution of OpenVINO™ toolkit Main Page](https://software.intel.com/openvino-toolkit) - learn more about the tools and use of the Intel® Distribution of OpenVINO™ toolkit for implementing inference on the edge

# <p style=background-color:#0071C5;color:white;padding:0.5em;display:table-cell;width:100pc;vertical-align:middle>
# <img style=float:right src="https://devcloud.intel.com/edge/static/images/svg/IDZ_logo.svg" alt="Intel DevCloud logo" width="150px"/>
# <a style=color:white>Intel® DevCloud for the Edge</a><br>   
# <a style=color:white href="#top">Top of Page</a> | 
# <a style=color:white href="https://devcloud.intel.com/edge/static/docs/terms/Intel-DevCloud-for-the-Edge-Usage-Agreement.pdf">Usage Agreement (Intel)</a> | 
# <a style=color:white href="https://devcloud.intel.com/edge/static/docs/terms/Colfax_Cloud_Service_Terms_v1.3.pdf">Service Terms (Colfax)</a>
# </p>
