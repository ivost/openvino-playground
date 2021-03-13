#!/usr/bin/env py
# coding: utf-8

# <a id="top"></a>
# # Person Counting

# This demo application demonstrates how Person Vehicle Detection algorithm and Object Tracking algorithm can be utilized for counting people crossing virtual regions in video frames. To understand how to use the algorithms for general purpose, please view the [Person Vehicle Detection sample](../../samples/person-vehicle-detector/vaspy_person_vehicle_detection_image.ipynb) and [Object Tracking sample](../../samples/object-tracker/vaspy_object_tracker_video.ipynb).

# ## Notice

# The algorithms in the video analytic suite are only for evaluation for a limited time. If you are looking for commercial version of the software, please contact Intel sales representative – Click [this link](https://devcloud.intel.com/edge/advanced/licensed_applications/) for more details.
# 
# You may only use the software and any image/video, if any, contained in the software, on the DevCloud. You may not download and use the software or any part thereof outside the DevCloud.

# ## Prerequisites

# This sample requires the following:
# 
# * All files are present and in the following directory structure:
#     * **person_counting_demo.ipynb** - This Jupyter* Notebook
#     * **person_vehicle_counting_sample.py** - Python* code for person counting demo
#     * **ava_people.mp4** - Input Video for person counting demo
#     
# <br><div class=note><i><b>Note: </b>It is assumed that the server this sample is being run on is on the Intel® DevCloud for the Edge which has Jupyter* Notebook customizations and all the required libraries including VAS 2020R3 already  installed.  If you download or copy to a new server, this sample may not run.</i></div>

# ## Demonstration objectives

# * Retail use-case demo using Object Tracker and Person Vehicle Detector in Video Analytics Suite.
# * Running inference across CPU, integrated GPU, VPU, and FPGA and comparing throughput and latency.
# * High throughput when using multiple video stream on HDDL-R.

# ## Configuration

# * **ot_configuration**: parameters for `ObjectTracker`
#     * **tracking_type**: `ObjectTracker` supports 4 types of tracking method, which are SHORT_TERM_KCFVAR(3), SHORT_TERM_IMAGELESS(4), ZERO_TERM_COLOR_HISTOGRAM(5) and ZERO_TERM_IMAGELESS(6). For detailed description, please refer to [ObjectTracker sample](../../samples/object-tracker/vaspy_object_tracker_video.ipynb#Configuration). 
# * **roi**:
#     * **boundary**: Border lines to determine whether an object passed the line or not. If a boundary is employed for bidirectional detection counting, two boundaries of equal value should be added.
#     * **movement**: "up" | "down" | "left" | "right". The direction of movement of people or vehicle to be counted when they are crossing the boundary
#     * **text_coordinates**: Point coordinate values (e.g. [x, y]) that defines the bottom left anchor of the generated text box. There must be a matching text coordinate for each defined boundary
#     * **display_polygon**: Optional. Polygon coordinates for displaying directions (e.g. arrows, triangles, etc)
# * **detection**:
#     * **pvd_cycle**: Period for `PersonVehicleDetector` to be run
#     * **plot_trail**: Flag to display object trajectory
#     * **trail_length**: Number of trajectory points to display
# * **input**:
#     * **path**: Required. Path of input video file
# * **output**:
#     * **filename**: Required. Output video file name. Output directory is predefined in results/JOB_ID, so output video file will be located in the directory.
# * **theme**:
#    * **trail_color**: BGR color representation value for detected object trajectory
#    * **track_object_color**: BGR color representation value for detected objects
#    * **text_color**: BGR color representation value for counter text
#    * **boundary_color**: BGR color representation value for boundary
#    * **display_polygon_color**: BGR color representation value for optionally displayed polygons

# <br><div class=note><i><b>Note: </b>`pvd_cycle` should be configured by tracking type of object tracker. Recommended cycle or detection period is as follows:</i></div>
# 
# <table>
#     <thead>
#         <th style="text-align:left">Tracking type</th>
#         <th style="text-align:left">Recommended `pvd_cycle` <br>for 30fps video</th>
#     </thead>
#     <tr>
#         <td style="text-align:left"><span style="font-weight:bold">Short-term-kcfvar</span></td>
#         <td style="text-align:left">5~10</td>
#     </tr>
#     <tr>
#         <td style="text-align:left"><span style="font-weight:bold">Short-term-imageless</span></td>
#         <td style="text-align:left">&lt;5 (less than 150ms)</td>
#     </tr>
#     <tr>
#         <td style="text-align:left"><span style="font-weight:bold">Zero-term-imageless</span></td>
#         <td style="text-align:left">1</td>
#     </tr>
#     <tr>
#         <td style="text-align:left"><span style="font-weight:bold">Zero-term-color-histogram</span></td>
#         <td style="text-align:left">1</td>
#     </tr>
# </table>

# ## Imports

# We begin by importing all required Python modules
# - [os](https://docs.python.org/3/library/os.html#module-os) - Operating system specific module
# - [sys](https://docs.python.org/3/library/sys.html#module-sys) - System-specific parameters and functions
# - [qarpo](https://github.com/ColfaxResearch/qarpo) - Provides utilities for displaying files and managing jobs from within this Jupyter Notebook
# - [job_interface](../../job_interface/__init__.py) - Interactive user interface allowing you to submit jobs to run the demo on multiple edge compute nodes selected by hardware devices
# 
# Run the cell below to import Python dependencies needed for displaying the results in this notebook.
# 
# <br><div class=tip><b>Tip: </b>Select a cell and then use **Ctrl+Enter** to run that cell.</div>

# In[1]:


import os
import sys

from qarpo.demoutils import videoHTML
from qarpo import liveQstat

module_path = os.path.abspath('../..')
sys.path.append(module_path)
from job_interface import VASInterface


# ## Input files

# Display input video

# In[2]:


videoHTML("Input Video",["ava_people.mp4"])


# ## Run Demo (1 stream)

# ### Generate configuration

# In[3]:


get_ipython().system('mkdir -p config')


# In[4]:


get_ipython().run_cell_magic('writefile', './config/person_counting.yml', 'ot_configuration:\n  tracking_type: 3 # SHORT_TERM_KCFVAR\n\nroi:\n  boundary: # Bidirectional boundary requires the same boundary\n    - [[480, 0], [480, 540]]\n    - [[480, 0], [480, 540]]\n  movement: # "up" | "down" | "left" | "right"\n    - \'right\'\n    - \'left\'\n  text_coordinates: # position of text box\n    - [490, 440]\n    - [460, 100]\n  display_polygon:\n    - [[480, 410], [480, 470], [520, 440]]\n    - [[440, 100], [480, 70], [480, 130]]\n\ndetection:\n  pvd_cycle: 3 # Detect and update every X frames\n  plot_trail: true\n  trail_length: 20\n\npvd_configuration:\n  # person, bicycle, motorbike, car, bus, truck, van\n  target_object:\n    - "person"\n\ninput:\n  path: \'ava_people.mp4\' # required\n\noutput:\n  filename: \'people_count.mp4\'\n\ntheme:\n  trail_color: [255, 0, 0]\n  track_object_color: [0, 255, 255]\n  text_color: [255, 255, 255]\n  boundary_color:\n    - [0, 0, 255] # Right\n    - [0, 255, 0] # Left\n  display_polygon_color:\n    - [255, 0, 0]\n    - [0, 255, 0]')


# ### Select target edge node and run

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
# Run the next cell to begin the interactive demo. <div id="job_interface1"></div>

# In[5]:


config_path = "./config/person_counting.yml"
job_interface = VASInterface(algo = "pvd",
                             filename = "person_vehicle_counting_sample.py",
                             args = [config_path])
job_interface.displayUI()


# Check if the jobs are done

# In[6]:


liveQstat()


# Remove output files and reset job_interface.
# 
# <br><div class=warn><small><b>Warning</b> Architecture selector widget above will not work after you run the following cell. If you want to run the sample again, please rerun [displayUI cell](#job_interface1) again.</small></div>

# In[7]:


job_interface.clear_output()


# ## Run Demo (multiple stream)

# ### Generate configurations

# All has the same configuration except output filename.

# In[8]:


get_ipython().system('mkdir -p config')


# In[9]:


template = """
ot_configuration:
  tracking_type: 3 # SHORT_TERM_KCFVAR

roi:
  boundary: # Bidirectional boundary requires the same boundary
    - [[480, 0], [480, 540]]
    - [[480, 0], [480, 540]]
  movement: # "up" | "down" | "left" | "right"
    - 'right'
    - 'left'
  text_coordinates: # position of text box
    - [490, 440]
    - [460, 100]
  display_polygon:
    - [[480, 410], [480, 470], [520, 440]]
    - [[440, 100], [480, 70], [480, 130]]

detection:
  pvd_cycle: 3 # Detect and update every X frames
  plot_trail: true
  trail_length: 20

pvd_configuration:
  # person, bicycle, motorbike, car, bus, truck, van
  target_object:
    - "person"

input:
  path: "ava_people.mp4" # required

output:
  filename: "people_count{}.mp4"

theme:
  trail_color: [255, 0, 0]
  track_object_color: [0, 255, 255]
  text_color: [255, 255, 255]
  boundary_color:
    - [0, 0, 255] # Right
    - [0, 255, 0] # Left
  display_polygon_color:
    - [255, 0, 0]
    - [0, 255, 0]
"""


# In[10]:


num_streams = 8
for i in range(num_streams):
    with open("config/person_counting{}.yml".format(i), "w") as f:
        f.write(template.format(i))


# ### Select target edge node and run

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
# 
# <br><div class=warn><small><b>Warning</b> The inference times in the performance charts are underestimated due to the logic not taking into account multithreaded use cases. To get accurate inference times, please try [Object Tracker Sample](../../samples/object-tracker/vaspy_object_tracker_video.ipynb) and [Person Vehicle Detector Sample](../../samples/person-vehicle-detector/vaspy_person_vehicle_detection_image.ipynb).</small></div><div id="job_interface2"></div>

# In[11]:


config_paths = ["config/person_counting{}.yml".format(i) for i in range(num_streams)]
job_interface = VASInterface(algo = "pvd",
                             multistream = True,
                             filename = "person_vehicle_counting_sample.py",
                             args = config_paths)
job_interface.displayUI()


# Check if the jobs are done

# In[17]:


get_ipython().system('qpeek 64147')


# In[13]:


liveQstat()


# Remove output files and reset job_interface.
# 
# <br><div class=warn><small><b>Warning</b> Architecture selector widget above will not work after you run the following cell. If you want to run the sample again, please rerun [displayUI cell](#job_interface2) again.</small></div>

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
