#!/usr/bin/env python
# coding: utf-8

# <a id="top"></a>
# # Queue Monitoring

# This demo application demonstrates how Head Detection algorithm can be utilized for checking how many people are in defined queue at store. To understand how to use the algorithm for general purpose, please view the [Head Detection sample](../../samples/head-detector/vaspy_head_detection_video.ipynb).

# ## Notice

# The algorithms in the video analytic suite are only for evaluation for a limited time. If you are looking for commercial version of the software, please contact Intel sales representative – Click [this link](https://devcloud.intel.com/edge/advanced/licensed_applications/) for more details.
# 
# You may only use the software and any image/video, if any, contained in the software, on the DevCloud. You may not download and use the software or any part thereof outside the DevCloud.

# ## Prerequisites

# This sample requires the following:
# 
# * All files are present and in the following directory structure:
#     * **queue_monitoring_demo.ipynb** - This Jupyter* Notebook
#     * **queue_monitoring_sample.py** - Python* code for the queue monitoring demo
#     * **ava_queue_monitoring.mp4** - Input video for the queue monitoring demo
#     
# <br><div class=note><i><b>Note: </b>It is assumed that the server this sample is being run on is on the Intel® DevCloud for the Edge which has Jupyter* Notebook customizations and all the required libraries including VAS 2020R3 already  installed.  If you download or copy to a new server, this sample may not run.</i></div>

# ## Demonstration objectives

# * Retail use-case demo using Head Detector in Video Analytics Suite.
# * Running inference across CPU, integrated GPU, VPU, and FPGA and comparing throughput and latency
# * High throughput when using multiple video stream on HDDL-R. 

# ## Configuration

# * **hd_configuration**: parameters for `HeadDetector`
#     * **min_head_ratio**: Minimum head ratio to the input frame. Minimum detectable head size is calculated by MIN(frame width, frame height) * min_head_ratio. Minimum detectable head size cannot be smaller than 16x16 in pixel. Even though you specify a value that can make head size smaller than 24, HeadDetector finds heads down to size of 24. Valid range: 0.0 $\le$ min_head_ratio $\le$ 1.0. Default value is 0.02.
#     * **threshold**: Threshold that affects the number of detected heads. HeadDetector internally has well-tuned threshold that can be applied in most cases. Users can change the threshold through this attribute. If this value is smaller than the default, HeadDetector generates more heads because threshold gets smaller. In this case, False acceptance ratio can increase. On the other hand, if this value is greater than the default, HeadDetector generates less heads because threshold gets bigger. Valid range: 0.0 $\le$ min_head_ratio $\le$ 1.0. Default value is 0.6.
#     
# * **roi**:
#     * **box_coordinates**: Input multiple polygon coordinates to set region of interest (ROI). There could be more than one polygon defined. Polygon input should be in clockwise or counterclockwise fashion. The last point must be connected to the first defined coordinate in the polygon.
#     * **text_coordinates**: Point coordinate values (e.g. [x, y]) that define the bottom left anchor of generated text boxes. There must exist a matching text coordinate for each defined polygon.
# 
# * **detection**:
#     * **box_max_person**: Maximum number of people that would activate or render the box to change color.
#     * **intersection**: detection method
#         * **type**: \['head_center' | 'area'\]. With 'head_center', the box would count head to be included if and only if the center of detected head is within the polygon boundary. With 'area', the overlapping area in percentage representation to head box area will be calculated. Upon passing predefined area_threshold, the ROI would count a head to be included.
#         * **area_threshold**: Threshold of intersected area in terms of head box area to detect head box inclusion in polygon. This is ignored when type is head_center.
# * **input**:
#     * **path**: Required. Path of input video file
# * **output**:
#     * **filename**: Required. Output video file name. Output directory is predefined in results/JOB_ID, so output video file will be located in the directory.
#     * **update_cycle**: Refresh rate of counter text for each defined polygon
# * **theme**:
#     * **head_default_color**: BGR color representation value for detected heads that are not included in any ROI
#     * **head_activation_color**: BGR color representation value for detected heads that are included in any ROI
#     * **roi_default_color**: BGR color representation value for ROIs containing detected heads less than box_max_person value
#     * **roi_activation_color**: BGR color representation value for ROIs containing detected heads greater tor equal to box_max_person value

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


videoHTML("Input Video",["ava_queue_monitoring.mp4"])


# ## Run Demo (1 stream)

# ### Generate configuration

# In[3]:


get_ipython().system('mkdir -p config')


# In[4]:


get_ipython().run_cell_magic('writefile', './config/queue_monitoring.yml', 'hd_configuration:\n  min_head_ratio: 0.02   # default: 0.02\n  threshold: 0.6         # default: 0.6\n\nroi:\n  box_coordinates:\n    - [[40, 50], [50, 490], [430, 490], [440, 50], [40, 50]] # trapezoid / clockwise coords\n    - [[520, 50], [530, 490], [910, 490], [920, 50], [520, 50]] # trapezoid / clockwise coords\n  text_coordinates:\n    - [60, 480]\n    - [540, 480]\n\ndetection:\n  box_max_person: 2\n  intersection:\n    type: \'area\' # [head_center | area]\n    area_threshold: 0.5\n\ninput:\n  path: "ava_queue_monitoring.mp4" # required\n\noutput:\n  filename: "queue-monitoring.mp4" # support mp4 format, default: "queue-monitoring.mp4"\n  update_cycle: 20                 # period to update counter\n\ntheme:\n  head_default_color: [255, 0, 0]\n  head_activation_color: [80, 127, 255]\n  roi_default_color: [0, 255, 0]\n  roi_activation_color: [0, 0, 255]\n  text_size: 0.7                          # default 0.7')


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


config_path = "./config/queue_monitoring.yml"
job_interface = VASInterface(algo = "hd",
                             filename = "queue_monitoring_sample.py",
                             args = [config_path])
job_interface.displayUI()


# Check if the jobs are done

# In[ ]:


liveQstat()


# Remove output files and reset job_interface.
# 
# <br><div class=warn><small><b>Warning</b> Architecture selector widget above will not work after you run the following cell. If you want to run the sample again, please rerun [displayUI cell](#job_interface1) again.</small></div>

# In[ ]:


job_interface.clear_output()


# ## Run Demo (multiple stream)

# ### Generate configurations

# All has the same configuration except output filename.

# In[ ]:


template = """
hd_configuration:
  min_head_ratio: 0.02   # default: 0.02
  threshold: 0.6         # default: 0.6

roi:
  box_coordinates:
    - [[40, 50], [50, 490], [430, 490], [440, 50], [40, 50]] # trapezoid / clockwise coords
    - [[520, 50], [530, 490], [910, 490], [920, 50], [520, 50]] # trapezoid / clockwise coords
  text_coordinates:
    - [60, 480]
    - [540, 480]

detection:
  box_max_person: 2
  intersection:
    type: 'area' # [head_center | area]
    area_threshold: 0.5

input:
  path: "ava_queue_monitoring.mp4" # required

output:
  filename: "queue-monitoring{}.mp4" # support mp4 format, default: "queue-monitoring.mp4"
  update_cycle: 20                 # period to update counter

theme:
  head_default_color: [255, 0, 0]
  head_activation_color: [80, 127, 255]
  roi_default_color: [0, 255, 0]
  roi_activation_color: [0, 0, 255]
  text_size: 0.7                          # default 0.7
"""


# In[ ]:


num_streams = 8
for i in range(num_streams):
    with open("config/queue_monitoring{}.yml".format(i), "w") as f:
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
# <br><div class=warn><small><b>Warning</b> The inference times in the performance charts are underestimated due to the logic not taking into account multithreaded use cases. To get accurate inference times, please try [Head Detector Sample](../../samples/head-detector/vaspy_head_detection_video.ipynb).</small></div><div id="job_interface2"></div>

# In[ ]:


config_paths = ["config/queue_monitoring{}.yml".format(i) for i in range(num_streams)]
job_interface = VASInterface(algo = "hd",
                             multistream = True,
                             filename = "queue_monitoring_sample.py",
                             args = config_paths)
job_interface.displayUI()


# Check if the jobs are done

# In[ ]:


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
