#!/usr/bin/env py
# coding: utf-8

# <a id="top"></a>
# # Video Analytics Suite - Object Tracker

# This sample is to demonstrate how to use Object Tracking algorithm in video analytics suite. To see the algorithm used in a real-world use-case, please view the [Person Counting Demo](../../demos/person-counting/person_counting_demo.ipynb) and [Vehicle Counting Demo](../../demos/vehicle-counting/vehicle_counting_demo.ipynb).

# ## Notice

# The algorithms in the video analytic suite are only for evaluation for a limited time. If you are looking for commercial version of the software, please contact Intel sales representative – Click [this link](https://devcloud.intel.com/edge/advanced/licensed_applications/) for more details.
# 
# You may only use the software and any image/video, if any, contained in the software, on the DevCloud. You may not download and use the software or any part thereof outside the DevCloud.

# ## Prerequisites

# This sample requires the following:
# 
# * All files are present and in the following directory structure:
#     * **vaspy_object_tracker_video.ipynb** - This Jupyter* Notebook
#     * **ot_video_sample.py** - Python* code for object tracking sample
#     * **ava_people.mp4** - Input video file for object tracking
#     
# <br><div class=note><i><b>Note: </b>It is assumed that the server this sample is being run on is on the Intel® DevCloud for the Edge which has Jupyter* Notebook customizations and all the required libraries including VAS 2020R3 already  installed.  If you download or copy to a new server, this sample may not run.</i></div>

# ## Introduction
# 

# This sample showcases how to use Object Tracker’s API, a component of Video Analytics Suite, on Intel® CPUs.
# 
# Object Tracker is a general tracker that can work with any kind of detector including Head Detector or Person Vehicle Detector of Video Analytics Suite.
# Once an object is added, it is tracked with video frames.
# 
# Object Tracker supports three types of tracking method: short-term, and zero-term.
# Each tracking type has different characteristics, so it could be selected depending on your system design.
# 
# Object Tracker is not a DNN-based component, and it runs only on CPU.
# 
# A tracked object includes a rectangle, tracking status, class label and tracking ID.
# 
# Further detail on tracked object can be found in the *`vaspy.ot.Object`* class.
# 
# Object Tracker gets its input frame by *`numpy.ndarray`*  type, a type used by OpenCV(`cv2`), so it is recommended to decode images or videos through the supported functionalities of OpenCV(`cv2`).
# 
# Object Tracker requires OpenVINO&trade; to run.

# ## Object Tracking Application

# The following sections will walk you through a sample application.

# ### Imports

# We begin by importing all required Python modules
# - [os](https://docs.python.org/3/library/os.html#module-os) - Operating system specific module (used for setting environment variable)
# - [sys](https://docs.python.org/3/library/sys.html#module-sys) - System-specific parameters and functions
# - [cv2](https://docs.opencv.org/trunk/) - OpenCV module
# - [time](https://docs.python.org/3/library/time.html#module-time) - Time tracking module (used for measuring execution time)
# - [numpy](http://www.numpy.org/) - N-dimensional array manipulation
# - [tqdm](https://github.com/tqdm/tqdm) - Progress bar to display inference progress.
# - [qarpo](https://github.com/ColfaxResearch/qarpo) - Provides utilities for displaying files and managing jobs from within this Jupyter Notebook
# - [job_interface](../../job_interface/__init__.py) - Interactive user interface allowing you to submit jobs to run the demo on multiple edge compute nodes selected by hardware devices
# 
# Run the cell below to import Python dependencies needed for displaying the results in this notebook.
# 
# <br><div class=tip><b>Tip: </b>Select a cell and then use **Ctrl+Enter** to run that cell.</div>

# In[25]:


import os
import sys
import cv2
import time

import numpy as np
from tqdm.notebook import tqdm
from qarpo.demoutils import videoHTML
from qarpo import liveQstat

module_path = os.path.abspath('../..')
sys.path.append(module_path)
from job_interface import VASInterface

print("Imported Python modules")


# ### Import VAS

# Before importing `vaspy`, environment variables `LD_LIBRARY_PATH` and `PYTHONPATH` need to be set. `LD_LIBRARY_PATH` should specify the VAS shared libraries directory and `PYTHONPATH` should specify the vaspy shared library directory. In DevCloud, these variables are set by the system administrator so that you don't need to configure them yourself. In case you'd like to import VAS on your local system, however, you should set the `LD_LIBRARY_PATH` and `PYTHONPATH` with proper path.
# 
# Object Tracker (**OT**) requires initial input of object. For this sample, Head Detector (**HD**) is used to detect objects.

# In[26]:


from vaspy import hd as HD
from vaspy import ot as OT
from vaspy.common import *

print("Imported VAS modules")


# ### Configuration

# Here we will create and set the following configuration parameters used by the sample:  
# * **model_path** - Path to a directory including the .bin IR file of the trained model to use for inference. This is for Head Detector, instead of Object Tracker.
# * **max_num_objects** - Maximum number of trackable objects in a frame. Valid range: 1 $\le$ max_num_objects. The value may be -1, which indicates an unconstrained maximum number. Default value is -1.
# * **tracking_type** - Tracking type of Object Tracker. It supports 4 tracking types: SHORT_TERM_KCFVAR, SHORT_TERM_IMAGELESS, ZERO_TERM_IMAGELESS, ZERO_TERM_COLOR_HISTOGRAM. 
# 
# <table>
#     <thead>
#         <th style="text-align:left">Tracking type</th>
#         <th style="text-align:left">Recommended detection period for 30fps video</th>
#         <th style="text-align:left">Description</th>
#     </thead>
#     <tr>
#         <td style="text-align:left"><span style="font-weight:bold">Short-term-kcfvar</span></td>
#         <td style="text-align:left">more than 5 frames<br>(Strongly recommended to use around 5~10 frames)</td>
#         <td style="text-align:left">
#             <ul>
#                 <li>This type utilizes color and feature information of input objects.</li>
#                 <li>Shorter period guarantees better accuracy.</li>
#                 <li>It achieves better accuracy.</li>
#             </ul>
#         </td>
#     </tr>
#     <tr>
#         <td style="text-align:left"><span style="font-weight:bold">Short-term-imageless</span></td>
#         <td style="text-align:left">less than 5 frames<br>(less than 150ms)</td>
#         <td style="text-align:left">
#             <ul>
#                 <li>This type utilizes object shape and position information.</li>
#                 <li>Longer periodic input of objects' detection cannot guarantee the accuracy. It achieves absolutely
#                     higher throughput.</li>
#                 <li> A user needs to consider the trade-off between throughput and accuracy for choosing imageless type.
#                 </li>
#             </ul>
#         </td>
#     </tr>
#     <tr>
#         <td style="text-align:left"><span style="font-weight:bold">Zero-term-imageless</span></td>
#         <td style="text-align:left">every frame</td>
#         <td style="text-align:left">
#             <ul>
#                 <li>This type utilizes position, shape and input image information such as RGB histogram.
#                 </li>
#                 <li>For each frame, detected objects are mapped with input rectangular coordinates.</li>
#             </ul>
#         </td>
#     </tr>
#     <tr>
#         <td style="text-align:left"><span style="font-weight:bold">Zero-term-color-histogram</span></td>
#         <td style="text-align:left">every frame</td>
#         <td style="text-align:left">
#             <ul>
#                 <li>This type only utilizes object shape and position information.
#                 </li>
#                 <li>For each frame, detected objects are mapped with input rectangular coordinates.</li>
#             </ul>
#         </td>
#     </tr>
# </table>
# 
# 
# We will set all parameters here excluding *input_path*, which we will modify later to reference different videos.

# In[27]:


PATH_VAS = os.environ['PATH_VAS']

# path to model
model_path = "{}/lib/intel64".format(PATH_VAS)

# maximum number of trackable objects in a frame
# Default: -1
max_num_objects = -1

# tracking type:
# SHORT_TERM_KCFVAR   | SHORT_TERM_IMAGELESS
# ZERO_TERM_IMAGELESS | ZERO_TERM_COLOR_HISTOGRAM
tracking_type = OT.TrackingType.SHORT_TERM_KCFVAR

print("Configuration")
print("- {:21s}: {}".format("model_path",model_path))
print("- {:21s}: {}".format("max_num_objects", max_num_objects))
print("- {:21s}: {}".format("tracking_type", tracking_type))


# ### Build HeadDetector
# 
# We will build `HeadDetector` to detect people heads from video frames. `ObjectTracker` alone does not perform object tracking so it requires an auxiliary detector. 

# In[28]:


# Init HD
hd_builder = HD.HeadDetector.Builder()
hd_builder.backend_type = BackendType.CPU

# Build
hd = hd_builder.Build(model_path)

print("Instance of HeadDetector is created")


# ### Build ObjectTracker
# 
# We will build `ObjectTracker` with the configuration we set in the [Configuration](#Configuration) section

# In[29]:


# Init ObjectTracker
ot_builder = OT.ObjectTracker.Builder()
ot_builder.max_num_objects = max_num_objects      

# Build
ot = ot_builder.Build(tracking_type)

print("Instance of ObjectTracker is created")


# ### Track objects and save output to video file

# Load video file

# In[30]:


input_path = "ava_people.mp4"

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise RuntimeError("fail to get source frame")
    
print("Video file is loaded")


# Display loaded video

# In[31]:


videoHTML("Input Video",[input_path])


# Set the output video file

# In[32]:


output_path = "./output_ot.mp4"
get_ipython().system('rm -f output_path')

# Define the codec and create VideoWriter object
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Font setting in output video
font_args = [cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1]

try:
    video_out = cv2.VideoWriter(output_path, fourcc, fps, (width,height))
    print("VideoWriter is created successfully.")
except Exception as e:
    print("Error while creaing VideoWriter")


# Set detection period

# In[33]:


detect_periods = {
    OT.TrackingType.SHORT_TERM_KCFVAR : 10,
    OT.TrackingType.SHORT_TERM_IMAGELESS : 3,
    OT.TrackingType.ZERO_TERM_IMAGELESS : 1,
    OT.TrackingType.ZERO_TERM_COLOR_HISTOGRAM : 1
}

detect_period = detect_periods[tracking_type]


# Here we will detect people's heads from the video frames by completing the following steps:
# 
# 1. Get Region Of Interest(ROI) bounding box coordinates for the detected objects by an auxiliary detector. This step has to be done periodically.
# 2. Create a list of `DetectedObject` with the bounding box coordinates.
# 3. Track the objects.

# Let's begin with the first frame of the video.
# 
# <br><div class=note><i><b>Note: </b>This sample is run on Intel® Xeon CPU hosting this Jupyter* notebook rather than edge compute node.</i></div>

# In[34]:


ret, frame = cap.read()
if ret:
    # 1. Get Region Of Interest(ROI) bounding box coordinates
    objects = hd.Detect(frame, False)
    
    # 2. Create a list of DetectedObject 
    detected_objects = [OT.DetectedObject(o.rect, 0) for o in objects]
    
    # 3. Track the objects
    tracked_objects = ot.Track(frame, detected_objects)
else:
    print("Error while reading frame #0")


# Track objects from the remaining video frames.

# In[35]:


bar_format = "{percentage:3.0f}%|{bar}|Time remaining: {remaining} ({rate_fmt} {postfix})"

try:
    with tqdm(total = frame_cnt-1, unit="frame", bar_format=bar_format) as t:
        for i in range(1, frame_cnt):
            # Update progress bar
            t.update()
            
            # Get frame
            ret, frame = cap.read()
            if not ret:
                print("Error while reading frame #", i)
                break
            detected_objects = []
        
            # 1. Get Region Of Interest(ROI) bounding box coordinates
            # 2. Create a list of DetectedObject 
            if i % detect_period == 0:
                objects = hd.Detect(frame, False)
                detected_objects = [OT.DetectedObject(o.rect, 0) for o in objects]

            # 3. Track the objects
            tracked_objects = ot.Track(frame, detected_objects)
            for object in tracked_objects:
                id = "TID: {}".format(object.tracking_id)
                if object.status != OT.TrackingStatus.LOST:
                    cv2.rectangle(frame, object.rect[:2],object.rect[2:], (0, 255, 0))
                    cv2.putText(frame, id, (object.rect[0] - 1, object.rect[1] - 1), *font_args)

            # Write output
            video_out.write(frame)
    print("Track() process complete")
    
except Exception as e:
    print(e)


# Close opened video input and output files

# In[13]:


cap.release()
video_out.release()

print("Closed video files")


# Display resulting video

# In[14]:


print("Display resulting video with face rectangles")

videoHTML("Object Tracking Result",[output_path])


# ### Cleanup outputs

# In[36]:


get_ipython().system('rm $output_path')


# ## Performance comparison among multiple backend devices

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
# Run the next cell to begin the interactive demo. <div id="job_interface"></div>

# In[ ]:


tracking_types = {
    "SHORT_TERM_KCFVAR" : 3,
    "SHORT_TERM_IMAGELESS" : 4,
    "ZERO_TERM_IMAGELESS" : 5,
    "ZERO_TERM_COLOR_HISTOGRAM" : 6
}

input_path = "ava_people.mp4"
tracking_type = tracking_types["SHORT_TERM_KCFVAR"]

job_interface = VASInterface(algo = "ot",
                             filename = "ot_video_sample.py",
                             args = ["-t",tracking_type, input_path],
                             output_type="video")
job_interface.displayUI()


# Check if the jobs are complete

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
