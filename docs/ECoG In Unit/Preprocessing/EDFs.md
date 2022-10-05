---
title: EDF files
parent: Preprocessing
has_children: false
---
# EDF files
## <span style="font-family:Arial;font-size:115%;color:#000000;">Check triggers when EDF files are uploaded to Box</span>

1.  <span style="font-family:Arial;font-size:115%;color:#000000;">After running tasks with a subject, we submit Task Name, Date, and Begin/End time of each task to the neurology team in the Neurology Timestamp doc.</span>
2.  <span style="font-family:Arial;font-size:115%;color:#000000;">A member of the neurology team will parse the long recordings into smaller EDF files that contain the range of data for each task. See Timestamps instructions</span>
3.  <span style="font-family:Arial;font-size:115%;color:#000000;">The EDF files will be uploaded to Box Sync\CoganLab\ECoG_Task_Data\TaskUploadDir</span>
4.  <span style="font-family:Arial;font-size:115%;color:#000000;">Once uploaded, we need to validate that the correct range was extracted.</span>
5.  <span style="font-family:Arial;font-size:115%;color:#000000;">Please take a look at view_triggers.m for further instructions for loading and visualizing the trigger channel contained in the edf file. The triggers should look like the following (Insert window of triggers from EDF section)</span>
6.  <span style="font-family:Arial;font-size:115%;color:#000000;">Once EDF is validated, you can move to Box Sync\CoganLab\ECoG_Task_Data\Cogan_EDF </span>

<span style="font-family:Arial;font-size:115%;color:#000000;">ECoG Preprocessing</span>

1.  <span style="font-family:Arial;font-size:115%;color:#000000;">Edit Matlab startup.m to contain the following:</span>

<span style="color:#000000;"> </span>

<span style="color:#000000;">2\. In Matlab command terminal, edit ecog_preprocessing.m</span>

<span style="color:#000000;">3\. Read through the comments carefully as they explain what to do step by step</span>

<span style="color:#000000;">4\. Each section of ecog_preprocessing.m is meant to be run one at a time, you can high the section and press F9 to run or click the section then hit this button</span>

<span style="font-family:Arial;font-size:115%;color:#000000;"> </span>