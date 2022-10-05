---
title: Data Analysis
has_children: false
parent: EEG
nav_order: 1
---
If your system is already setup - please traverse to the Source localization pipeline ([Pipeline / Cogan Lab (pbworks.com)](/w/page/146866377/Pipeline))

<span style="font-size:200%;">System Setup:</span>

       <span style="font-size:150%;">Running both Python MNE and FreeSurfer on Windows requires a WSL (Windows Subsystem Linux)<span style="font-family:mceinline;">:</span></span>

1.  <span style="font-size:150%;"><span style="font-family:mceinline;"> FreeSurfer runs on all OS's besides Windows, to accomodate for this we need to use a WSL</span></span>

1.  <span style="font-size:130%;"><span style="font-family:mceinline;"><span style="font-family:mceinline;">A WSL is a Linux virtual machine that runs simultaneously on a Windows machine with shared file systems</span></span></span>
2.  <span style="font-size:130%;"><span style="font-family:mceinline;"><span style="font-family:mceinline;">We want to do this so that python MNE can reference the files containing our FreeSurfer reconstructions - using the Environment Variables (paths) created by the Freesurfer Environment</span></span></span>

3.  <span style="font-size:150%;">Install WSL</span>

1.  <span style="font-size:130%;">if you are not a _____ user you will have to do the following steps (ii --> vi)</span>
2.  <span style="font-size:130%;">Changing your settings</span>
3.  <span style="font-size:130%;">Changing your Securities</span>
4.  <span style="font-size:130%;">obtaining a distribution </span>

5.  <span style="font-size:130%;">Installing X-Server</span>

1.  <span style="font-size:130%;">Installing Xserver</span>

<span style="font-size:130%;"> </span>

1.  <span style="font-size:130%;">Installing ___ to be able to traverse/view your WSL file directory</span>

1.  <span style="font-size:130%;"> </span>

3.  <span style="font-size:130%;">Testing your GUI</span>

1.  <span style="font-size:130%;">FSL - MRI viewer </span>

5.  <span style="font-size:130%;">Install FreeSurfer on the WSL</span>

1.  <span style="font-size:130%;">Traverse to the Freesurfer website and select a Linux distribution</span>
2.  <span style="font-size:130%;">Copy the link to the distribution</span>
3.  <span style="font-size:130%;">wget the link </span>

7.  <span style="font-size:130%;">Install miniconda or anaconda on the WSL </span>

1.  <span style="font-size:130%;">After installation make sure to say "yes" when prompted on whether you want conda or anaconda to initialize in your bash window </span>

9.  <span style="font-size:130%;">Pip install the MNE environment on the WSL</span>
10.  <span style="font-size:130%;">Acitvate the MNE environment</span>

1.  <span style="font-size:130%;">anaconda or conda activate mne </span>

12.  <span style="font-size:130%;">Make sure the Subject Directories are pointing to the correct locations</span>

1.  <span style="font-size:130%;">export SUBJECTS_DIRS & export FREESURFER_HOME - will change these pointers </span>

How to Modify your Bash window so that Conda or Anaconda Activates: