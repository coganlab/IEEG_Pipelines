---
title: Instructions
parent: Preprocessing
has_children: false
---
# Intructions
<span style="text-decoration:underline;"><span style="font-size:115%;">**<span style="font-family:Calibri, sans-serif;">Preprocessing Set Up:</span>**</span></span>

Create folder “<span style="color:#c00000;">InUnit Preprocessing</span>” to house all preprocessing files:

![](/f/1631896590/pp1.png)

Within “InUnit Preprocessing”:

*   <span style="font-family:Symbol;"><span style="font-family:'Times New Roman';">     </span></span>Create folders for each of the participants <span style="color:#4472c4;">(D##)</span> you will be preprocessing
*   <span style="font-family:Symbol;"><span style="font-family:'Times New Roman';">      </span></span>Download <span style="color:#70ad47;">maketrigtimes</span> and <span style="color:#70ad47;">view_triggers</span> matlab scripts from box

*   <span style="color:#70ad47;">maketrigtimes</span>: Box Sync\ECoG_Recon\matlab_code\ecog_preprocessing
*   <span style="color:#70ad47;">view_triggers</span>: Box Sync\CoganLab\ECoG_Task_Data\TaskUploadDir
*   <span style="color:#ff6600;">Startup.mat</span> will also be saved here. See page 2 for details

![](/f/1631896590/pp2.png)

Within the participant folder, create folders for each <span style="color:#bf8f00;">task</span> completed/to be preprocessed

<span style="text-decoration:underline;"><span style="font-size:115%;">**<span style="font-family:Calibri, sans-serif;">ECoG</span>****<span style="font-family:Calibri, sans-serif;"> preprocessing:</span>**</span></span>

*   <span style="font-size:100%;font-family:Calibri, sans-serif;">Setup your </span><span style="font-size:100%;font-family:Calibri, sans-serif;">Matlab</span><span style="font-size:100%;font-family:Calibri, sans-serif;"> to include the following:</span>

<span style="font-size:100%;font-family:Calibri, sans-serif;"> </span>

![](/f/1631896588/pp3.png)<span style="font-size:100%;font-family:Calibri, sans-serif;"> </span>

<table border="0" cellspacing="0" cellpadding="0" width="100%">

<tbody>

<tr>

<td>

<span style="font-family:'Courier New';color:#ed7d31;">globalFsDir will need to be set specific to your own computer</span>

</td>

</tr>

</tbody>

</table>

**<span style="font-size:100%;font-family:Calibri, sans-serif;">COPY below:</span>**

<span style="font-size:100%;font-family:'Courier New';">global </span><span style="font-size:100%;font-family:'Courier New';">globalFsDir</span><span style="font-size:100%;font-family:'Courier New';">;</span>

<span style="font-size:100%;font-family:'Courier New';">global RECONDIR;</span>

<span style="font-size:100%;font-family:'Courier New';">global RECONDIR_FULL;</span>

<span style="font-size:100%;font-family:'Courier New';">global TASKSTIM;</span>

<span style="font-size:100%;font-family:'Courier New';">RECONDIR = 'E:\Box Sync\Box Sync\</span><span style="font-size:100%;font-family:'Courier New';">ECoG_Recon</span><span style="font-size:100%;font-family:'Courier New';">';</span>

<span style="font-size:100%;font-family:'Courier New';">RECONDIR_FULL = 'E:\Box Sync\Box Sync\</span><span style="font-size:100%;font-family:'Courier New';">CoganLab</span><span style="font-size:100%;font-family:'Courier New';">\</span><span style="font-size:100%;font-family:'Courier New';">ECoG_Recon_Full</span><span style="font-size:100%;font-family:'Courier New';">';</span>

<span style="font-size:100%;font-family:'Courier New';">TASKSTIM = 'E:\Box Sync\Box Sync\</span><span style="font-size:100%;font-family:'Courier New';">CoganLab</span><span style="font-size:100%;font-family:'Courier New';">\</span><span style="font-size:100%;font-family:'Courier New';">task_stimuli</span><span style="font-size:100%;font-family:'Courier New';">';</span>

<span style="font-size:100%;font-family:'Courier New';">%</span><span style="font-size:100%;font-family:'Courier New';">globalFsDir</span><span style="font-size:100%;font-family:'Courier New';"> = 'c:/users/pa112/Desktop/xxxx/</span><span style="font-size:100%;font-family:'Courier New';">xxxxxx</span><span style="font-size:100%;font-family:'Courier New';">';</span>

<span style="font-size:100%;font-family:'Courier New';">addpath</span><span style="font-size:100%;font-family:'Courier New';">(</span><span style="font-size:100%;font-family:'Courier New';">genpath</span><span style="font-size:100%;font-family:'Courier New';">(</span><span style="font-size:100%;font-family:'Courier New';">fullfile</span><span style="font-size:100%;font-family:'Courier New';">(RECONDIR, '</span><span style="font-size:100%;font-family:'Courier New';">matlab_code</span><span style="font-size:100%;font-family:'Courier New';">')));</span>

<span style="text-decoration:underline;"><span style="font-size:100%;font-family:Calibri, sans-serif;">This code will need to be run before beginning EACH preprocessing session</span></span>

<span style="font-size:115%;font-family:Calibri, sans-serif;">**<span style="text-decoration:underline;">To Begin Preprocessing:</span>**</span>

*   **<span style="font-size:100%;font-family:Calibri, sans-serif;">Download EDF file</span>** <span style="font-size:100%;font-family:Calibri, sans-serif;">(location: Box Sync\</span><span style="font-size:100%;font-family:Calibri, sans-serif;">CoganLab</span><span style="font-size:100%;font-family:Calibri, sans-serif;">\</span><span style="font-size:100%;font-family:Calibri, sans-serif;">ECoG_Task_Data</span><span style="font-size:100%;font-family:Calibri, sans-serif;">\</span><span style="font-size:100%;font-family:Calibri, sans-serif;">TaskUploadDir</span><span style="font-size:100%;font-family:Calibri, sans-serif;">) for the participant/task you will be preprocessing</span>

*   <span style="font-size:100%;font-family:Calibri, sans-serif;">Place EDF file into designated participant (D##)/ task folder from page 1</span>

*   **<span style="font-size:100%;font-family:Calibri, sans-serif;">Download **<span style="font-size:100%;font-family:Calibri, sans-serif;">trialdata.mat</span>** </span>**<span style="font-size:100%;font-family:Calibri, sans-serif;"> for same participant/</span><span style="font-size:100%;font-family:Calibri, sans-serif;">task (location: Box Sync\CoganLab\ECoG_Task_Data\Cogan_Task_Data)  </span>

*   Place into same folder as EDF (TIMIT preprocessing requires all trialdata.mat files for <span style="text-decoration:underline;">each block and practice</span> <span style="font-family:Calibri, sans-serif;font-size:100%;">to be downloaded)</span>

1.  <span style="font-size:100%;font-family:Calibri, sans-serif;">In </span><span style="font-size:100%;font-family:Calibri, sans-serif;">matlab: make current folder desired partcicipant (D##)>task folder</span>
2.  <span style="font-size:100%;font-family:Calibri, sans-serif;">Run startup.mat</span>
3.  <span style="font-size:100%;font-family:Calibri, sans-serif;">Open ecog_preprocessing.m</span>

**<span style="font-size:100%;font-family:Calibri, sans-serif;">Each section of </span>****<span style="font-size:100%;font-family:Calibri, sans-serif;">ecog_preprocessing.m</span>****<span style="font-size:100%;font-family:Calibri, sans-serif;"> is meant to be run one at a time, you can highlight the section and press F9 to run.</span>**

<span style="font-family:Calibri, sans-serif;">*Add a new </span><span style="font-family:'Courier New';color:#0000ff;">case</span><span style="font-family:Calibri, sans-serif;"> for each task/recording sections</span><span style="font-family:Calibri, sans-serif;">:</span>

![](/f/1631896588/pp4.png)

*   <span style="font-size:100%;font-family:Calibri, sans-serif;">Copy and paste the most recent</span> <span style="font-size:100%;font-family:'Courier New';color:#0000ff;">case</span> <span style="font-family:Calibri, sans-serif;">(example figure <span style="color:#ff0000;">A</span>) and replace purple text with current participant information and file locations.</span>
*   <span style="font-family:Calibri, sans-serif;">Once filled in, highlight lines ‘cd’ to ‘rec’ and press F9 to run.</span>

<span style="font-family:Calibri, sans-serif;">Scroll down, highlight and run (shown in figure </span>**<span style="color:#4472c4;">B</span>**<span style="font-family:Calibri, sans-serif;">):</span>

![](/f/1631896589/pp5.png)

<span style="font-family:Calibri, sans-serif;">Scroll down, highlight and run (shown in figure </span>**<span style="color:#ffd966;">C</span>**<span style="font-family:Calibri, sans-serif;">)</span>

![](/f/1631896589/pp6.png)

Variable “labels” will be created in workspace.

*   Check here to confirm <span style="font-size:100%;font-family:'Courier New';color:#000000;">trigger_chan_index, mic_chan_index, neural_chan_index</span> (final portion of example <span style="color:#ff0000;">A</span>)<span style="font-size:100%;font-family:'Courier New';color:#000000;">.</span>

*   <span style="font-size:100%;font-family:'Courier New';color:#000000;"> </span>Enter channel values into last 3 lines of example <span style="color:#ff0000;">A</span>, highlight and run(F9).

Use to double check number of triggers during maketrigtimes.m step

<span style="font-family:Calibri, sans-serif;">Number of triggers:</span>

<span style="font-family:Calibri, sans-serif;">Timit- 336</span>

<span style="font-family:Calibri, sans-serif;">Delay- 336</span>

<span style="font-family:Calibri, sans-serif;">No Delay- 504</span>

<span style="font-family:Calibri, sans-serif;">Uniqueness Point- 480</span>

<span style="font-family:Calibri, sans-serif;">Sentence Rep- 270</span>

<span style="font-family:Calibri, sans-serif;">PhonemeS- 208</span>

<span style="font-family:Calibri, sans-serif;">Environmental- 1176</span>

<span style="font-family:Calibri, sans-serif;">Neighborhood- 1120</span>