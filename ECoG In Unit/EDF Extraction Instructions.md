---
title: EDF Extraction
has_children: false
parent: ECoG In Unit
---
<span style="text-decoration:underline;"><span style="font-size:115%;">**Follow:**</span></span>

[<span style="font-size:115%;">https://citrix.duke.edu/Citrix/DukeWeb/</span>](https://citrix.duke.edu/Citrix/DukeWeb/)

<span style="font-size:115%;color:#ff0000;">***Clipping and exporting EDFs is a slow process- allow all windows to fully load before continuing each step***</span>

<span style="font-size:115%;"><span style="font-family:'Times New Roman';">Once logged into NATUS, wait for file </span><span style="font-family:'Times New Roman';">synchronization to fully complete (loading bar is located at lower right side of window)</span></span>

<span style="text-decoration:underline;"><span style="font-size:115%;">**Locating Patient files**</span><span style="font-size:115%;">**:**</span></span>

<span style="font-family:'Times New Roman';font-size:115%;">By pressing the "Search" icon or using the search bar on the left, only **select areas** indicated by red arrows, **enter pt name**, **search.**</span>

![](tempsnip.png)

<span style="font-family:'Times New Roman';font-size:115%;">Once results have fully loaded:</span>

1.  <span style="font-family:'Times New Roman';font-size:115%;">Sort files by "Start time" so that <span style="text-decoration:underline;">most recent files are at the top</span>.</span>
2.  <span style="font-family:'Times New Roman';font-size:115%;">Press ![](folder.JPG) to sort by file type </span>

<span style="font-family:'Times New Roman';">![](ODF.JPG)O</span>**riginal** <span style="font-family:'Times New Roman';">data files      ![](cut%20data%20files.JPG)</span><span style="font-family:'Times New Roman';"> Clipped data files   </span>

<span style="font-family:'Times New Roman';font-size:115%;">*Make sure to extract from ORIGINAL data files only*</span>

<span style="font-family:'Times New Roman';font-size:115%;">Refer to **Neurology_TaskData_Timestamps.slxs** on box for tasks/dates/times </span>

<span style="font-family:'Times New Roman';font-size:115%;">EDFs are recorded in 24hr epochs- select 24hr window in which specific task occurred using "Start time" and "End time"</span>

<span style="font-family:'Times New Roman';font-size:115%;">     *You cannot clip from a live EDF recording. Icon must be blue- shown above. If red, recording is live. </span>

<span style="font-family:'Times New Roman';font-size:115%;">Double click to open</span>

_____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

<span style="text-decoration:underline;"><span style="font-size:115%;">**Clipping EDF:**</span></span>

<span style="font-size:115%;"><span style="font-family:'Times New Roman';">Shown below is the NATUS screen:</span></span>

*   <span style="font-size:115%;"><span style="font-family:'Times New Roman';">The **top** box and arrow indicate the **date and time** of day selected</span></span>
*   <span style="font-size:115%;"><span style="font-family:'Times New Roman';"> </span></span>The **bottom** <span style="font-family:'Times New Roman';font-size:115%;">box and arrow show the **time bar**</span>

<span style="font-family:'Times New Roman';font-size:115%;"></span>

![](NATUS%20screen.JPG)

<span style="font-family:'Times New Roman';font-size:115%;">By clicking along the **time bar** you can select a specific time- refer to **Neurology_TaskData_Timestamps.slxs** for start and end times</span>

*   <span style="font-family:'Times New Roman';font-size:115%;">Once desired **start** time has been reached, select ![](start.JPG) icon to begin clip</span>

<span style="font-family:'Times New Roman';font-size:115%;">Once the play icon is pressed, a window will pop up:</span>

<span style="font-family:'Times New Roman';font-size:115%;">![](clip%201.JPG) </span>

*   <span style="font-family:'Times New Roman';font-size:115%;">In Note Details type **'D## taskdate COGAN_TASKNAME'** and hit OK</span>

<span style="font-family:'Times New Roman';font-size:115%;">Navigate along the bottom time bar to the desired **end time** and press![](end.JPG)  to end clip</span>

<span style="font-family:'Times New Roman';font-size:115%;">Time bar should now show yellow indication of clipped area</span>

![](time%20bar.JPG)

<span style="font-family:'Times New Roman';font-size:115%;">Clipped area must now be pruned and extracted.</span>

_____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

<span style="text-decoration:underline;">**<span style="font-size:115%;">Pruning the Clip:</span>**</span>

<span style="font-family:'Times New Roman';font-size:115%;">Press ![](prune%20tool.JPG) to prune the clip you just created</span>

<span style="font-size:115%;font-family:'Times New Roman';">A window should pop up</span>

<span style="font-size:115%;">![](prune%20window.JPG) </span>

*   <span style="font-size:115%;font-family:'Times New Roman';">*Multiple clips may automatically be selected. </span>
*   <span style="font-family:'Times New Roman';font-size:115%;">*Make sure to select ONLY the clip you just created- do this by double checking start and end times</span>
*   <span style="font-family:'Times New Roman';font-size:115%;">*Only prune one task clip at a time</span>
*   <span style="font-family:'Times New Roman';font-size:115%;">*Uncheck box in camera column- **ONLY scissors box** should be selected  </span>

*   <span style="font-family:'Times New Roman';font-size:115%;">Press "Prune..." button</span>

<span style="font-family:'Times New Roman';font-size:115%;">A window should pop up</span>

![](prune%2022.JPG)

*   <span style="font-size:115%;font-family:'Times New Roman';">The Study Name should match the input from 'note details' entered earlier- double check to make sure **D#, taskdate, task** are correct</span>
*   <span style="font-size:115%;font-family:'Times New Roman';">Press 'Ok'- this will take about 15 minutes to complete.</span>

<span style="font-size:115%;font-family:'Times New Roman';">Once complete, press 'Ok' and then 'Close' the pruning window</span>

<span style="font-family:'Times New Roman';font-size:115%;">Close out of the current NATUS window.</span>
<span style="font-family:'Times New Roman';font-size:115%;">When prompted, click 'Yes' to save changes</span>

<span style="font-family:'Times New Roman';font-size:115%;">Once back at the patient window, organize by "Study Name"</span>

*   <span style="font-family:'Times New Roman';font-size:115%;">The files just pruned should be visible    </span>

_____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

<span style="text-decoration:underline;"><span style="font-size:115%;">**Deidentify and Exporting:**</span></span>

<span style="font-family:'Times New Roman';font-size:115%;">*EDF cannot be exported to local drive, can only be exported to network drive.</span>

<span style="font-family:'Times New Roman';font-size:115%;">Right click on desired file and select 'Export'</span>

<span style="font-family:'Times New Roman';font-size:115%;">A window will pop up:</span>

<span style="font-family:'Times New Roman';font-size:115%;">![](edf%20export%2011.jpg)
</span>

*   <span style="font-family:'Times New Roman';font-size:115%;">Select EDF/EDF+</span>
*   <span style="font-family:'Times New Roman';font-size:115%;">Select destination for file to save to </span>
*   <span style="font-family:'Times New Roman';font-size:115%;">Right click on file under 'Template' and select 'Edit Template...'</span>

<span style="font-family:'Times New Roman';font-size:115%;">A window will pop up:</span>

![](edf%20edit%20options.jpg) 

*   <span style="font-family:'Times New Roman';font-size:115%;">Name: DEIDENTIFIED EDF </span>
*   <span style="font-family:'Times New Roman';font-size:115%;">Select:</span>

*   <span style="font-family:'Times New Roman';font-size:115%;">'Options' tab</span>
*   <span style="font-family:'Times New Roman';font-size:115%;">EDF+</span>
*   <span style="font-family:'Times New Roman';font-size:115%;">Deidentify patient information    </span>

*   <span style="font-family:'Times New Roman';font-size:115%;">Unselect 'Invert AC channels' </span>
*   <span style="font-family:'Times New Roman';font-size:115%;">Press 'Ok' </span>

<span style="font-family:'Times New Roman';font-size:115%;">Press 'Export' </span>

<span style="font-family:'Times New Roman';font-size:115%;">Once file has exported to network drive, rename in format "D# TaskDate COGAN_TASKNAME.edf</span>

<span style="font-family:'Times New Roman';font-size:115%;">Upload to box **<span style="color:#000000;font-family:'Times New Roman';font-size:115%;">Box Sync\CoganLab\ECoG_Task_Data\TaskUploadDir </span>**</span>

<span style="color:#000000;font-family:'Times New Roman';">*TaskDate should be in format YYMMDD</span>

<span style="color:#000000;font-family:'Times New Roman';">____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________</span>

**<span style="text-decoration:underline;"><span style="font-size:115%;">Archived EDF on NATUS</span></span>**

<span style="font-family:'Times New Roman';font-size:115%;">To access older participant EDF on NATUS, follow previous instructions. Double click on desired EDF, when prompted with: </span>

<span style="font-family:'Times New Roman';font-size:115%;">![](archive%20step%201.PNG)
</span>

<span style="font-family:'Times New Roman';font-size:115%;">The archive location should be auto-filled. Press 'Ok'.</span>

<span style="font-family:'Times New Roman';font-size:115%;">A second window will pop up:</span>

<span style="font-family:'Times New Roman';font-size:115%;">![](archive%20step%202.PNG)
</span>

<span style="font-family:'Times New Roman';font-size:115%;">Select 'No'.</span>

<span style="font-size:115%;font-family:'Times New Roman';">Continue with pruning and extraction instructions above. This should be the only difference when accessing archived EDF. </span>

________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

**<span style="text-decoration:underline;"><span style="font-size:115%;">Troubleshooting</span></span>**

<span style="font-size:115%;"><span style="font-family:mceinline;">NATUS can be a fickle beast. Sometimes, for seemingly unknown reasons, it will fail to produce participant data from the search menu. In this case, close out of NATUS and Citrix and re-log-in. This *should* fix it. If not, you will need to submit an IT ticket.</span></span>

<span style="font-family:'Times New Roman';font-size:115%;"> </span>

<span style="font-family:'Times New Roman';font-size:115%;"> </span>