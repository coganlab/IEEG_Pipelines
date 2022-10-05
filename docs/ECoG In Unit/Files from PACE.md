---
title: PACE
has_children: false
parent: ECoG In Unit
---
# PACE

<span style="color:#5a5a5a;">Protected Analytics Computing Environment (PACE) is a highly protected virtual network space that serves as a marketplace where approved users can work with identifiable protected health information. PACE simplifies the effort of obtaining EHR (Electronic Health Record) data from Duke Health enterprise data warehouse and Duke's Maestro Care (Epic) EHR system, while supporting collaborators from around the world with approved NetIDs. The marketplace offers a rich set of tools, services, and resources required by research and quality initiatives. Within the protected enclave, PACE users are provided the ability to select operating systems, analytic tools (e.g. R, SAS, Python), services (e.g. an Honest Broker or Transfer Agent service to securely release data outside of PACE), compute and data sources (e.g. Microsoft Azure, Exadata, OIT GPU, DEDUCE). Future expansion of the marketplace will include access to cloud offerings like Amazon AWS and Google Cloud Platform.</span>

PACE website: [https://pace.ori.duke.edu/](https://pace.ori.duke.edu/)

## Setting up a PACE account (Duke Health Employee)

1.  Register for the "PACE training for Duke Employees and Affiliates" course in Duke LMS. You will need to score at least 80% on the exam in order to earn course completion. 
2.  <span style="color:#5a5a5a;">Obtain a CoreResearch@Duke Service Request ID </span>

1.  <span style="color:#5a5a5a;">As of January 1, 2021, PACE has begun billing for services using </span><span style="text-decoration:underline;"><span style="color:#0000ff;">[CoreResearch@Duke](https://coreresearch.duke.edu/ "CR@D")</span></span><span style="color:#5a5a5a;">, Duke's enterprise-wide shared resource request system. When requesting new PACE resources via GetIT, you will need to provide a CoreResearch Service Request ID. If you are not familiar with how to meet this requirement for your project, you may need to contact your PI, PI Delegate, Financial or Grant Manager for assistance.  </span>

4.  Submit a PACE Account Request

1.  Provide IRB #, RPR, QI, or equivalent number at time of request (requester must be Key Personnel on an IRB study with an approved or exempt status)  

## Accessing the PACE Environment

1.  Login to secure.citrix.duke.edu with your  netID
2.  Click the DESKTOPS button in the top banner
3.  Click the PACE - SIMPLE icon to launch the environment 

## Software to load in PACE

1.  MicroDicom is a free software that can view CT and MRI images 
2.  As a first time user of PACE, request from the PACE admin to save a .zip of MicroDicom application on the PACE software share drive (<span style="font-size:100%;color:#1f497d;">S:\Open Source\Windows\MicroDicom\)</span>
3.  Save MicroDicom as a desktop shortcut in PACE 

## Viewing MRI and CT files in PACE

1.  Open MicroDicom
2.  File > Scan for DICOM files and point to the mri/D# directory 
3.  Identify which scan to use for the MRI reconstruction and CT 
4.  Check that identifiers have been removed (top left corner) 
5.  Export the series 

1.  File > Export > To DICOM file...
2.  Specify Destination and Filename prefix
3.  Select source: Current series
4.  Check box: Separate files for every Image
5.  Image size original 

## Moving files out of the PACE Environment

1.  Based on the specified destination of exported files, .zip the series and rename with Subject ID and Series name (i.e. MRI or CT)
2.  Move the new .zip folder to Pro00065476-TA-[netID]
3.  In Windows OS login to pacedata.duhs.duke.edu with netID credentials and 2FA
4.  Click on the Pro00065476-TA-[netID] folder and download the folder
5.  This .zip folder will disappear from the Windows OS and PACE environment after download 

## File Upload

1.  Downloaded .zip files can be uploaded to Duke Box > MRI
2.  Send notification to relevant parties on new files