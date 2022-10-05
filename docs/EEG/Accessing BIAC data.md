---
title: Accessing BIAC data
has_children: false
parent: EEG
---

Refer to the [BIAC wiki](https://wiki.biac.duke.edu/biac:gettingstarted) for detail instructions for detail instructions for obtaining data and setting up a BIAC account.

**Setting up Moba X Term to access cluster**

1.  Login to the VPN using netID credentials
2.  Launch MobaXTerm

**Setting up the Session: SSH**

1.  Click Session in the top banner 
2.  Click SSH button in the top banner in the new window
3.  Remote host: cluster.biac.duke.edu
4.  Specify username: netID (must have approved BIAC access)
5.  A new tab will open in the Main terminal
6.  Login as: netID
7.  netID@cluster.biac.duke.edu's password: 
8.  After logging in, type 'qinteract' 
9.  This will mount to the interactive blade (this session will be valid for 4 days; Warning: permanently added '[blade17.dhe.duke.edu]' )

**Navigating to Cogan folder (EEGanat.01)**

1.  Type in command line: cd /mnt/munin2/Cogan/EEGanat.01/Data/Anat
2.  This folder houses the MRIs uploaded by BIAC
3.  Type 'ls' to see the list of folders of MRI scans (format: YYYYMMDD_Scan#) 
4.  If you are unsure of what the scan number is you can cross reference with the BIAC Subject calendar ([https://www.biac.duke.edu/calendar/](https://www.biac.duke.edu/calendar/)) and the Cogan Lab calendar. 

**Copying files to home directory**

1. To copy a folder to your home directory:

   1. cp -R YYYYMMDD_Scan# /home/netID/
   2. The left side panel will show the copied folder (may have to click the Refresh button) 

**Move files to Box**

1.  Download the folder (right click) 
2.  Move folder to  Box > CoganLab > EEG > Data > MRI_SubjectList
3.  Cross reference the Scan# (BIAC assigned) with the MRI Series Sequence.xlsx spreadsheet
4.  Rename the files to correspond with the subjectID from spreadsheet