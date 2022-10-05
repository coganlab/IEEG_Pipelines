---
title: Box Drive
has_children: false
parent: Computing
nav_order: 1
---
The following are the steps to port the physical storage location of box drive over to a secondary drive

1.  Close Box drive
2.  Move the cache folder (go to %USERPROFILE%\AppData\Local\Box\Box and move the 'cache' folder to your new location. I recommend you back up the current cache folder temporarily.)
3.  Create the symbolic link (Open command prompt as an administrator and then use the mklink command to make a link named 'cache' pointing to the new location - for example, mklink /D "C:\Users\<insert user root folder name>\AppData\Local\Box\Box\cache" "D:\<insert new cache folder name>"
4.  Restart Box