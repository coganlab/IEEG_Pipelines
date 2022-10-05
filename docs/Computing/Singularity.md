---
title: Singularity
has_children: false
parent: Computing
nav_order: 1
---
Singularity is a Linux software that allows application-level isolation. You can build a container filled with whatever applications you wish outside of the duke cluster, then move the container to the cluster to run the programs.

<span style="background-color:#ccffff;">Commands to be run outside of Singularity container</span>

<span style="background-color:#ffff99;">Commands to be run inside container</span>

duke compute cluster is running singularity 2.3.2 as of 10/30/2017\. (Duke cluster upgraded to 2.4 in 12/17\. These build instructions should still work, but now you can run 2.4 containers on the server. See email below.)

New versions of singularity use build instead of create & bootstrap commands. Just amend some of the commands below, but the overall procedure will be the same.

install that version on a computer you have root access to (VMBox works)

<span style="background-color:#ccffff;">sudo singularity create -s 12000 deb.simg</span> (size in MB, you can expand the container, so start small and add more space as needed)

<span style="background-color:#ccffff;">sudo singularity bootstrap deb.simg spec</span>

where spec contains the recipe

<span style="background-color:#ffcc99;">Bootstrap:docker</span>

<span style="background-color:#ffcc99;">From: debian:jessie</span>

<span style="background-color:#ffcc99;">%post</span>

<span style="background-color:#ffcc99;">    mkdir /dscrhome</span>

<span style="background-color:#ffcc99;">    mkdir /global</span>

<span style="background-color:#ffcc99;">    mkdir /matlab</span>

/dscrhome is required for singularity on duke computer cluster

in host, download matlab zip from mathworks

<span style="background-color:#ccffff;">mkdir global</span>, and put zip there

<span style="background-color:#ccffff;">sudo singularity shell --writable -B global:/global deb.simg</span>

Note, X system should be working, you can test this by running xeyes, xcalc, or xclock.

If these aren't installed, install x11-apps

You should see these gui apps running. If not, try binding the home directory, <span style="background-color:#ccffcc;">sudo singularity shell --writable -B global:/global -B /home/:/home/ deb.simg</span>

%% now inside container

<span style="background-color:#ffff99;">apt-get update</span>

<span style="background-color:#ffff99;">apt-get install xorg unzip</span>

<span style="background-color:#ffff99;">mv /global/matlab.zip /matlab</span>

<span style="background-color:#ffff99;">cd /matlab</span>

<span style="background-color:#ffff99;">unzip matlab.zip</span>

<span style="background-color:#ffff99;">./install</span>

Don't activate Matlab quite yet, unless you need to test the installation.

complete installation.

We must make licenses folder write enabled for all, for manual activation on the duke cluster

<span style="background-color:#ffff99;">chmod -R 777 /usr/local/MATLAB/R2017b</span>

<span style="background-color:#ffff99;">chmod 777 /dscrhome</span>

<span style="background-color:#ffff99;">chmod 777 /global</span>

exit container

<span style="background-color:#ccffff;">sudo tar cjvf deb.simg.tar.bz2 deb.simg</span> (slower, smaller file)

or

<span style="background-color:#ccffff;">sudo tar czvf deb.simg.tar.gzip deb.simg</span> (much faster, larger file)

move tar to pace system using browswer ftp,

pacedata.duhs.duke.edu/

log into duke cluster using <span style="background-color:#ccffff;">ssh -X</span> to enable X forwarding

make sure pace storage is mounted (see [PACE](/w/page/120968496/PACE)) and you can see coganlab/ folder.

<span style="background-color:#ccffff;">cd coganlab/</span>

<span style="background-color:#ccffff;">tar xvf deb.simg.tar.bz2 to uncompress into coganlab/ folder</span>

<span style="background-color:#ccffff;">singularity shell --writable deb.simg</span>

<span style="background-color:#ffff99;">cd /usr/local/MATLAB/R2017b/bin</span>

<span style="background-color:#ffff99;">./activation_matlab.sh</span>, choose license.lic file.

If you don't have license file, you need to create one through mathworks.com. Click the help button

and it will give you information on how to obtain license through mathworks.com.

(includes hostname ID, which I believe is just mac address of network interface)

ip link show, look at link/ether string under eth0 device

e.g. 005056aeb9c8

Upload that license.lic file to pace. Then re-run activation_matlab.sh and point to that license file.

You may need to launch the shell with the --writable flag from now on, e.g.  <span style="background-color:#ccffff;">singularity shell --writable deb.simg</span>

At some point you might wish to downsize your container,

while in container, run <span style="background-color:#ffff99;">df /</span> to see how much space you are using (add some buffer space, maybe 2 gb)

create a new container with desired size

<span style="background-color:#ccffff;">sudo singularity create -s 10000 deb2.simg</span>

then pipe the content from old to new

<span style="background-color:#ccffff;">sudo singularity export deb.simg | sudo singularity import deb2.simg</span>

To uninstall singularity, run the following:

<span style="background-color:#ccffff;">sudo rm -rf /usr/local/libexec/singularity</span>

<span style="background-color:#ccffff;">sudo rm -rf /usr/local/etc/singularity</span>

<span style="background-color:#ccffff;">sudo rm -rf /usr/local/include/singularity</span>

<span style="background-color:#ccffff;">sudo rm -rf /usr/local/lib/singularity</span>

<span style="background-color:#ccffff;">sudo rm -rf /usr/local/var/lib/singularity/</span>

<span style="background-color:#ccffff;">sudo rm /usr/local/bin/singularity</span>

<span style="background-color:#ccffff;">sudo rm /usr/local/bin/run-singularity</span>

<span style="background-color:#ccffff;">sudo rm /usr/local/etc/bash_completion.d/singularity</span>

<span style="background-color:#ccffff;">sudo rm /usr/local/man/man1/singularity.1</span>

=============================================

<span style="font-size:100%;">All:</span>

<span style="font-size:100%;">You’re getting this email because at one time or another you used the CI process on gitlab.oit.duke.edu with the DCC CI Runner to build a Singularity container. This runner used version 2.3.1 of Singularity. At the end of August Docker changed the format of their Docker containers on Docker Hub. This broke Singularity version 2.3.1\. Singularity released a version 2.3.2 (which is installed on most of the servers) that was able to use the new format. Unfortunately, I didn’t catch that the version on the runner stayed the same so any builds that used a container with the new format failed. Everyone must have already had their working containers built as there were no complaints till this week. We’ve updated the DCC GitLab CI Runner to version 2.4.1 of Singularity. I will be pushing that version out to the servers that have Singularity installed. There’s no need to rebuild your containers if they are working, version 2.4.1 will run containers built with an earlier version.</span>

<span style="font-size:100%;">This updated version of Singularity has some new features (see [http://singularity.lbl.gov/index.html](http://singularity.lbl.gov/index.html) for documentation). One new feature is that the containers now use Squashfs. This means you don’t have to prebuild an image that you install into the new build command will create the container with the required space. It also makes the container read only. If you need a writable container you can convert it. Also, Singularity has changed their recommended naming convention for the definition files from Singularity.def to just Singularity. With these new features in mind the CI build scripts were modified. Since Squashfs is already compressed there is no longer a need to compress the built image. If you use the new process you can just curl or wget the built image directly and then you won’t have to extract it from a “.tar.gz” anymore. If you have old projects that you don’t want to modify they will still work, it’s just the “.tar.gz” file will not be that different in size than the extracted container. The following is the recommended “.gitlab-ci.yml” to put in your projects:</span>

<span style="font-size:100%;">stages:</span>

<span style="font-size:100%;">  - build</span>

<span style="font-size:100%;">  - deploy</span>

<span style="font-size:100%;">build_image:</span>

<span style="font-size:100%;">  stage: build</span>

<span style="font-size:100%;">  script:</span>

<span style="font-size:100%;">    - build_image</span>

<span style="font-size:100%;">  tags:</span>

<span style="font-size:100%;">    - dcc</span>

<span style="font-size:100%;">  artifacts:</span>

<span style="font-size:100%;">    paths:</span>

<span style="font-size:100%;">      - $CI_PROJECT_NAME.img</span>

<span style="font-size:100%;">    expire_in: 1 day</span>

<span style="font-size:100%;">deploy_image:</span>

<span style="font-size:100%;">  stage: deploy</span>

<span style="font-size:100%;">  script:</span>

<span style="font-size:100%;">    - deploy_output</span>

<span style="font-size:100%;">  tags:</span>

<span style="font-size:100%;">    - dcc</span>

<span style="font-size:100%;">Mike Newton</span>