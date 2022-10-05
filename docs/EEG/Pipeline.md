---
title: Pipeline
has_children: false
parent: EEG
---
## **LINUX SETUP:**

1. <span style="font-family:'Times New Roman';">Install freesurfer and mne onto your ubuntu</span>

   1. <span style="font-family:'Times New Roman';">make sure that you change your freesurfer environment variables to the folders you want them to be (on the shared drive between the linux distrib and the PC)</span>

      1. <span style="font-family:'Times New Roman';">Example:</span>

         1. <span style="font-family:'Times New Roman';">export FREESURFER_HOME=/share/freesurfer </span>
         2. <span style="font-family:'Times New Roman';">export SUBJECTS_DIR=$FREESURFER_HOME/subjects </span>
         3. <span style="font-family:'Times New Roman';">source $FREESURFER_HOME/SetUpFreeSurfer.sh</span>
         4. <span style="font-family:'Times New Roman';">Recomended - Select Shared or Box Folder</span>

2. <span style="font-family:'Times New Roman';">Install Xserver follow this website: _______</span>
3. <span style="font-family:'Times New Roman';">Install anaconda-navigator or conda onto your linux distrib</span>
4. <span style="font-family:'Times New Roman';">Activate mne (conda activat mne)</span>
5. <span style="font-family:'Times New Roman';">Open the GUI for the navigator </span>
6. <span style="font-family:'Times New Roman';">Switch to the MNE driver</span>
7. <span style="font-family:'Times New Roman';">Open spyder and a jupyter qt console</span>
8. <span style="font-family:'Times New Roman';">Open a second Linux distrib to do coregistration</span>
9. <span style="font-family:'Times New Roman';">Copy and paste all code from document into jupyter qt console</span>
10. <span style="font-family:'Times New Roman';">Type %matplotlib qt --> allows for graphs to display</span>
11. <span style="font-family:'Times New Roman';">Get environment variables so mne can work with your FreeSurfer data</span>

    1. <span style="font-family:'Times New Roman';">a. FREESURFER_HOME = os.getenv('FREESURFER_HOME')     </span>
    2. <span style="font-family:'Times New Roman';">b. SUBJECTS_DIR = os.getenv('SUBJECTS_DIR')</span>

12. <span style="font-family:'Times New Roman';"> For 3D plotting do the following:</span>

    1. <span style="font-family:'Times New Roman';">mne.viz.set_3d_backend('pyvista')</span>
    2. <span style="font-family:'Times New Roman';">b. Export MNE_3D_OPTION_ANTIALIAS=false OR  mne.viz.set_3d_options(antialias=False)</span>

       1. <span style="font-family:'Times New Roman';">*Turns on antialiasing*</span>

## <span style="font-family:'Times New Roman';">**PIPELINE:**</span>

<span style="font-family:'Times New Roman';">1. Given your specific subject, create 2 new folders in their sub folder:</span>

*   <span style="font-family:'Times New Roman';">Digitization - used for Localization</span>
*   <span style="font-family:'Times New Roman';">Epochs - used for making the BEM</span>

<span style="font-family:'Times New Roman';">CLEAN DATA: *check out make_montage and change what files you traverse to based on your user and file structure*</span>

<span style="font-family:'Times New Roman';">1\. find Montage file (captrak) and load in EEG data and align it to correct channels</span>

*   <span style="font-family:'Times New Roman';">    montage, raw = make_montage()</span>

<span style="font-family:'Times New Roman';">2\. Common Average Reference and select noisy channels</span>

*   <span style="font-family:'Times New Roman';">    raw2 = set_data(raw, montage) #common average reference</span>

<span style="font-family:'Times New Roman';">3\. Save Captrak file to raw EEG file and then save it as a .fif</span>

*   <span style="font-family:'Times New Roman';">    raw2.save(pathlib.Path('/mnt/d/Aidan/' + patient + '/Digitization') / 'Captrak_Digitization.fif', overwrite = True)</span>

<span style="font-family:'Times New Roman';">4\. Check to see if you chose the right channels</span>

*   <span style="font-family:'Times New Roman';">    epochs = erp(raw2, montage)</span>

<span style="font-family:'Times New Roman';">5\. Run autorejection algorithms to detect noisy correlated epochs</span>

*   <span style="font-family:'Times New Roman';">    epochs_ar = autorej(epochs)</span>

<span style="font-family:'Times New Roman';">6\. Run an ICA on the autorejection data</span>

*   <span style="font-family:'Times New Roman';">    ica = ICA_auto_rej(epochs_ar)</span>
*   <span style="font-family:'Times New Roman';">    ica.plot_sources(raw2, show_scrollbars=True)     # to plot ICA vs time</span>

<span style="font-family:'Times New Roman';">7\. Remove ICA channels that correspond to artifacts</span>

*   <span style="font-family:'Times New Roman';">    ica.exclude = [0,1,2,5] # MANUAL INPUT</span>

<span style="font-family:'Times New Roman';">8\. Identify channels correlated with EOG data and remove them</span>

*   <span style="font-family:'Times New Roman';">    reconst_evoked, reconst_raw = EOG_check_ar(epochs_ar, ica)</span>

## <span style="font-family:'Times New Roman';">**Coregistration:**</span>

<span style="font-family:'Times New Roman';">1\. in qtconsole generate a bem</span>

*   <span style="font-family:'Times New Roman';">epoch_data = '/mnt/d/'INSERT USERNAME'/'INSERT PATIENT OR SUBJECT'/Epochs/epochs_for_source_epo.fif'</span>

*   <span style="font-family:'Times New Roman';">i.e. epoch_data = '/mnt/d/Aidan/E46/Epochs/epochs_for_source_epo.fif'</span>

*   <span style="font-family:'Times New Roman';">epoch_info = mne.io.read_info(epoch_data)</span>

*   <span style="font-family:'Times New Roman';">mne.bem.make_watershed_bem(subject=patient, subjects_dir=SUBJECTS_DIR, overwrite=True, volume='T1')</span>

<span style="font-family:'Times New Roman';">2\. Display bem</span>

*   <span style="font-family:'Times New Roman';">mne.viz.plot_bem(subject = patient, subjects_dir = SUBJECTS_DIR, orientation = 'coronal')</span>

<span style="font-family:'Times New Roman';">3\. Double Check your file has fiducials</span>

*   <span style="font-family:'Times New Roman';">mne.coreg.get_mni_fiducials('E2', SUBJECTS_DIR)</span>

<span style="font-family:'Times New Roman';">4.Now open a new Ubuntu and activate mne</span>

*   <span style="font-family:'Times New Roman';">    conda activate mne</span>

<span style="font-family:'Times New Roman';">5\. Perform Coregistration</span>

*   <span style="font-family:'Times New Roman';">type mne coreg into your Linux distribution</span>
*   <span style="font-family:'Times New Roman';">make sure your digitization bem files are the same</span>
*   <span style="font-family:'Times New Roman';">then click where your fiducials should be</span>
*   <span style="font-family:'Times New Roman';">then hit fit fiducials</span>

<span style="font-family:'Times New Roman';">6\. save the Coregistration data as a trans.fif file</span>

*   <span style="font-family:'Times New Roman';">hit the button on the coregistration gui (bottom right)</span>

## <span style="font-family:'Times New Roman';">**Localization: (you can do this half of the pipeline with epoched or evoked data, I used evoked)**</span>

<span style="font-family:'Times New Roman';">1\. generate Noise covariance and baseline evoked data</span>

*   <span style="font-family:'Times New Roman';">    noise_cov, fig_cov, fig_spectra, evoked = covariance()</span>

<span style="font-family:'Times New Roman';">2\. double checkyour patient is your current subject: i.e. E46</span>

*   <span style="font-family:'Times New Roman';">patient</span>

<span style="font-family:'Times New Roman';">3\. define a path to traverse to your subjects given digitization info</span>

*   <span style="font-family:'Times New Roman';">data_path = '/mnt/d/Aidan/' + patient + '/Digitization'</span>

<span style="font-family:'Times New Roman';">4\. save your digitization data to a specific variable</span>

*   <span style="font-family:'Times New Roman';">data = pathlib.Path(data_path) / 'Captrak_Digitization.fif'</span>

<span style="font-family:'Times New Roman';">5\. read in that data from the file</span>

*   <span style="font-family:'Times New Roman';">a. use baseline evoked</span>
*   <span style="font-family:'Times New Roman';">info = mne.io.read_info(data)</span>

<span style="font-family:'Times New Roman';">b. use the evoked data you have already cleaed</span>

*   <span style="font-family:'Times New Roman';">info = reconst_evoked.info</span>

<span style="font-family:'Times New Roman';">6\. read in your coregistration data</span>

*   <span style="font-family:'Times New Roman';">trans = pathlib.Path(data_path)/ 'E2-trans.fif'</span>

<span style="font-family:'Times New Roman';">7\. plot your coregistration data just to double check</span>

*   <span style="font-family:'Times New Roman';">fig = mne.viz.plot_alignment(info=info, trans=trans, subject=patient, subjects_dir=SUBJECTS_DIR, dig=True, verbose=True)</span>

<span style="font-family:'Times New Roman';">8\. compute your space (2D) - oct4, oct5 or oct6 are recommended (these determine the number of sources), you can also change the 'conductance'</span>

*   <span style="font-family:'Times New Roman';">src = mne.setup_source_space(subject=patient, spacing='oct5', subjects_dir = SUBJECTS_DIR)</span>

<span style="font-family:'Times New Roman';">9\. you can now display this alignment as well with your coregistration</span>

*   <span style="font-family:'Times New Roman';">mne.viz.plot_alignment(info=info, trans=trans, subject=patient, src=src, subjects_dir=SUBJECTS_DIR, dig=True)</span>

<span style="font-family:'Times New Roman';">10\. Generate a BEM model</span>

*   <span style="font-family:'Times New Roman';">model = mne.make_bem_model(subject=patient, subjects_dir=SUBJECTS_DIR)</span>

<span style="font-family:'Times New Roman';">11\. Create a BEM 'solution'</span>

*   <span style="font-family:'Times New Roman';">bem_sol=mne.make_bem_solution(model)</span>

<span style="font-family:'Times New Roman';">12\. Create a forward solution(mindist = ignore elcetrodes minimum distance from skull, n_jobs =  of jobs to run in parallel when computing, ignore_ref=True  can be used to ignore reference electrode (Common Average Reference))</span>

*   <span style="font-family:'Times New Roman';">fwd = mne.make_forward_solution(info, trans=trans, src=src, bem=bem_sol, meg=False, eeg=True, mindist=5.0, n_jobs=1)</span>

<span style="font-family:'Times New Roman';">13\. save your bem, and fwd solutions</span>

*   <span style="font-family:'Times New Roman';">bem_fname = pathlib.Path(data_path)/ 'E46_bem.fif'</span>
*   <span style="font-family:'Times New Roman';">mne.bem.write_bem_solution(bem_fname, bem_sol, overwrite = True)</span>
*   <span style="font-family:'Times New Roman';">fwd_fname = pathlib.Path(data_path)/'E46_fwd_soln.fif'</span>
*   <span style="font-family:'Times New Roman';">mne.write_forward_solution(fwd_fname, fwd, overwrite=True)</span>

<span style="font-family:'Times New Roman';">14\. Create inverse operator (optional variables: loose=?, depth=?)</span>

*   <span style="font-family:'Times New Roman';">Inv = mne.minimum_norm.make_inverse_operator(info, fwd, noise_cov)</span>

<span style="font-family:'Times New Roman';">15\. Now apply the inverse operator to gain your Inverse Solution, (recommended method = 'sLORETA' or 'eLORETA')</span>

*   <span style="font-family:'Times New Roman';">stc = mne.minimum_norm.apply_inverse(reconst_evoked, inv_reloaded, method = 'eLORETA')</span>

## <span style="font-family:'Times New Roman';">**Plotting:**</span>

<span style="font-family:'Times New Roman';">1\. Plot the inverse solution</span>

*   <span style="font-family:'Times New Roman';">brain = stc.plot()</span>

<span style="font-family:'Times New Roman';">2\. Use this variable to fine tune your plots</span>

*   <span style="font-family:'Times New Roman';">surfer_kwargs = dict(hemi = 'both', surface='pial', subjects_dir = SUBJECTS_DIR, clim=dict(kind='value', lims=[3.5e-11, 6e-11, 8e-11]), views='lateral', initial_time=0.19, time_unit='s', size=(800,800), smoothing_steps=5)</span>
*   <span style="font-family:'Times New Roman';">'surface' --> determines what parts of the brain will be viewed ('white', 'pial', 'inflated')</span>
*   <span style="font-family:'Times New Roman';">'lims' -->  dictionary of bounds of amplitude activation</span>
*   <span style="font-family:'Times New Roman';">'hemi' --> which hemisphere are you viewing</span>

<span style="font-family:'Times New Roman';">3\. replot</span>

*   <span style="font-family:'Times New Roman';">brain = stc.plot(**surface_kwargs)</span>