function preProcess_ChannelOutlierRemoval(Task, Subject)
% Task is a structure with .Name = Task Name

global BOX_DIR
global RECONDIR
global TASK_DIR
global experiment
global DUKEDIR
%BOX_DIR=DirStruct.BOX_DIR;%'C:\Users\gcoga\Box';
%RECONDIR=DirStruct.RECONDIR;%'C:\Users\gcoga\Box\ECoG_Recon';

%addpath(genpath([BOX_DIR '\CoganLab\Scripts\']));

TASK_DIR=Task.Directory;
%TASK_DIR=([BOX_DIR '\CoganLab\D_Data\' Task.Name]);
DUKEDIR=TASK_DIR;


% Populate Subject file
if ~exist('Subject','var')
    Subject = popTaskSubjectData(Task);
end



for iSN=1:length(Subject)
    load([TASK_DIR '/' Subject(iSN).Name '/mat/experiment.mat']);
    load([TASK_DIR '/' Subject(iSN).Name '/' Subject(iSN).Date '/mat/Trials.mat'])
   
    % find bad channels based on 
    if ~exist([TASK_DIR '/' Subject(iSN).Name '/badChannels.mat'])
        badChannels = channelOutlierRemoval(Subject(iSN),Task,Trials,Task.Outlier.Field);
        save([TASK_DIR '/' Subject(iSN).Name '/badChannels.mat'],'badChannels');
        Subject(iSN).badChannels=badChannels;
    end    
end

