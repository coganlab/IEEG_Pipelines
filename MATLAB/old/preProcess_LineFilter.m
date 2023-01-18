function preProcess_LineFilter(Task)
% Task is a structure with .Name = Task Name

global BOX_DIR
global RECONDIR
global TASK_DIR
global experiment
global DUKEDIR
%BOX_DIR='C:\Users\gcoga\Box';
%RECONDIR=DirStruct.RECONDIR;%'C:\Users\gcoga\Box\ECoG_Recon';

%addpath(genpath([BOX_DIR '\CoganLab\Scripts\']));

TASK_DIR=Task.Directory;
%TASK_DIR=([BOX_DIR '\CoganLab\D_Data\' Task.Name]);
DUKEDIR=TASK_DIR;


% Populate Subject file
Subject = popTaskSubjectData(Task);



for iSN=1:length(Subject)
    load([TASK_DIR '/' Subject(iSN).Name '/mat/experiment.mat']);
    load([TASK_DIR '/' Subject(iSN).Name '/' Subject(iSN).Date '/mat/Trials.mat'])
    
    % linefilter if not already done
    for iR=1:length(Subject(iSN).Rec)
        if Subject(iSN).Rec(iR).lineNoiseFiltered==0
            ntools_procCleanIEEG([TASK_DIR '/' Subject(iSN).Name '/' ...
                Subject(iSN).Date '/00' num2str(iR) ...
                '/' Subject(iSN).Rec(iR).fileNamePrefix]);
        end
    end
end

