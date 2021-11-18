global BOX_DIR
global RECONDIR
global TASK_DIR
global experiment
global DUKEDIR
BOX_DIR='C:\Users\gcoga\Box';
RECONDIR='C:\Users\gcoga\Box\ECoG_Recon';
addpath(genpath([BOX_DIR '\CoganLab\Scripts\']));

Task=[];
%Task.Name='Phoneme_Sequencing';
%Task.Name='LexicalDecRepDelay';
Task.Name='SentenceRep';
Task.Outlier.Field='Auditory';
% set fields for stats with times (ms) in them



TASK_DIR=([BOX_DIR '\CoganLab\D_Data\' Task.Name]);
DUKEDIR=TASK_DIR;


% Populate Subject file
Subject = popTaskSubjectData(Task.Name);



for iSN=1:length(Subject)
    load([BOX_DIR '/CoganLab/D_Data/' Task.Name '/' Subject(iSN).Name '/mat/experiment.mat']);
    load([TASK_DIR '/' Subject(iSN).Name '/' Subject(iSN).Date '/mat/Trials.mat'])
    
    % linefilter if not already done
    for iR=1:length(Subject(iSN).Rec)
        if Subject(iSN).Rec(iR).lineNoiseFiltered==0
            ntools_procCleanIEEG([TASK_DIR '/' Subject(iSN).Name '/' ...
                Subject(iSN).Date '/00' num2str(iR) ...
                '/' Subject(iSN).Rec(iR).fileNamePrefix]);
        end
    end
    
    % find bad channels based on 
    if ~exist([TASK_DIR '/' Subject(iSN).Name '/badChannels.mat']);
        badChannels = channelOutlierRemoval(Subject(iSN),Task,Trials,Task.Outlier.Field);
        save([TASK_DIR '/' Subject(iSN).Name '/badChannels.mat'],'badChannels');
        Subject(iSN).badChannels=badChannels;
    end
    
    
end

