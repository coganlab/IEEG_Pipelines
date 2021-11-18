global BOX_DIR
global RECONDIR
global TASK_DIR
global experiment
global DUKEDIR

% Greg Laptop
% BOX_DIR='C:\Users\gcoga\Box';
% RECONDIR='C:\Users\gcoga\Box\ECoG_Recon';
%

% BOX_DIR='C:\Users\sr241\Box Sync';
% RECONDIR='C:\Users\sr241\Box Sync\ECoG_Recon';

%addpath(genpath([BOX_DIR '\CoganLab\Scripts\']));

Task=[];
%Task.Name='Phoneme_Sequencing';
Task.Name='LexicalDecRepDelay';
% set fields for stats with times (ms) in them
Task.Stats.nPerm=10000;
Task.Stats.nTails=1;
Task.Stats.Field(1).Name='Auditory';
Task.Stats.Field(1).Epoch='Auditory';
Task.Stats.Field(1).Time=[0 500];
Task.Stats.Field(2).Name='Go';
Task.Stats.Field(2).Epoch='Go';
Task.Stats.Field(2).Time=[500 1000];
Task.Stats.Field(3).Name='Start';
Task.Stats.Field(3).Epoch='Start';
Task.Stats.Field(3).Time=[0 500];
Task.Stats.Field(4).Name='Maintenance';
Task.Stats.Field(4).Epoch='Auditory';
Task.Stats.Field(4).Time=[500 1000];
Task.Stats.Field(5).Name='Response';
Task.Stats.Field(5).Epoch='ResponseStart';
Task.Stats.Field(5).Time=[0 1000];
Task.Stats.Baseline.Name='Start';
Task.Stats.Baseline.Epoch='Start';
Task.Stats.Baseline.Time=[-500 0];
Task.Stats.Freq(1).Name='HighGamma';
Task.Stats.Freq(1).Range=[70:120];


TASK_DIR=([BOX_DIR '/CoganLab/D_Data/' Task.Name]);
DUKEDIR=TASK_DIR;

if ~exist([TASK_DIR '/Stats']);
    mkdir([TASK_DIR '/Stats']);
end

% Populate Subject file
Subject = popTaskSubjectData(Task.Name);

SNList = [1:16]; % D47 (17) for phoneme sequencing not coded yet
for iSN=1:length(SNList)
    SN=SNList(iSN);
    experiment=Subject(SN).experiment;
%     load([BOX_DIR '/CoganLab/D_Data/' Task.Name '/' Subject(iSN).Name '/mat/experiment.mat']);
%     load([TASK_DIR '/' Subject(iSN).Name '/' Subject(iSN).Date '/mat/Trials.mat'])
%     
%   
%     
%     Subject(SN).Trials=Trials;
    AnalParams.Channel=Subject(SN).goodChannels;
    SelectedChannels=AnalParams.Channel; % really loose: accounts for practice trial confound
    AnalParams.ReferenceChannels = SelectedChannels;
    AnalParams.TrialPooling = 1; %1;  %1; % used to be 1
    AnalParams.dn=0.05;
    AnalParams.Tapers = [0.5 10];
    AnalParams.fk = 200;
    AnalParams.Reference = 'Grand average'; % 'IndRef'; %'Grand average', 'Grand average induced'% induced' 'Single-ended','IndRef';%AnalParams.RefChans=subjRefChansInd(Subject(SN).Name);
    AnalParams.ArtifactThreshold = 12; %8 %12;
    AnalParams.TrialPooling = 0; %1;  %1; % used to be 1
    srate=experiment.recording.sample_rate;
    srate2=srate/4;
    if srate<2048
        AnalParams.pad=2;
    else
        AnalParams.pad=1;
    end
    
    CondParams.Conds=[1:1];
    CondParams.Field = Task.Stats.Baseline.Epoch;
    CondParams.bn = Task.Stats.Baseline.Time;
    
    for iCode = 1:length(CondParams.Conds)
        
        if isfield(CondParams,'Conds2')
            CondParams.Conds = CondParams.Conds2(iCode);
        else
            CondParams.Conds = iCode;
        end
        tic
        [Base_Spec{iCode}, Base_Data, Base_Trials{iCode}] = subjSpectrum(Subject(SN), CondParams, AnalParams);
        toc
        display(['Cond = ' num2str(iCode)])
    end
    
    
    for iF=1:length(Task.Stats.Field)
        clear Field_Spec Field_Data Field_Trials
        CondParams.Conds=[1:1];
        CondParams.Field = Task.Stats.Field(iF).Epoch;
        CondParams.bn = Task.Stats.Field(iF).Time;
        for iCode = 1:length(CondParams.Conds)
            
            if isfield(CondParams,'Conds2')
                CondParams.Conds = CondParams.Conds2(iCode);
            else
                CondParams.Conds = iCode;
            end
            tic
            [Field_Spec{iCode}, Field_Data, Field_Trials{iCode}] = subjSpectrum(Subject(SN), CondParams, AnalParams);
            toc
            display(['Cond = ' num2str(iCode)])
        end
        
        nPerm=Task.Stats.nPerm;
        nTails=Task.Stats.nTails;
        for iFreq=1:size(Task.Stats.Freq,1)
            pVals=zeros(length(Subject(SN).goodChannels),2);
            for iChan=1:length(Subject(SN).goodChannels)           
                sig1=sq(mean(mean(Field_Spec{1}{iChan}(:,:,Task.Stats.Freq(iFreq).Range),2),3));
                [goodIdx1 badIdx1]=outlierRemoval(log(sig1),3);
                sig2=sq(mean(mean(Base_Spec{1}{iChan}(:,:,Task.Stats.Freq(iFreq).Range),2),3));
                [goodIdx2 badIdx2]=outlierRemoval((sig2),3);                
                pVals(iChan,:) = shufflePermTest(sig1(goodIdx1),sig2(goodIdx2),nPerm,nTails);
            end
            save([TASK_DIR '/Stats/' Subject(SN).Name '_' Task.Name '_' Task.Stats.Field(iF).Name '_' Task.Stats.Freq(iFreq).Name '_' num2str(Task.Stats.nTails) 'Tail.mat'],'pVals');
        end
    end
end
