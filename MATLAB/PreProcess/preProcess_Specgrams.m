function preProcess_Specgrams(Task)

global TASK_DIR
global experiment
global DUKEDIR
global BOX_DIR
%BOX_DIR='C:\Users\gcoga\Box';
BOX_DIR='H:\Box Sync';
%RECONDIR='C:\Users\gcoga\Box\ECoG_Recon';
RECONDIR=[BOX_DIR '\ECoG_Recon'];
% 
Task=[];
%Task.Name='LexicalDecRepDelay';
Task.Name='Phoneme_Sequencing';
%Task.Name='SentenceRep';
%Task.Name='LexicalDecRepNoDelay';

if strcmp(Task.Name,'LexicalDecRepNoDelay')
    Task.Fig.Field(1).Name='Auditory';
    Task.Fig.Field(1).Epoch='Auditory';
    Task.Fig.Field(1).Time=[0 1000]; %[0 500];
    Task.Fig.Field(2).Name='Start';
    Task.Fig.Field(2).Epoch='Start';
    Task.Fig.Field(2).Time=[0 500];
    Task.Fig.Baseline.Name='Start';
    Task.Fig.Baseline.Epoch='Start';
    Task.Fig.Baseline.Time=[-500 0];
else
    Task.Fig.Field(1).Name='Auditory';
    Task.Fig.Field(1).Epoch='Auditory';
    Task.Fig.Field(1).Time=[0 500]; %[0 500];
    Task.Fig.Field(2).Name='Go';
    Task.Fig.Field(2).Epoch='Go';
    Task.Fig.Field(2).Time=[0 1000];
    Task.Fig.Field(3).Name='Start';
    Task.Fig.Field(3).Epoch='Start';
    Task.Fig.Field(3).Time=[0 500];
    Task.Fig.Field(4).Name='Maintenance';
    Task.Fig.Field(4).Epoch='Auditory';
    Task.Fig.Field(4).Time=[500 1000];
    Task.Fig.Field(5).Name='Response';
    Task.Fig.Field(5).Time=[-500 500];
    Task.Fig.Field(5).Epoch='ResponseStart';    
    Task.Fig.Baseline.Name='Start';
    Task.Fig.Baseline.Epoch='Start';
    Task.Fig.Baseline.Time=[-500 0];
end

TASK_DIR=([BOX_DIR '/CoganLab/D_Data/' Task.Name]);
DUKEDIR=TASK_DIR;

% Populate Subject file
Subject = popTaskSubjectData(Task.Name);




%DUKEDIR=['C:\Users\gcoga\Box\CoganLab\D_Data'];
DUKEDIR=[BOX_DIR '\CoganLab\D_Data'];
DUKEDIR = [DUKEDIR '/' Task.Name];
SNList=1:length(Subject);
%SNList=18; %21 %[18,20]
counterChan=0;
for iSN=1:length(SNList);
    AnalParams=[];
    CondParams=[];
    SN=SNList(iSN);
    Subject(SN).Experiment = loadExperiment(Subject(SN).Name);

experiment = Subject(SN).Experiment;
load([DUKEDIR '/' Subject(SN).Name '/' Subject(SN).Date '/mat/trialInfo.mat'])
load([DUKEDIR '/' Subject(SN).Name '/' Subject(SN).Date '/mat/Trials.mat']);

if strcmp(Task.Name,'SentenceRep')
    for iTrials=1:length(Trials);
        Trials(iTrials).StartCode=1;
        Trials(iTrials).AuditoryCode=26;
        Trials(iTrials).GoCode=51;
    end
end
    
Subject(SN).Trials=Trials;
experiment = Subject(SN).Experiment;

CondParams.Conds=1:1


AnalParams.Channel=setdiff(Subject(SN).ChannelNums,Subject(SN).badChannels);
CondParams.Conds=[1:1];
%NumTrials = SelectChannels(Subject(SN).Trials, CondParams, AnalParams);
SelectedChannels=AnalParams.Channel; % really loose: accounts for practice trial confound
AnalParams.ReferenceChannels = SelectedChannels;
AnalParams.Channel = SelectedChannels;
AnalParams.TrialPooling = 1; %1;  %1; % used to be 1
AnalParams.dn=0.05;
AnalParams.Tapers = [.5,10];
AnalParams.fk = 200;
AnalParams.Reference = 'Grand average';% 'IndRef'; %'Grand average', 'Grand average induced'% induced' 'Single-ended','IndRef';%AnalParams.RefChans=subjRefChansInd(Subject(SN).Name);
AnalParams.ArtifactThreshold = 12; %8 %12;
srate=experiment.recording.sample_rate;
srate2=srate/4;
if srate<2048
    AnalParams.pad=2;
else
   AnalParams.pad=1;
end 

    CondParams.Conds=[1:1];
    CondParams.Field = Task.Fig.Baseline.Epoch;
    CondParams.bn = Task.Fig.Baseline.Time;
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


for iF=1:length(Task.Fig.Field)
    
    clear Field_Spec
    CondParams.Conds=[1:1];
    CondParams.Field = Task.Fig.Field(iF).Epoch;
    CondParams.bn = [Task.Fig.Field(iF).Time(1)-500 Task.Fig.Field(iF).Time(2)+500] ;
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
    
    
    base=0;
    %base = zeros(1,size(Auditory_Spec{iCode}{iCh},2));
    for iCh = 1:length(Field_Spec{iCode})
        base=0;
        for iCode = 1:length(Field_Spec)
            %base = base + sq(Auditory_Spec{iCode}{iCh}(5,:)); % standard
            %   base= base+mean(sq(Auditory_Spec{iCode}{iCh}(1:10,:)),1); % used to be 1:9
            base= base+mean(sq(Base_Spec{iCode}{iCh}(:,:)),1); % used to be 1:9
            
            %base2(iCode,:)=std(sq(Auditory_Spec{iCode}{iCh}(1:6,:)),1); % std across time bins?
            
        end
        base = base./length(Field_Spec);
        for iCode = 1:length(Field_Spec)
            Field_nSpec(iCode,iCh,:,:) = Field_Spec{iCode}{iCh}(:,:)./base(ones(1,size(Field_Spec{iCode}{iCh},1)),:);          
        end
        
    end

totChanBlock=ceil(length(AnalParams.Channel)./60);
iChan2=0;
for iG=0:totChanBlock-1;
    FigS=figure('Position', get(0, 'Screensize'));
    for iChan=1:min(60,length(AnalParams.Channel)-iChan2);
        subplot(6,10,iChan);
        iChan2=iChan+iG*60;
        tvimage(sq((Field_nSpec([1],iChan2,:,1:200))),'XRange',[CondParams.bn(1)./1000,CondParams.bn(2)./1000]);
        title(experiment.channels(AnalParams.Channel(iChan2)).name);
        caxis([0.7 1.2]);
      %  caxis([0.7 1.4]);

    end   
    F=getframe(FigS);
    if ~exist([DUKEDIR '/Figs/' Subject(SN).Name],'dir')
        mkdir([DUKEDIR '/Figs/' Subject(SN).Name])
    end
    imwrite(F.cdata,[DUKEDIR '/Figs/' Subject(SN).Name '/' Subject(SN).Name '_' Task.Name '_' Task.Fig.Field(iF).Name '_SpecGrams_200Hz_0.7to1.2C_' num2str(iG+1) '.png'],'png');    
  %  imwrite(F.cdata,[DUKEDIR '/Figs/' Subject(SN).Name '/' Subject(SN).Name '_NeighborhoodSternberg_Auditory_SpecGrams_200Hz_0.7to1.4C_' num2str(iF+1) '.png'],'png');
    close
end  


totChanBlock=ceil(length(AnalParams.Channel)./60);
iChan2=0;
for iG=0:totChanBlock-1;
    FigS=figure('Position', get(0, 'Screensize'));
    for iChan=1:min(60,length(AnalParams.Channel)-iChan2);
        subplot(6,10,iChan);
        iChan2=iChan+iG*60;
        tvimage(sq((Field_nSpec([1],iChan2,:,1:200))),'XRange',[CondParams.bn(1)./1000,CondParams.bn(2)./1000]);
        title(experiment.channels(AnalParams.Channel(iChan2)).name);
        caxis([0.7 1.4]);
      %  caxis([0.7 1.4]);

    end;
    F=getframe(FigS);
    if ~exist([DUKEDIR '/Figs/' Subject(SN).Name],'dir')
        mkdir([DUKEDIR '/Figs/' Subject(SN).Name])
    end
    imwrite(F.cdata,[DUKEDIR '/Figs/' Subject(SN).Name '/' Subject(SN).Name '_' Task.Name '_' Task.Fig.Field(iF).Name '_SpecGrams_200Hz_0.7to1.4C_' num2str(iG+1) '.png'],'png');    
  %  imwrite(F.cdata,[DUKEDIR '/Figs/' Subject(SN).Name '/' Subject(SN).Name '_NeighborhoodSternberg_Auditory_SpecGrams_200Hz_0.7to1.4C_' num2str(iF+1) '.png'],'png');
    close
end  

clear Field_nSpec;
end

end



