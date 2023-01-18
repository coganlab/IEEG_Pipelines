function preProcess_Specgrams(Task)

global TASK_DIR
global experiment
global DUKEDIR
global BOX_DIR
global RECONDIR
%BOX_DIR='C:\Users\gcoga\Box';
%BOX_DIR='H:\Box Sync';
%RECONDIR='C:\Users\gcoga\Box\ECoG_Recon';
RECONDIR=[BOX_DIR '\ECoG_Recon'];
%
Task=[];
%Task.Name='LexicalDecRepNoDelay';

%Task.Name='Phoneme_Sequencing';
%Task.Name='SentenceRep';
%Task.Name='LexicalDecRepNoDelay';
%Task.Name='Neighborhood_Sternberg';
if strcmp(Task.Name,'LexicalDecRepNoDelay')
    Task.Fig.Field(1).Name='Start';
    Task.Fig.Field(1).Epoch='Start';
    Task.Fig.Field(1).Time=[0 500];
    Task.Fig.Field(2).Name='Auditory_Decision';
    Task.Fig.Field(2).Epoch='Auditory';
    Task.Fig.Field(2).Time=[0 1000]; %[0 500];
    Task.Fig.Field(3).Name='Auditory_JustListen';
    Task.Fig.Field(3).Epoch='Auditory';
    Task.Fig.Field(3).Time=[0 1000];
    Task.Fig.Field(4).Name='Auditory_Repeat';
    Task.Fig.Field(4).Epoch='Auditory';
    Task.Fig.Field(4).Time=[0 1000];
    Task.Fig.Field(5).Name='Response_Repeat';
    Task.Fig.Field(5).Epoch='ResponseStart';
    Task.Fig.Field(5).Time=[-500 500];
    Task.Fig.Field(6).Name='Button_Decision';
    Task.Fig.Field(6).Epoch='ReactionTime';
    Task.Fig.Field(6).Time=[-500 500];
    
    Task.Fig.Baseline.Name='Start';
    Task.Fig.Baseline.Epoch='Start';
    Task.Fig.Baseline.Time=[-500 0];
elseif strcmp(Task.Name,'Neighborhood_Sternberg')
    Task.Fig.Field(1).Name='Auditory';
    Task.Fig.Field(1).Epoch='FirstStimAuditory';
    Task.Fig.Field(1).Time=[0 2000]; %[0 500];
    Task.Fig.Field(2).Name='ListenCue';
    Task.Fig.Field(2).Epoch='ListenCueOnset';
    Task.Fig.Field(2).Time=[0 500];
    Task.Fig.Field(3).Name='Maintenance';
    Task.Fig.Field(3).Epoch='MaintenanceOnset';
    Task.Fig.Field(3).Time=[1000 2000];
    Task.Fig.Field(4).Name='Probe';
    Task.Fig.Field(4).Epoch='ProbeAuditory';
    Task.Fig.Field(4).Time=[0 1500];
    Task.Fig.Field(5).Name='Response';
    Task.Fig.Field(5).Epoch='ResponseOnset';
    Task.Fig.Field(5).Time=[0 1500];
    Task.Fig.Baseline.Name='ListenCue';
    Task.Fig.Baseline.Epoch='ListenCueOnset';
    Task.Fig.Baseline.Time=[-500 0];
elseif strcmp(Task.Name,'SentenceRep')
    Task.Fig.Field(1).Name='Auditory_LSW';
    Task.Fig.Field(1).Epoch='Auditory';
    Task.Fig.Field(1).Time=[0 1000]; %[0 500];
    Task.Fig.Field(2).Name='Go_LSW';
    Task.Fig.Field(2).Epoch='Go';
    Task.Fig.Field(2).Time=[0 1500];
    Task.Fig.Field(3).Name='Start_LSW';
    Task.Fig.Field(3).Epoch='Start';
    Task.Fig.Field(3).Time=[0 500];
    Task.Fig.Field(4).Name='Maintenance_LSW';
    Task.Fig.Field(4).Epoch='Go';
    Task.Fig.Field(4).Time=[-1000 -500];
    
    Task.Fig.Field(5).Name='Auditory_LSS';
    Task.Fig.Field(5).Epoch='Auditory';
    Task.Fig.Field(5).Time=[0 3500]; %[0 500];
    Task.Fig.Field(6).Name='Go_LSS';
    Task.Fig.Field(6).Epoch='Go';
    Task.Fig.Field(6).Time=[0 3500];
    Task.Fig.Field(7).Name='Start_LSS';
    Task.Fig.Field(7).Epoch='Start';
    Task.Fig.Field(7).Time=[0 500];
    Task.Fig.Field(8).Name='Maintenance_LSS';
    Task.Fig.Field(8).Epoch='Go';
    Task.Fig.Field(8).Time=[-1000 -500];
    
    Task.Fig.Field(9).Name='Auditory_JLW';
    Task.Fig.Field(9).Epoch='Auditory';
    Task.Fig.Field(9).Time=[0 1000]; %[0 500];
    Task.Fig.Field(10).Name='Go_JLW';
    Task.Fig.Field(10).Epoch='Go';
    Task.Fig.Field(10).Time=[0 1500];
    Task.Fig.Field(11).Name='Start_JLW';
    Task.Fig.Field(11).Epoch='Start';
    Task.Fig.Field(11).Time=[0 500];
    Task.Fig.Field(12).Name='Maintenance_JLW';
    Task.Fig.Field(12).Epoch='Go';
    Task.Fig.Field(12).Time=[-1000 -500];
    
    Task.Fig.Field(13).Name='Auditory_JLS';
    Task.Fig.Field(13).Epoch='Auditory';
    Task.Fig.Field(13).Time=[0 3500]; %[0 500];
    Task.Fig.Field(14).Name='Go_JLS';
    Task.Fig.Field(14).Epoch='Go';
    Task.Fig.Field(14).Time=[0 3500];
    Task.Fig.Field(15).Name='Start_JLS';
    Task.Fig.Field(15).Epoch='Start';
    Task.Fig.Field(15).Time=[0 500];
    Task.Fig.Field(16).Name='Maintenance_JLS';
    Task.Fig.Field(16).Epoch='Go';
    Task.Fig.Field(16).Time=[-1000 -500];
    
    Task.Fig.Field(17).Name='Auditory_LMW';
    Task.Fig.Field(17).Epoch='Auditory';
    Task.Fig.Field(17).Time=[0 1000]; %[0 500];
    Task.Fig.Field(18).Name='Go_LMW';
    Task.Fig.Field(18).Epoch='Go';
    Task.Fig.Field(18).Time=[0 1500];
    Task.Fig.Field(19).Name='Start_LMW';
    Task.Fig.Field(19).Epoch='Start';
    Task.Fig.Field(19).Time=[0 500];
    Task.Fig.Field(20).Name='Maintenance_LMW';
    Task.Fig.Field(20).Epoch='Go';
    Task.Fig.Field(20).Time=[-1000 -500];
    
    
    Task.Fig.Baseline.Name='Start';
    Task.Fig.Baseline.Epoch='Start';
    Task.Fig.Baseline.Time=[-500 0];
    
elseif strcmp(Task.Name,'LexicalDecRepDelay')
    Task.Fig.Field(1).Name='AuditoryRepeat';
    Task.Fig.Field(1).Epoch='Auditory';
    Task.Fig.Field(1).Time=[0 1000]; %[0 500];
    Task.Fig.Field(2).Name='GoRepeat';
    Task.Fig.Field(2).Epoch='Go';
    Task.Fig.Field(2).Time=[0 1500];
    Task.Fig.Field(3).Name='StartRepeat';
    Task.Fig.Field(3).Epoch='Start';
    Task.Fig.Field(3).Time=[0 500];
    Task.Fig.Field(4).Name='MaintenanceRepeat';
    Task.Fig.Field(4).Epoch='Go';
    Task.Fig.Field(4).Time=[-1000 -500];
    Task.Fig.Field(5).Name='ResponseRepeat';
    Task.Fig.Field(5).Time=[-500 500];
    Task.Fig.Field(5).Epoch='ResponseStart';
    
    Task.Fig.Field(6).Name='AuditoryDecision';
    Task.Fig.Field(6).Epoch='Auditory';
    Task.Fig.Field(6).Time=[0 1000]; %[0 500];
    Task.Fig.Field(7).Name='GoDec';
    Task.Fig.Field(7).Epoch='Go';
    Task.Fig.Field(7).Time=[0 1500];
    Task.Fig.Field(8).Name='StartDecision';
    Task.Fig.Field(8).Epoch='Start';
    Task.Fig.Field(8).Time=[0 500];
    Task.Fig.Field(9).Name='MaintenanceDecision';
    Task.Fig.Field(9).Epoch='Go';
    Task.Fig.Field(9).Time=[-1000 -500];
    Task.Fig.Field(10).Name='ResponseDecision';
    Task.Fig.Field(10).Time=[-500 500];
    Task.Fig.Field(10).Epoch='ResponseStart';
    Task.Fig.Baseline.Name='Start';
    Task.Fig.Baseline.Epoch='Start';
    Task.Fig.Baseline.Time=[-500 0];
else
    Task.Fig.Field(1).Name='Auditory';
    Task.Fig.Field(1).Epoch='Auditory';
    Task.Fig.Field(1).Time=[0 1000]; %[0 500];
    Task.Fig.Field(2).Name='Go';
    Task.Fig.Field(2).Epoch='Go';
    Task.Fig.Field(2).Time=[0 1500];
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

Task.Directory=([BOX_DIR '/CoganLab/D_Data/' Task.Name]);
addpath(genpath(Task.Directory));
%TASK_DIR=([BOX_DIR '/CoganLab/D_Data/' Task.Name]);
TASK_DIR=Task.Directory;
DUKEDIR=TASK_DIR;
% Populate Subject file
Subject = popTaskSubjectData(Task);
SNList=[36]; %30 %1:length(Subject);




%DUKEDIR=['C:\Users\gcoga\Box\CoganLab\D_Data'];
%DUKEDIR=[BOX_DIR '\CoganLab\D_Data'];
%DUKEDIR = [DUKEDIR '/' Task.Name];

% SNList=1:6;
% SNList=4;
%SNList=[3:11,14,16:18];
%SNList=18; %21 %[18,20]
    counterChan=0;
    for iSN=1:length(SNList);
        AnalParams=[];
        CondParams=[];
        SN=SNList(iSN);
        %   Subject(SN).Experiment = loadExperiment(Subject(SN).Name);

        experiment = Subject(SN).Experiment;
        load([DUKEDIR '/' Subject(SN).Name '/' Subject(SN).Date '/mat/trialInfo.mat'])
        load([DUKEDIR '/' Subject(SN).Name '/' Subject(SN).Date '/mat/Trials.mat']);
        TrialsAll=Trials;

        if strcmp(Task.Name,'SentenceRep')
            for iTrials=1:length(Trials);
                Trials(iTrials).StartCode=1;
                Trials(iTrials).AuditoryCode=26;
                Trials(iTrials).GoCode=51;
            end
            TrialsRecode=Trials;
        elseif strcmp(Task.Name,'Neighborhood_Sternberg')
            for iTrials=1:length(Trials);
                Trials(iTrials).StartCode=1;
                Trials(iTrials).ListenCueOnsetCode=1;
                Trials(iTrials).FirstStimAuditoryCode=1;
                Trials(iTrials).MaintenanceOnsetCode=1;
                Trials(iTrials).ProbeCueOnsetCode=1;
            end
            
            
        elseif strcmp(Task.Name,'LexicalDecRepNoDelay')
                condIdx=lexSortNoDel(trialInfo); 
                iiD=find(condIdx<=4);
                for iTrials=1:length(iiD)
                    if strcmp(trialInfo{iiD(iTrials)}.Omission,'Responded')
                    Trials(iiD(iTrials)).ReactionTime=30000*trialInfo{iiD(iTrials)}.ReactionTime...
                        +Trials(iiD(iTrials)).Auditory;
                    else
                        Trials(iiD(iTrials)).ReactionTime=[];
                    end
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
        if strcmp(Subject(SN).Name,'D28')
         %   AnalParams.ArtifactThreshold = 20;
            AnalParams.ArtifactThreshold = 16;
        else
            AnalParams.ArtifactThreshold = 12;%12; %8 %12;
        end
        srate=experiment.recording.sample_rate;
        %srate2=srate/4;
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
            clear Field_nSpec
            if strcmp(Task.Name,'SentenceRep')
                condIdx=[];
                Subject(SN).Trials=TrialsRecode;
                if iF<=4
                    counter=0;
                    for iTrials=1:length(TrialsAll)
                        if TrialsAll(iTrials).StartCode<=4
                            condIdx(counter+1)=iTrials;
                            counter=counter+1;
                        end
                    end
                    Subject(SN).Trials=Subject(SN).Trials(condIdx);
                elseif iF>4 && iF<=8
                    counter=0;
                    for iTrials=1:length(TrialsAll)
                        if TrialsAll(iTrials).StartCode>4 && TrialsAll(iTrials).StartCode<=7
                            condIdx(counter+1)=iTrials;
                            counter=counter+1;
                        end
                    end
                    Subject(SN).Trials=Subject(SN).Trials(condIdx);
                elseif iF>8 && iF<=12
                    counter=0;
                    for iTrials=1:length(TrialsAll)
                        if TrialsAll(iTrials).StartCode>7 && TrialsAll(iTrials).StartCode<=11
                            condIdx(counter+1)=iTrials;
                            counter=counter+1;
                        end
                    end
                    Subject(SN).Trials=Subject(SN).Trials(condIdx);
                elseif iF>12 && iF<=16
                    counter=0;
                    for iTrials=1:length(TrialsAll)
                        if TrialsAll(iTrials).StartCode>12 && TrialsAll(iTrials).StartCode<=14
                            condIdx(counter+1)=iTrials;
                            counter=counter+1;
                        end
                    end
                    Subject(SN).Trials=Subject(SN).Trials(condIdx);
                elseif iF>16 && iF<=20
                    counter=0;
                    for iTrials=1:length(TrialsAll)
                        if TrialsAll(iTrials).StartCode>14 && TrialsAll(iTrials).StartCode<=18
                            condIdx(counter+1)=iTrials;
                            counter=counter+1;
                        end
                    end
                    Subject(SN).Trials=Subject(SN).Trials(condIdx);
                end
           

            elseif strcmp('LexicalDecRepDelay',Task.Name)
                condIdx=[];

                for iTrials=1:length(trialInfo);
                    if trialInfo{iTrials}.Trigger>=200
                        condIdx(iTrials)=1; % rep
                    else
                        condIdx(iTrials)=2; % dec
                    end
                end
                iiR=find(condIdx==1);
                iiD=find(condIdx==2);
                if iF<=5
                    Subject(SN).Trials=TrialsAll(iiR);
                else
                    Subject(SN).Trials=TrialsAll(iiD);
                end
                
            elseif strcmp(Task.Name,'LexicalDecRepNoDelay')
                condIdx=lexSortNoDel(trialInfo);
                iiD=find(condIdx<=4);
                iiR=find(condIdx>4 & condIdx<=8);
                iiJL=find(condIdx>8);
                if iF == 2 || iF == 6
                    counter=0;
                    goodIdx=[];
                    for iTrials=1:length(trialInfo(iiD));
                        if strcmp(trialInfo{iiD(iTrials)}.Omission,'Responded')
                            goodIdx(counter+1)=iTrials;
                            counter=counter+1;
                        end
                    end
                    iiD=iiD(goodIdx);
                    Subject(SN).Trials=TrialsAll(iiD);

                elseif iF == 3
                    Subject(SN).Trials=TrialsAll(iiJL);
                elseif iF== 4 || iF == 5
                    Subject(SN).Trials=TrialsAll(iiR);
                end
            end
            
            
            
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
                    %       base= base+mean(sq(Field_Spec{iCode}{iCh}(:,:)),1); % used to be 1:9

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



