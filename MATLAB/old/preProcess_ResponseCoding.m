global BOX_DIR
global RECONDIR
global TASK_DIR
global experiment
global DUKEDIR
global RESPONSE_DIR
BOX_DIR='H:\Box Sync';
%BOX_DIR='C:\Users\ae166\Box';
RECONDIR=[BOX_DIR '\ECoG_Recon'];
RESPONSE_DIR=[BOX_DIR '\CoganLab\ECoG_Task_Data\response_coding\response_coding_results_PaleeEdits'];

addpath(genpath([BOX_DIR '\CoganLab\Scripts\']));
addpath(genpath([RECONDIR]));
Task=[];
%Task.Name='Phoneme_Sequencing';
%Task.Name='LexicalDecRepDelay';
%Task.Name='LexicalDecRepNoDelay';
Task.Name='SentenceRep';
%Task.Name='Uniqueness_Point';
TASK_DIR=([BOX_DIR '/CoganLab/D_Data/' Task.Name]);
DUKEDIR=TASK_DIR;
Task.Directory=DUKEDIR;

% Populate Subject file
Subject = popTaskSubjectData(Task);

%Subject=Subject([1:7]);
% if strcmp(Task.Name,'LexicalDecRepDelay')
%     Task.Number=1;
% elseif strcmp(Task.Name,'Phoneme_Sequencing')
%     Task.Number=4;
% elseif strcmp(Task.Name,'SentenceRep')
%     Task.Number=3;
% elseif strcmp(Task.Name','Uniqueness_Point')
%     Task.Number=5;
% end

% figure out which of them have response coding behaviorally coded 
counter=0;
for iSN=1:length(Subject);
   % if exist([RESPONSE_DIR '\' Subject(iSN).Name '_task00' num2str(Task.Number)],'dir')
    if exist([RESPONSE_DIR '\' Task.Name '\' Subject(iSN).Name],'dir')

        Subject(iSN).ResponseCoding=1;
SNList(counter+1)=iSN;
counter=counter+1;
    else
        Subject(iSN).ResponseCoding=0;
    end
end

% figure out which are done already?
counter=0;
SNListDone=[];
for iSN=1:length(SNList);
    load([DUKEDIR '\' Subject(SNList(iSN)).Name '\' Subject(SNList(iSN)).Date '\mat\Trials.mat']);
if isfield(Trials,'ResponseStart')
    SNListDone(counter+1)=SNList(iSN);
    counter=counter+1;
end
end
SNListNotDone=setdiff(SNList,SNListDone);


for iSN=1:length(SNListNotDone);
    SN=SNListNotDone(iSN);
    errors=[];
    cue_events=readtable([RESPONSE_DIR '\' Task.Name '\' Subject(SN).Name  '\cue_events.txt']);
    condition_events=readtable([RESPONSE_DIR '\'  Task.Name '\' Subject(SN).Name '\condition_events.txt']);
    
    response_coding=readtable([RESPONSE_DIR '\' Task.Name '\' Subject(SN).Name '\response_coding.txt']);
    if exist([RESPONSE_DIR '\' Task.Name '\' Subject(SN).Name '\errors.txt']);
        errors = readtable([RESPONSE_DIR '\' Subject(SN).Name '_task00' num2str(Task.Number) '\errors.txt']);
    else
        errors=[];
    end
    response_coding=readtable([RESPONSE_DIR '\' Task.Name '\' Subject(SN).Name '\response_coding.txt']);

% does size of T equal size of Trials

if size(cue_events,1)~=length(Subject(SN).Trials)
    display('cue events length mismatch!')
end

if size(response_coding,1)+size(errors,1)~=length(Subject(SN).Trials)
    display('response/errors coding length mismatch!')
end

% get cue onset times
cue_onsets=[];
for iTrials=1:size(cue_events,1);
    cue_onsets(iTrials)=table2array(cue_events(iTrials,1));
end


%LEX DEC REP ONLY get condition values (1 = decision, 2 = repeat)
if strcmp(Task.Name,'LexicalDecRepDelay')
    for iTrials=1:size(condition_events,1)
        if contains(table2array(condition_events(iTrials,3)),'Yes')
            condition_vals(iTrials)=1;
        else
            condition_vals(iTrials)=2;
        end
    end
elseif strcmp(Task.Name,'LexicalDecRepNoDelay')
    for iTrials=1:size(condition_events,1)
        if contains(table2array(condition_events(iTrials,3)),'Yes')
            condition_vals(iTrials)=1; % dec
        elseif contains(table2array(condition_events(iTrials,3)),'Repeat')
            condition_vals(iTrials)=2; % repeat
        else
            condition_vals(iTrials)=3; % just listen
        end
    end
elseif strcmp(Task.Name,'SentenceRep')
    load([TASK_DIR '/' Subject(iSN).Name '/' Subject(iSN).Date '/mat/trialInfo.mat'])
    for iTrials=1:size(trialInfo,2)
        if ~iscell(trialInfo)
            if strcmp(trialInfo(iTrials).go,'Speak') % Listen Speak
                condition_vals(iTrials)=1;
            elseif strcmp(trialInfo(iTrials).go,'Mime') % Listen Mime
                condition_vals(iTrials)=2;
            else
                condition_vals(iTrials)=3; % Just Listen
            end
        else
            if strcmp(trialInfo{iTrials}.go,'Speak') % Listen Speak
                condition_vals(iTrials)=1;
            elseif strcmp(trialInfo{iTrials}.go,'Mime') % Listen Mime
                condition_vals(iTrials)=2;
            else
                condition_vals(iTrials)=3; % Just Listen
            end
        end
    end
  
else
    condition_vals=zeros(size(condition_events,1),1);
end


% repeat = 1, yes = 2, no = 3;, 4 = error; missing = 0; 5 = mismatch errors
response_vals=zeros(length(condition_vals),3);
for iTrials=1:size(response_coding,1)
    tmp1=table2array(response_coding(iTrials,1));
    tmp2=table2array(response_coding(iTrials,2));
    tmp3=table2array(response_coding(iTrials,3));
    [I]=find(cue_onsets<tmp1);
    if tmp1-cue_onsets(I(end))<7 && ~strcmp(Task.Name,'SentenceRep') || tmp1-cue_onsets(I(end))<12 && strcmp(Task.Name,'SentenceRep')% make sure they are within the same trial
        response_vals(I(end),1)=tmp1;
        response_vals(I(end),2)=tmp2;
        if strcmp(Task.Name,'LexicalDecRepDelay')
            if contains(table2cell(response_coding(iTrials,3)),'yes')
                response_vals(I(end),3)=2;
            elseif contains(table2cell(response_coding(iTrials,3)),'no')
                response_vals(I(end),3)=3;
            else
                response_vals(I(end),3)=1;
            end
        else
            response_vals(I(end),3)=1;
        end
        
    end
    if isstring(tmp3) && contains(tmp3,'noisy')
        response_vals(I(end),3)=5;
    end
end

% code for errors in errors file (4)
if size(errors,1)>0
for iTrials=1:size(errors,1)   
tmp1=table2array(errors(iTrials,1));
    tmp2=table2array(errors(iTrials,1));
    [I]=find(cue_onsets<tmp1);
    
    if tmp1-cue_onsets(I(end))<7 && ~strcmp(Task.Name,'SentenceRep') % make sure they are within the same trial
        response_vals(I(end),3)=4;
    elseif  tmp1-cue_onsets(I(end))<20 && strcmp(Task.Name,'SentenceRep') 
       response_vals(I(end),3)=4;

    end
end
end

% ??
if ~strcmp(Task.Name,'SentenceRep')
    for iTrials=1:length(response_vals)
        if response_vals(iTrials,3)==2 && condition_vals(iTrials)==2
            response_vals(iTrials,3)=5;
        elseif response_vals(iTrials,3)==3 && condition_vals(iTrials)==2
            response_vals(iTrials,3)=5;
        elseif response_vals(iTrials,3)==1 && condition_vals(iTrials)==1
            response_vals(iTrials,3)=5;
        end
    end
end


Trials=Subject(SN).Trials;
trialNum=[];
for iTrials=1:length(Trials);
    trialNum(iTrials)=Trials(iTrials).Trial;
end
for iTrials=1:length(Trials);    
    if response_vals(iTrials,1)>0
        Trials(iTrials).ResponseStart=30000*(response_vals(trialNum(iTrials),1)-cue_onsets(trialNum(iTrials)))+Trials(iTrials).Auditory;
        Trials(iTrials).ResponseEnd=30000*(response_vals(trialNum(iTrials),2)-cue_onsets(trialNum(iTrials)))+Trials(iTrials).Auditory;
    else
        Trials(iTrials).NoResponse=1;
        Trials(iTrials).ResponseStart=[];
        Trials(iTrials).ResponseEnd=[];
    end

    if response_vals(trialNum(iTrials),3)>=4
        Trials(iTrials).Noisy=1;
%         Trials(iTrials).ResponseStart=[];
%         Trials(iTrials).ResponseEnd=[];
    end
    
end

save([TASK_DIR '/' Subject(SN).Name '/' Subject(SN).Date '/mat/Trials.mat'],'Trials')
end
%         
% %save(
% load([RESPONSE_DIR '\' Subject(SN).Name '_task00' num2str(Task.Number) '\trialInfo.mat']);
% condIdx = lexSort(trialInfo);
% allLatencies=zeros(length(condIdx),1);
% for iTrials=1:length(Trials);if Trials(iTrials).NoResponse==0;allLatencies(trialNum(iTrials))=Trials(iTrials).ResponseOnset-Trials(iTrials).Go;end;end;
% [m s]=normfit(allLatencies);
% allLatenciesN=(allLatencies-m)./std(allLatencies);
% %iiE=find(allLatencies<=.25*30000 | allLatencies>2*30000);
% iiE=find(abs(allLatenciesN>2));
% counter=0;
% for iCond=1:8;
%     condLat(iCond)=mean(allLatencies(setdiff(find(condIdx==iCond),iiE)))./30;
%     condLatEach{iCond}=allLatencies(setdiff(find(condIdx==iCond),iiE))./30;
%     condLatGroup(counter+1:counter+length(condLatEach{iCond}),1)=condLatEach{iCond};
%     condLatGroup(counter+1:counter+length(condLatEach{iCond}),2)=iCond;
%     counter=counter+length(condLatEach{iCond});
% end
% 
% 
% figure;plot(allLatencies./30);
% figure;plot(allLatencies(setdiff(1:length(allLatencies),iiE))./30)
% figure;plot(condLat);
