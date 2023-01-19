function badChannels = channelOutlierRemoval(Subj,Task,Trials,Field)
global BOX_DIR
global TASK_DIR
global experiment
outlierSD=3;
ieeg=trialIEEG(Trials,Subj.ChannelNums,Field,[-2000 5000]);
%ieegR=zeros(size(ieeg,2),size(ieeg,1)*size(ieeg,3));
% for iChan=1:size(ieeg,2);
%     disp([iChan '/' size(ieeg,2)]);
%     ieegR(iChan,:)=reshape(ieeg(:,iChan,:),1,size(ieeg,1)*size(ieeg,3));
% end
ieegR=reshape(ieeg,size(ieeg,1),size(ieeg,2)*size(ieeg,3));
ieegR2=detrend(ieegR').^2;
iiZero=find(ieegR2==0);
ieegR2(iiZero)=.000000001;

ieegSTD=std(log(ieegR2),[],1);
ieegSTD=std(ieegR2,[],1);

[m s]=normfit(ieegSTD);
iiOutPlus1=find(ieegSTD>(outlierSD*s+m));
chanIn=setdiff(1:size(ieegSTD,2),iiOutPlus1);
[m s]=normfit(ieegSTD(chanIn));
iiOutPlus2=find(ieegSTD(chanIn)>(3*s+m));

badChannels=sort(cat(2,iiOutPlus1,chanIn(iiOutPlus2)));














