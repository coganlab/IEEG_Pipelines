function [minLossChannel,minLossAll,minLossIdAll,minLossIdChannel,sig2analyzeAllFeature,labels,Call,CChannel] =  electrodeImportanceBagging(ieeg,fs,tw,labels,numDim)
time = linspace(-0.5,1,size(ieeg,3));
timeSelect = time>=tw(1)&time<=tw(2);

sig2analyzeAllFeature = reshape(permute((ieeg(:,:,timeSelect)),[2,1,3]),[size(ieeg,2),size(ieeg,1)*size(ieeg(:,:,timeSelect),3)]); % reshaping the vector
varSig = var(sig2analyzeAllFeature,0,2); % Calculating the variance at each trial
sig2analyzeAllFeature = sig2analyzeAllFeature(varSig<1e4,:); %Removing trials with low snr
labels = labels(varSig<1e4);
ieeg = ieeg(:,varSig<1e4,:);

%sig2analyzeAllFeature = (sig2analyzeAllFeature - mean(sig2analyzeAllFeature,2))./std(sig2analyzeAllFeature,0,2); % Normalizing the data
[lossVect,~,Call] = scoreSelect(sig2analyzeAllFeature,labels,numDim);
[minLossAll,minLossIdAll] = min(mean(lossVect,1));

for i = 1:size(ieeg,1)
    i
    ieegSelect = ieeg(setdiff(1:size(ieeg,1),i),:,:);
    sig2analyzeAllFeatureChannel = reshape(permute((ieegSelect(:,:,timeSelect)),[2,1,3]),[size(ieegSelect,2),size(ieegSelect,1)*size(ieegSelect(:,:,timeSelect),3)]);
   % sig2analyzeAllFeatureChannel = (sig2analyzeAllFeatureChannel - mean(sig2analyzeAllFeatureChannel,2))./std(sig2analyzeAllFeatureChannel,0,2); % Normalizing the data
    [lossVect,~,CChannel{i}] = scoreSelect(sig2analyzeAllFeatureChannel,labels,numDim);
    [minLossChannel(i),minLossIdChannel(i)] = min(mean(lossVect,1));
end

end