function [silScore,silRatio] = tsneScoreExtract(ieegSplit,labels,tw,chanMap,etw,varVal,nElec,nIter)
% Extract tsne score after SVD decomposition
% Input arguments
arguments 
    ieegSplit double % Ieeg epochs - channels x trials x time
    labels double % label ids - (1 x trials)
    tw (1,2) double % time-windos of ieeg epochs    
    chanMap double % 2D micro-ECoG channel Map
    etw (1,2) double = tw % epoched time-window for analysis; Defaults to tw
    varVal double = 80 % Variance explained for SVD; Defaults to 80
    nElec double = 1:size(ieegSplit,1) % number of electrodes for Poisson disc sampling; Defaults to total number of electrodes
    nIter double = 50 % Number for iterations; Defaults to 50
end
% Output arguments
% silScore - 2D array of silhouette score
% silRatio - 2D array of silhouette ratio
% Function body

selectedChannels = sort(chanMap(~isnan(chanMap)))';

timeSplit = linspace(tw(1),tw(2),size(ieegSplit,3));
timeSelect = timeSplit>=etw(1)&timeSplit<=etw(2);
for iTer = 1:nIter
    for iSamp = 1:length(nElec)
        elecPtIds = ceil(poissonDisc2([size(chanMap,1),size(chanMap,2)],nElec(iSamp)));
        elecPt = [];
        for iElec = 1:size(elecPtIds,1)
            elecPt(iElec) = chanMap(elecPtIds(iElec,1),elecPtIds(iElec,2));
        end
        elecPt = elecPt(~isnan(elecPt)); 
        elecPtcm = ismember(selectedChannels,elecPt);
        sum(elecPtcm);
        if(sum(elecPtcm)<2)
            iSamp = iSamp - 1;
            continue;
        end
        ieegSelect =squeeze(ieegSplit(elecPtcm,:,timeSelect));
        ieegShape = size(ieegSelect);
        ieegSelect = reshape(permute(ieegSelect,[2 1 3]),[ieegShape(2) ieegShape(1)*ieegShape(3)]);
        [~,scoreIeeg,~,~,explained] = pca(ieegSelect,'Centered',false);
        nModes = find(cumsum(explained)>varVal,1);
        
        Y = tsne(scoreIeeg(:,1:nModes));
        size(Y);
        [s] = silhouette(Y,labels,'Euclidean');
        sihouetteRatio = sum(s>0)/length(s);
        meanSilhouette = mean(s(s>0));
        silScore(iSamp,iTer) = meanSilhouette;
        silRatio(iSamp,iTer) = sihouetteRatio;
    end

end