function [accAll,ytestAll,ypredAll,optimVarAll,aucAll, modelWeightsAll] = pcaLinearDecoderWrap(ieegSplit,labels,tw,etw,varVector,numFolds,isauc)
% The function performs supervised PCA-LDA decoding on ephys time-series dataset;
% Step 1: Hyperparameter optimization through nested cross-validation to
% identify the optimal number of PC dimensions
% Step 2: Linear discriminant decoding
%
% Input
% ieegSplit (channels x trials x time): Input dataset
% labels (trials): Input labels
% tw: epoch time-window
% etw: selected time-window for decoding
% numDim: Number of PCA dimensions to include
% numFolds: Number of folds for cross-validation
% Output
% accAll - overall accuracy (0 - 1)
% ytestAll - tested labels
% ypredall - predicted labels

timeSplit = linspace(tw(1),tw(2),size(ieegSplit,3));
timeSelect = timeSplit>=etw(1)&timeSplit<=etw(2);
ieegModel = ieegSplit(:,:,timeSelect);

accAll = 0;
ytestAll = [];
ypredAll = [];
optimVarAll = [];
aucAll = [];
lossVectAll = [];
modelWeightsAll = [];
if(numFolds>0)
    cvp = cvpartition(labels,'KFold',numFolds,'Stratify',true);
else
    cvp = cvpartition(labels,'LeaveOut');
end
    for nCv = 1:cvp.NumTestSets
        nCv
        train = cvp.training(nCv);
        test = cvp.test(nCv);
        ieegTrain = ieegModel(:,train,:);
        ieegTest = ieegModel(:,test,:);
        matTrain = size(ieegTrain);
        gTrain = reshape(permute(ieegTrain,[2 1 3]),[matTrain(2) matTrain(1)*matTrain(3)]);
        matTest = size(ieegTest);
        gTest = reshape(permute(ieegTest,[2 1 3]),[matTest(2) matTest(1)*matTest(3)]);

        labelTrain = labels(train);
        labelTest = labels(test);

        % Perform SMOTE on training and testing
        gTrainGen = genTrials4Nan(gTrain,labelTrain,genType = 2);
        %[gTrainGen,labelTrain] = balanceSmoteTrials(gTrainGen,labelTrain,genType = 2);
        %size(gTrainSmote)
        %size(labelTrain)
        gTestGen = gTest;
        noiseMatrix = -1 + 2.*rand(size(gTest));
        
        nanMask = isnan(gTest);
        gTestGen(nanMask) = noiseMatrix(nanMask);
%         gTestSmote = gTest;
%         gTestSmote(isnan(gTest)) = 0;
        if(length(varVector)>1)
            if(numFolds>0)
                [lossVect] = scoreSelect(gTrainGen,labelTrain,varVector,1,numFolds); % Hyper parameter tuning
            else
                [lossVect] = scoreSelect(gTrainGen,labelTrain,varVector,0,numFolds);
            end

             lossVectAll(nCv,:) = mean(lossVect,1);
             %phonErrorVectAll(nCv,:) = mean(phonErrorVect,1);
            [~,optimVarId] = min(mean(lossVect,1)); % Selecting the optimal principal components
            optimVar = varVector(optimVarId);
        else
            optimVar = varVector;
        end
%        mean(squeeze(aucVect(:,nDim,:)),1)
        [lossMod,Cmat,yhat,aucVect,nModes,modelweights] = pcaDecodeVariance(gTrainGen,gTestGen,labelTrain,...
                       labelTest,optimVar,isauc);
    optimVarAll = [optimVarAll optimVar];
%     size(pTest)
     %size(aucVect)
    ytestAll = [ytestAll labelTest];
    ypredAll = [ypredAll yhat'];
    accAll = accAll + 1 - lossMod;
    modelWeightsAll{nCv} = modelweights;

    if(isauc)
    aucAll = [aucAll; aucVect];
    end
    %CmatAll = CmatAll + Cmat;
    end
%     figure; 
%     plot(varVector,lossVectAll);
%     xlabel('Variance explained');
%     ylabel('Classification loss');
    
%     figure; 
%     plot(varVector,phonErrorVectAll);
%     xlabel('Variance explained');
%     ylabel('Phoneme error (bits)');
    
    
    
end