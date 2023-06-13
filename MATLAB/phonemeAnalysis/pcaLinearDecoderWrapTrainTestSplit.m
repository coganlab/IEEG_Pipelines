function [accAll,ytestAll,ypredAll,optimVarAll,aucAll] = pcaLinearDecoderWrapTrainTestSplit(ieegSplitTrain,ieegSplitTest,labels,tw,etwTrain,etwTest,varVector,numFolds,isauc)
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


timeSplit = linspace(tw(1),tw(2),size(ieegSplitTrain,3));
timeSelectTrain = timeSplit>=etwTrain(1)&timeSplit<=etwTrain(2);
timeSelectTest = timeSplit>=etwTest(1)&timeSplit<=etwTest(2);



accAll = 0;
ytestAll = [];
ypredAll = [];
optimVarAll = [];
aucAll = [];
accVectAll = [];


if(numFolds>0)
    cvp = cvpartition(labels,'KFold',numFolds,'Stratify',true);
else
    cvp = cvpartition(labels,'LeaveOut');
end
    for nCv = 1:cvp.NumTestSets
        
        train = cvp.training(nCv);
        test = cvp.test(nCv);
        ieegTrain = ieegSplitTrain(:,train,timeSelectTrain);
        ieegTest = ieegSplitTest(:,test,timeSelectTest);
        
        matTrain = size(ieegTrain);
        gTrain = reshape(permute(ieegTrain,[2 1 3]),[matTrain(2) matTrain(1)*matTrain(3)]);
        matTest = size(ieegTest);
        gTest = reshape(permute(ieegTest,[2 1 3]),[matTest(2) matTest(1)*matTest(3)]);
        pTrain = labels(train);
        pTest = labels(test);

       
        if(length(varVector)>1)
            if(numFolds>0)
                [lossVect] = scoreSelect(gTrain,pTrain,varVector,1,numFolds); % Hyper parameter tuning
            else
                [lossVect] = scoreSelect(gTrain,pTrain,varVector,0,numFolds);
            end

             
            [~,optimVarId] = min(mean(lossVect,1)); % Selecting the optimal principal components
            optimVar = varVector(optimVarId);
        else
            optimVar = varVector;
        end
%        mean(squeeze(aucVect(:,nDim,:)),1)
        [lossMod,Cmat,yhat,aucVect] = pcaDecodeVariance(gTrain,gTest,pTrain,...
                       pTest,optimVar,isauc);
        optimVarAll = [optimVarAll optimVar];
    %     size(pTest)
         %size(aucVect)
        ytestAll = [ytestAll pTest];
        ypredAll = [ypredAll yhat'];
        accAll = accAll + 1 - lossMod;
        if(isauc)
            aucAll = [aucAll; aucVect];
        end
    %CmatAll = CmatAll + Cmat;
    end
end
%     figure; 
%     plot(varVector,accVectAll);
    
