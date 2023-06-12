function [ytestAll,ypredAll,optimDimAll] = pcaLinearRegressDecoderWrapTrainTest(ieegSplit,predictors,tw,etwTrain,etwTest,varVector,numFolds)
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
timeSelectTrain = timeSplit>=etwTrain(1)&timeSplit<=etwTrain(2);
timeSelectTest = timeSplit>=etwTest(1)&timeSplit<=etwTest(2);

lossAll = 0;
ytestAll = [];
ypredAll = [];
optimDimAll = [];
accVectAll = [];
if(numFolds>0)
    cvp = cvpartition(length(predictors),'KFold',numFolds);
else
    cvp = cvpartition(length(predictors),'LeaveOut');
end
% figure; 
    for nCv = 1:cvp.NumTestSets
        
        train = cvp.training(nCv);
        test = cvp.test(nCv);
        ieegTrain = ieegSplit(:,train,timeSelectTrain);
        ieegTest = ieegSplit(:,test,timeSelectTest);
        matTrain = size(ieegTrain);
        gTrain = reshape(permute(ieegTrain,[2 1 3]),[matTrain(2) matTrain(1)*matTrain(3)]);
        matTest = size(ieegTest);
        gTest = reshape(permute(ieegTest,[2 1 3]),[matTest(2) matTest(1)*matTest(3)]);

        pTrain = predictors(train);
        pTest = predictors(test);
        
        if(length(varVector)>1)
            if(numFolds>0)
                [lossVect] = scoreSelectRegress(gTrain,pTrain,varVector,1,numFolds); % Hyper parameter tuning
            else
                [lossVect] = scoreSelectRegress(gTrain,pTrain,varVector,0,numFolds);
            end

             accVectAll(nCv,:) = mean(lossVect,1);
            [~,optimVarId] = min(mean(lossVect,1)); % Selecting the optimal principal components
            
%             plot(varVector,mean(lossVect,1));
%             hold on;
            optimVar = varVector(optimVarId);
        else
            optimVar = varVector;
        end
        
         
      
        
        [yhat,loss] = pcaDecodeRegress(gTrain,gTest,pTrain,...
                       pTest,optimVar);
        
    optimDimAll = [optimDimAll optimVar];
    ytestAll = [ytestAll pTest];
    ypredAll = [ypredAll yhat'];
    lossAll = [lossAll loss];
%     for iTest = 1:size(pTest,1)
%         for iLabel = 1:multiLabel
%             labelIndices = ((iLabel-1)*labelUnique)+1:iLabel*labelUnique;
%             [~,maxIdTest] = max(pTest(iTest,labelIndices));
%              ytestAll = [ytestAll maxIdTest];  
%              [~,maxIdPredict] = max(yscore(iTest,labelIndices));
%              ypredAll = [ypredAll maxIdPredict]; 
%         end        
%     end
    
    end
end