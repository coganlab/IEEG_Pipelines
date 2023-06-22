% extractPcaLdaModelWeights - Extract model weights from PCA-LDA models.
%
% Syntax: modelweight = extractPcaLdaModelWeights(modelWeights, numClass, numChan, numTime)
%
% Inputs:
%   modelWeights       - Cell array of PCA-LDA models (each element represents a fold)
%   numClass           - Number of classes
%   numChan            - Number of channels
%   numTime            - Number of time points
%
% Output:
%   modelweight        - Structure containing model weights
%                         - modelweightChan: Model weights for each class and channel
%                         - modelweightchantime: Model weights for each class, channel, and time point
%
% Example:
%   modelWeights = {model1, model2, model3};
%   numClass = 3;
%   numChan = 8;
%   numTime = 100;
%   modelweight = extractPcaLdaModelWeights(modelWeights, numClass, numChan, numTime);
%


function modelweight = extractPcaLdaModelWeights(modelWeights, numClass, numChan, numTime)
    % Initialize variables to store model weights
    modelweightchantimefold = [];
    modelweightChan = [];
    
    % Loop over each fold of model weights
    for iFold = 1:length(modelWeights)
        modelTemp = modelWeights{iFold};
        modelweightall = [];
        classSelected = [];
        
        % Compute model weights for each pairwise class comparison
        for iClassx = 1:numClass-1
            for iClassy = iClassx+1:numClass
                modelweightall{iClassx, iClassy} = modelTemp.pcaScore * modelTemp.ldamodel.Coeffs(iClassy, iClassx).Linear;
                classSelected = [classSelected; [iClassx, iClassy]];
            end
        end
        
        modelweightclass = [];
        
        % Compute mean model weights for each class
        for iClass = 1:numClass
            [classIdx, classIdy] = find(ismember(classSelected, iClass));
            modelweightclasstemp = [];
            
            if(length(classIdx) == 1)
                modelweightclasstemp = modelweightall{1, 2}';
            else
                for iId = 1:length(classIdx)
                    modelweightclasstemp = [modelweightclasstemp; (modelweightall{classSelected(classIdy(iId), 1), classSelected(classIdx(iId), 2)})'];
                end
            end
            
            modelweightclass(iClass, :) = mean(modelweightclasstemp, 1);
        end
        
        modelweightclass = reshape(modelweightclass, numClass, numChan, numTime);
        modelweightchantimefold(iFold, :, :, :) = modelweightclass;
    end
    
    % Compute average model weights across folds
    modelweightchantime = squeeze(mean(modelweightchantimefold, 1));
    
    % Compute norm of model weights for each class and channel
    for iClass = 1:numClass
        for iChan = 1:numChan
            modelweightChan(iClass, iChan) = norm(squeeze(modelweightchantime(iClass, iChan, :)));
        end
    end
    
    % Store model weights in a structure
    modelweight.modelweightChan = modelweightChan;
    modelweight.modelweightchantime = modelweightchantime;
end
