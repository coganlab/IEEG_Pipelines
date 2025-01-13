function [featureGenAll,labelGenAll] = balanceSmoteTrials(features,labels,options)
    arguments
        features double
        labels double
        options.genType = 1; % 1 - SMOTE, 2 - Mix-up
    end
    featureGen = [];
    labelGen = [];
    [uniLabel,~,ix] = unique(labels);
    C = accumarray(ix,1)';
    maxCount = max(C);
    for iLabel = 1:length(uniLabel)
        
        % Iterating through each trial
        if(maxCount == C(iLabel))
            % If no sparse ids, just continue
            continue;
        end

        num2pad = maxCount - C(iLabel);
        % Identify the label to smote
        label2Pad = uniLabel(iLabel);
        % Select all the ids of same labels except for the chosen
        % label
        labelIds = (find(ismember(labels,label2Pad)));
        if(length(labelIds)<2)
            continue;
        end
        featurePad = [];
       for iPad = 1:num2pad
            remLabelSample = datasample(labelIds,2,'Replace',false);
            
            genfeature1 = features(remLabelSample(1),:);
            genfeature2 = features(remLabelSample(2),:);
            switch(options.genType)
                case 1
                    featurePad(iPad,:) = computeSmote(genfeature1,genfeature2);
                case 2
                    featurePad(iPad,:) = computeMixup(genfeature1,genfeature2);
            end
       end
       
       featureGen = cat(1,featureGen,featurePad);
       labelGen = [labelGen repmat(label2Pad,1,num2pad)];
    end

    featureGenAll = cat(1,features,featureGen);
    labelGenAll = [labels labelGen];
    size(labelGenAll)
end