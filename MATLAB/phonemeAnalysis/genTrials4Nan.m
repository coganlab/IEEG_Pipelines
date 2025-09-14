function featureGen = genTrials4Nan(features, labels, options)
    arguments
        features double
        labels double
        options.genType = 1; % 1 - SMOTE, 2 - Mix-up
    end

    featureGen = features;
    emptyTrial = isnan(features);
    
    % Find trial indices with empty (NaN) features
    emptyTrialIndices = find(any(emptyTrial, 2));

    for iTrial = emptyTrialIndices'
        % Identify the label to smote
        label2Pad = labels(iTrial);

        % Select all the ids of the same labels except for the chosen label
        remainingLabelIds = find(labels == label2Pad & (1:numel(labels)) ~= iTrial);

        if length(remainingLabelIds) < 2
            continue;
        end

        % Find sparse feature points
        emptyTrialFeature = find(emptyTrial(iTrial, :));

        iTer = 1;
        while true
            remLabelSample = randsample(remainingLabelIds, 2, false);

            if all(~isnan(features(remLabelSample, emptyTrialFeature)))
                break;
            end

            iTer = iTer + 1;

            if iTer == 100
                disp('Generation not successful');
                break;
            end
        end

        genfeature1 = features(remLabelSample(1), emptyTrialFeature);
        genfeature2 = features(remLabelSample(2), emptyTrialFeature);

        switch options.genType
            case 1
                featureGen(iTrial, emptyTrialFeature) = computeSmote(genfeature1, genfeature2);
            case 2
                featureGen(iTrial, emptyTrialFeature) = computeMixup(genfeature1, genfeature2);
        end
    end

    disp(['Total number of nan features before trial generation: ' num2str(sum(emptyTrial(:)))]);
    disp(['Total number of nan features after trial generation: ' num2str(sum(isnan(featureGen(:))))]);
end

% function featureGen = genTrials4Nan(features,labels,options)
% arguments
%     features double
%     labels double
%     options.genType = 1; % 1 - SMOTE, 2 - Mix-up
% end
%   featureGen = features;
%         emptyTrial = isnan(features);
%         % Finding sparse ids
%         for iTrial = 1:size(emptyTrial,1)
%             % Iterating through each trial
%             if(sum(emptyTrial(iTrial,:))==0)
%                 % If no sparse ids, just continue
%                 continue;
%             end
%             % Identify the label to smote
%             label2Pad = labels(iTrial);
%             % Select all the ids of same labels except for the chosen
%             % label
%             remainingLabelIds = setdiff(find(ismember(labels,label2Pad)),iTrial);
%             if(length(remainingLabelIds)<2)
%                 continue;
%             end
% 
%             % Find sparse feature points
%             emptyTrialFeature = find(emptyTrial(iTrial,:));
% 
% %             for iTime = 1:length(emptyTrialFeature)
% %                 iTime
%                 % Iterating through the sparse feature point
%                 iTer = 1;
%                 while true
%                     remLabelSample = datasample(remainingLabelIds,2,'Replace',false);
%                     if((sum(isnan(features(remLabelSample(1),emptyTrialFeature)))==0) && (sum(isnan(features(remLabelSample(2),emptyTrialFeature)))==0) )
%                         break;
%                     end
%                     iTer = iTer + 1;
% 
%                     if(iTer==100)
%                         disp('Generation not successful')
%                         break;
%                     end
% 
%                 end
%                 genfeature1 = features(remLabelSample(1),emptyTrialFeature);
%                 genfeature2 = features(remLabelSample(2),emptyTrialFeature);
%                 switch(options.genType)
%                     case 1
%                         featureGen(iTrial,emptyTrialFeature) = computeSmote(genfeature1,genfeature2);
%                     case 2
%                         featureGen(iTrial,emptyTrialFeature) = computeMixup(genfeature1,genfeature2);
%                 end
% %             end
%         end
%         disp(['Total number of nan features before trial generation : ' num2str(sum(isnan(features),"all"))])
%         disp(['Total number of nan features after trial generation: ' num2str(sum(isnan(featureGen),"all"))])
% end