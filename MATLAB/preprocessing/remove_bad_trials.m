function [NumTrials,goodtrials] = remove_bad_trials(data,threshold)
    % Removes the bad trials based on threshold detectin
    % INPUT
    %       data - electrodes x trials x samples
    %       threshold - threshold of standard deviation to remove noisy
    %       trials
    % OUTPUT
    %       NumTrials - goodtrials in each channel
    %       goodtrials - trial indices after removing bad trials
    
thresh = threshold;

for iCh =1:size(data,1)
    tmp = squeeze(data(iCh,:,:));
    tmp = detrend(tmp);
    sd = std(tmp(:));
    %th = thresh*std(abs(tmp(:)))+mean(abs(tmp(:)));
    e = max(abs(tmp')); % Finds the maximum SINGLE point
    if thresh < 100   % ArtifactTheshold is in terms of SD
      %  th = thresh.*sd+mean(tmp(:)); 
        th = thresh.*sd;
    else
        th = 10*thresh;  % ArtifactTheshold is in terms of uV, account for preamp gain
    end
    NumTrials(iCh) = length(find(e<th));
    goodtrials{iCh} = find(e<th);
end

end