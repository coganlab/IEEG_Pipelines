function ieegSpaceArrange = spatialMap(ieeg,chanMap)

    selectedChannels = sort(chanMap(~isnan(chanMap)))';
    ieegSpaceArrange = nan(size(chanMap,1),size(chanMap,2),size(ieeg,2));
    for c = 1 : length(selectedChannels)
        [cIndR, cIndC] = find(ismember(chanMap,selectedChannels(c)));
        ieegSpaceArrange(cIndR,cIndC,:)=ieeg(c,:);
    end
end