function [asiRank,asiRankP,attributeAll,aLabel] = attributeSelection(phonemePower, pLabel)
    attributeAll = attribute2phoneme([],'list');
    for pID = 1:length(pLabel)
        attrib{pID} = phoneme2attribute(pLabel(pID));    
    end
    attributePower = []; aLabel = [];
    for aID = 1:numel(attributeAll)
        for pID = 1:length(pLabel)
            if(~isempty(find(strcmp(attrib{pID},attributeAll(aID)), 1)))
                attributePower = [attributePower phonemePower(:,pID)];
                aLabel = [aLabel aID];
            end
        end
    end
    
    [asiRank,asiRankP,aComb] = phonemeSelective(attributePower,aLabel,zeros(size(attributePower,1)));
end