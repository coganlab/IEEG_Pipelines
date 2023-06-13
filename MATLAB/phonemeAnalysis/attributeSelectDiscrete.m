function [attributePower,aLabel,attributeAll] = attributeSelectDiscrete(phonemePower,attributeId,pLabel)
attributeAll = attribute2phoneme([],'list');
attributeAll = attributeAll(attributeId)

    for pID = 1:length(pLabel)
        attrib{pID} = phoneme2attribute(pLabel(pID));    
    end
    aLabel = []; attributePower = [];
    for pID = 1:length(pLabel)
        for aID = 1:numel(attributeAll)        
            pID
            if(~isempty(find(strcmp(attrib{pID},attributeAll(aID)), 1)))
                attributePower = cat(3,attributePower, phonemePower(:,:,pID));
                aLabel = [aLabel aID];
            end
        end
    end
end