% Function: attributeSelectDiscrete
% Description: Selects attributes and corresponding phoneme power values based on attribute IDs.
% Inputs:
%   - phonemePower: Phoneme power values (3D array: channels x time x phonemes)
%   - attributeId: IDs of attributes to be selected
%   - pLabel: Phoneme labels
% Outputs:
%   - attributePower: Attribute-specific phoneme power values (3D array: channels x time x attribute-specific phonemes)
%   - aLabel: Attribute labels corresponding to attributePower
%   - attributeAll: List of all attributes

function [attributePower, aLabel, attributeAll] = attributeSelectDiscrete(phonemePower, attributeId, pLabel)
    % Get the list of all attributes
    attributeAll = attribute2phoneme([],'list');
    % Select only the attributes specified by attributeId
    attributeAll = attributeAll(attributeId);

    % Convert each phoneme label to its corresponding attribute
    attrib = cell(1, length(pLabel));
    for pID = 1:length(pLabel)
        attrib{pID} = phoneme2attribute(pLabel(pID));
    end

    aLabel = []; % Initialize attribute label array
    attributePower = []; % Initialize attribute-specific phoneme power array

    % Iterate through each phoneme and attribute
    for pID = 1:length(pLabel)
        for aID = 1:numel(attributeAll)
            pID

            % Check if the attribute of the current phoneme matches the current attribute
            if(~isempty(find(strcmp(attrib{pID}, attributeAll(aID)), 1)))
                % Append the phoneme power values to attributePower
                attributePower = cat(3, attributePower, phonemePower(:, :, pID));
                % Append the attribute ID to aLabel
                aLabel = [aLabel aID];
            end
        end
    end
end
