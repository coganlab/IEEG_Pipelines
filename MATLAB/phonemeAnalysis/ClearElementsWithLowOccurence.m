% Function: ClearElementsWithLowOccurence
% Description: Removes elements from an array with low occurrence frequencies.
% Inputs:
%   - array: Input array
%   - minimalFrequency: Minimum occurrence frequency for an element to be considered
% Outputs:
%   - array: Updated array with elements removed
%   - indicesToRemove: Indices of the removed elements
%   - elemRemoved: Elements that were removed

function [array, indicesToRemove, elemRemoved] = ClearElementsWithLowOccurence(array, minimalFrequency)
    % Get unique elements in the array
    elements = unique(array);
    indicesToRemove = []; % Initialize array for indices of elements to remove
    elemRemoved = []; % Initialize array for removed elements
    
    % Iterate through each unique element
    for i = 1:length(elements)
        indeces = find(array == elements(i)); % Find indices of the current element
        if (length(indeces) < minimalFrequency)
            % If the occurrence frequency is less than minimalFrequency,
            % add the indices and element to the arrays for removal
            indicesToRemove = [indicesToRemove indeces];
            elemRemoved = [elemRemoved elements(i)];
        end
    end
    
    % Remove elements from the array using the indicesToRemove
    array(indicesToRemove) = [];
end
