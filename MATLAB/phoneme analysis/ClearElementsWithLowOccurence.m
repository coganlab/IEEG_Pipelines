function [array,indicesToRemove,elemRemoved]= ClearElementsWithLowOccurence(array,minimalFrequency)
elements = unique(array);
indicesToRemove = []; elemRemoved = [];
for i = 1:length(elements)
   indeces = find(array==elements(i));
   if (length(indeces) < minimalFrequency)
      indicesToRemove = [indicesToRemove indeces];
      elemRemoved = [elemRemoved elements(i)];
   end
end
array(indicesToRemove) = [];