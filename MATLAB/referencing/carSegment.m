function car = carSegment(ieeg)
n = size(ieeg,1); % number of channels
c = 1:n;
for tr = 1:size(ieeg,2)
    tr
    ieegtrial = squeeze(ieeg(:,tr,:));
    ieegmean = mean(ieegtrial,1);
    corrIeegMean = ieegmean*ieegmean';
    for i = 1:n % Iterating through channels        
           car(i,tr,:) = ieegtrial(i,:) - (ieegtrial(i,:)*ieegmean').*ieegmean./corrIeegMean;            
    end
end
end