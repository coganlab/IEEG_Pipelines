function goodTrialsCommon = extractCommonTrials(goodTrials)
%aL =  1:length(goodTrials);
goodTrialsCommon = goodTrials{1};
for na = 2:length(goodTrials)
  goodTrialsCommon = intersect(goodTrialsCommon,goodTrials{na});
end
end