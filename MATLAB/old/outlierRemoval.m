function [goodIdx badIdx]=outlierRemoval(sig1,threshVal)

[m s]=normfit(sig1);
badIdx=find(abs(sig1)>(threshVal*s+m));
goodIdx=setdiff(1:length(sig1),badIdx);