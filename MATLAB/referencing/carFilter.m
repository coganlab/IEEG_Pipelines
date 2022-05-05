function ieegFilt = carFilter(ieeg)
% ieeg - channels x trials x time
ieegFilt = ieeg - mean(ieeg,1);
end