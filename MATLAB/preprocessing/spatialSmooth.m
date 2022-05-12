function ieegSpaceSmooth = spatialSmooth(ieeg,chanMap,window)
    ieegSpace = spatialMap(ieeg,chanMap);
    h = ones(window)./(window(1).*window(2))
    ieegSpaceSmooth = imfilter(ieegSpace,h);
    ieegSpaceSmooth = reshape(ieegSpaceSmooth,size(ieeg));
end