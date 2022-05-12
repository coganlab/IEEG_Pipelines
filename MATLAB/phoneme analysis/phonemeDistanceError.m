function [phonError,cmatvect,phonemeDistVect] = phonemeDistanceError(CMatNorm, pOccur)
    load('phonemeCH.mat');
    
    phonemeDist = squareform(pdist(phonemeCH,'hamming')).*17;
    cmatvect = CMatNorm(:);
    phonemeDistVect = phonemeDist(:);
%     size(CMatNorm)
%     size(phonemeDist(pOccur,pOccur))
    cmatError = CMatNorm.*phonemeDist(pOccur,pOccur);
    
    phonError = sum(cmatError(:))/(2*length(pOccur)); 
    
end