function [psiRank,psiRankP,pComb] = phonemeSelective(phonemePower,pLabelId,pvalsMClean)
    phonUnique = unique(pLabelId);
    pComb = nchoosek(phonUnique,2);
    pComb = sortrows([pComb; fliplr(pComb)]);
    pCombId = nchoosek(1:length(phonUnique),2);
    pCombId = sortrows([pCombId; fliplr(pCombId)]);
    psiRankP = [];
    for iChan = 1:size(phonemePower,1)
        iChan
        for bClassId = 1 : size(pComb,1)
            
            X = phonemePower(iChan,ismember(pLabelId,pComb(bClassId,:)));
            Y = pLabelId(ismember(pLabelId,pComb(bClassId,:)));
            yval = pComb(bClassId,:);
            
            xval1 = rmoutliers(X(Y==yval(1)));
            xval2 = rmoutliers(X(Y==yval(2)));
            minx = min([length(xval1) length(xval2)]);
            pSamp = [];
            for iSamp = 1:100
                samp1 = datasample(xval1,minx,'Replace',false);
                samp2 = datasample(xval2,minx,'Replace',false);           
                pSamp(iSamp) = ranksum(samp1,samp2,'tail','right');
            end
             if (ismember(iChan,find(pvalsMClean)) && (bClassId > 682 && bClassId < 690))
            figure;
%             subplot(2,1,1);
%             h1 = histogram(X(Y==yval(1)),100);
%             hold on;
%             h2 = histogram(X(Y==yval(2)),100);            
%             subplot(2,1,2);
            h1 = histogram(samp1,25);
            hold on;
            h2 = histogram(samp2,25);
            end
            psiRankP(iChan,pCombId(bClassId,1),pCombId(bClassId,2)) = mean(pSamp);           
        end
        psiMat = squeeze(psiRankP(iChan,:,:));
        psiMark = psiMat<0.01;
        figure;
        imagesc(double(psiMark'));
        colormap(grey);
        psiRank(iChan,:) = sum(psiMark);
    end
end