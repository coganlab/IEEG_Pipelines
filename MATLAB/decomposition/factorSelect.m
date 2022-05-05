function factorSelect(sig2analyzeAllFeature,labels,nFactors)
    cvp = cvpartition(labels,'KFold',10,'Stratify',true);
    linearTemplate = templateDiscriminant('DiscrimType','pseudolinear');
    for f = 1:nFactors
     %covMatrix = covCor(sig2analyzeAllFeature');   
    [Lambda,Psi,T,stats,F] = factoranupdate(sig2analyzeAllFeature,1,'xtype','data','scores','wls','rotate','varimax','maxit',1000);
    
    end
end