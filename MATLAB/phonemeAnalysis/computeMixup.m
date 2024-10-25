function trialGen = computeMixup(trial1,trial2)

 betaP = betarnd(2,2,1);
% trialDiff = trial1 - trial2;
trialGen = betaP.*trial1 + (1-betaP).*trial2;

end

