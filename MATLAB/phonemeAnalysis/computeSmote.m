function trialGen = computeSmote(trial1,trial2)
%COMPUTESMOTE Compute smote trial
%   Detailed explanation goes here
trialDiff = trial1 - trial2;
trialGen = trial1 + rand(1).*trialDiff;

end

