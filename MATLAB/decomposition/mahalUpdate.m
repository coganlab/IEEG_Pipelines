function dist = mahalUpdate(noise,signal,sigma)
meanN = mean(noise,1)';
% sigmaInv = inv(sigma);
nTrials = size(noise,1);
for tr = 1:nTrials
    dist(tr) = sqrt((signal(tr,:)'-meanN)'/sigma*(signal(tr,:)'-meanN));
end
end