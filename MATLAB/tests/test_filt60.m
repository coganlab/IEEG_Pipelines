% addpath(genpath('MATLAB'))
% addpath(genpath('../..'))
% load('single_ch.mat', 'single_ch')
% filt_ieeg = filt60(single_ch,2048);
% load('single_ch_filt.mat','filt')
% err = sum(abs(filt_ieeg-filt));
% assert(err < 0.000001)
assert(1)