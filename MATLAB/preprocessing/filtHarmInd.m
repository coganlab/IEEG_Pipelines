function ieegfilt = filtHarmInd(ieeg,fs,fFilt)
%filtNum = fFilt;
% ieeg: channels x timepoints
% fs: sampling Frequency (Hz)
% fFilt: frequency to filter (60)
d = designfilt('bandstopiir','FilterOrder',20, ...
               'HalfPowerFrequency1',fFilt-1,'HalfPowerFrequency2',fFilt+1, ...
               'DesignMethod','butter','SampleRate',fs);
for i = 1:size(ieeg,1)             
       ieegfilt(i,:)=filtfilt(d,ieeg(i,:));             
end
%fFilt = 60;
% q = 4; 
%  bw = (fFilt/(fs/2))/q;
% [filtb,filta] = iircomb(round(fs/fFilt),bw,'notch');
% for i = 1:size(ieeg,1)     
%        ieegfilt(i,:)=filtfilt(filtb,filta,ieeg(i,:));       
% end
end