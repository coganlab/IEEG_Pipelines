function ieegfilt = filt60(ieeg,fs)
%filtNum = [60:60:60*notches];
% for i = 1:length(filtNum)
%     f60 = filtNum(i);
%     q = 20; 
%     bw = (f60/(fs/2))/q;
%     [filtnumer(i,:),filtdenom(i,:)] = iirnotch((f60/(fs/2)),bw);
% end
% parfor i = 1:size(ieeg,1)   
%      
%        for k = 1:length(filtNum)
%        ieegfilt(i,:)=filtfilt(filtnumer(k,:),filtdenom(k,:),ieeg(i,:));      
%        end
% end
f60 = 60;
q = 10; 
 bw = (f60/(fs/2))/q;
[filtb,filta] = iircomb(round(fs/f60),bw,'notch');
for i = 1:size(ieeg,1) 
       
       ieegfilt(i,:)=filtfilt(filtb,filta,ieeg(i,:));      
       
end
end