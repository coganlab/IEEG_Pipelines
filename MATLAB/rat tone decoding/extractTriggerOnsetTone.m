function [locs,time] = extractTriggerOnsetTone(trigger,fs)
     [~,locs] = findpeaks(trigger,fs,'MinPeakDistance',0.5,'MinPeakHeight',0.8);
     time = [0:length(trigger)-1]./fs;
     figure;
     plot(time,trigger);
     hold on;
     scatter(locs,max(trigger)./2.*ones(1,length(locs)));
end