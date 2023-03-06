function locs = extractTriggerOnset(trigger,fs)
    
    time = [0:length(trigger)-1]./fs;
     figure;
     plot(time,trigger);
     tw(1) = input('Enter the starting time');
     tw(2) = input('Enter the ending time');
     amp = input('Enter the amplitude');
     mpd = input('Enter the minimum peak distance');
     timeSelectInd = time>=tw(1) & time <=tw(2);
     timeSelect = time(timeSelectInd);
     [~,locs] = findpeaks(trigger(timeSelectInd),fs,'MinPeakDistance',mpd,'MinPeakHeight',amp);
     locs = locs + tw(1);
     figure;
     plot(timeSelect,trigger(timeSelectInd));
     hold on;
     scatter(locs,max(trigger(timeSelectInd)).*ones(1,length(locs)));
end