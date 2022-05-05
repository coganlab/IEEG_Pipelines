function ieegSplit = splitIeeg(ieeg,trigOnset,tw,fs)
   % ieeg = [];
     for iTrial = 1:length(trigOnset)   
          iTrial
%             round(tw(1)*fs)+trigOnset(t)
%             trigOnset(t)+round(fs*tw(2))
            ieegSplit(:,iTrial,:) = ieeg(:,round(tw(1)*fs)+trigOnset(iTrial):trigOnset(iTrial)+round(fs*tw(2)));
     end
     ieegSplit = ieegSplit(:,:,1:end-1);
end