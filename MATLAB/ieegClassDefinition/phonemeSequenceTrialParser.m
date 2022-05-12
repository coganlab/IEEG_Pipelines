classdef phonemeSequenceTrialParser
   
    properties
        syllableUnit % 1 - C 2 - V
        phonemeUnit % 1 - 9: a, ae, i, u, b, p, v, g, k
        phonemeClass % 1 - 4: low, high, labial, dorsal
        phonoTactic % phonotactic probabilities
        
    end
    methods
        function obj = phonemeSequenceTrialParser(trialInfo)
            load('phonotactic.mat');
            syllableUnit = nan(length(trialInfo),3);
            phonemeUnit = nan(length(trialInfo),3);
            phonemeClass = nan(length(trialInfo),3);
            phonoTactic = nan(length(trialInfo),10);
            if(iscell(trialInfo))
            
            
                for iTrial = 1:length(trialInfo)

                    if(iscell(trialInfo))
                        trialNames = (trialInfo{iTrial}.sound(1:end-4));
                    else
                        trialNames = (trialInfo(iTrial).sound(1:end-4));
                    end
                    trialNames = strrep(trialNames,'ae','z');
                    trialNames = num2cell(trialNames);
                    trialNames = strrep(trialNames,'z','ae');
                    for iPhon = 1:3
                         [syllableUnit(iTrial,iPhon),phonemeClass(iTrial,iPhon),phonemeUnit(iTrial,iPhon)] = phonemeEncoder(trialNames{iPhon});          
                    end

                    phonid = find(strcmp([PhonemeSequencingInfoS1.TOKEN],trialNames));

                    if(isempty(phonid))
                        trialNames                    
                    else
                        phonoTactic(iTrial,:) = table2array(PhonemeSequencingInfoS1(phonid,2:11));
                    end
                end
            else
                phon1 = [trialInfo.phon1];
                phon2 = [trialInfo.phon2];
                phon3 = [trialInfo.phon3];
                phonAll = [phon1; phon2; phon3 ];
                phonSequence =strcat(phon1,phon2,phon3);
                
                for iTrial = 1:size(phonAll,2)
                    for iPhon = 1:3
                        [syllableUnit(iTrial,iPhon),phonemeClass(iTrial,iPhon),phonemeUnit(iTrial,iPhon)] = phonemeEncoder(phonAll{iPhon,iTrial});
                        
                    end  
                         phonid = find(strcmp([PhonemeSequencingInfoS1.TOKEN],phonSequence{iTrial}));
                    
                        if(isempty(phonid))                            
                           phonSequence{iTrial} 
                        else
                            phonoTactic(iTrial,:) = table2array(PhonemeSequencingInfoS1(phonid,2:11));
                        end
                    
                end
                
            end
            obj.syllableUnit = syllableUnit;
            obj.phonemeClass = phonemeClass;
            obj.phonemeUnit = phonemeUnit;
            obj.phonoTactic = phonoTactic;
            
        end
    
    end
end