classdef phonemeSequenceTrialParser
   
    properties
        syllableUnit % 1 - C 2 - V
        phonemeUnit % 1 - 9: a, ae, i, u, b, p, v, g, k
        phonemeClass % 1 - 4: low, high, labial, dorsal
        phonoTactic % phonotactic probabilities
        tokenIdentity % TrialInfo trigger
        tokenName
        
    end
    methods
        function obj = phonemeSequenceTrialParser(trialInfo)
            load('phonotactic.mat');
            obj.syllableUnit = nan(length(trialInfo),3);
            obj.phonemeUnit = nan(length(trialInfo),3);
            obj.phonemeClass = nan(length(trialInfo),3);
            obj.phonoTactic = nan(length(trialInfo),10);
            obj.tokenIdentity = nan(length(trialInfo),1);
            if(iscell(trialInfo))
            
            
                for iTrial = 1:length(trialInfo)

                    if(iscell(trialInfo))
                        trialNames = (trialInfo{iTrial}.sound(1:end-4));
                    else
                        trialNames = (trialInfo(iTrial).sound(1:end-4));
                    end
                    trialNamesTemp = strrep(trialNames,'ae','z');
                    trialNamesTemp = num2cell(trialNamesTemp);
                    trialNamesTemp = strrep(trialNamesTemp,'z','ae');
                    
                  tokenIdentity(iTrial) = trialInfo{iTrial}.Trigger; 
                  obj.tokenName{iTrial} = trialNames;
                    
                    for iPhon = 1:3
                         [obj.syllableUnit(iTrial,iPhon),obj.phonemeClass(iTrial,iPhon),obj.phonemeUnit(iTrial,iPhon)] = phonemeEncoder(trialNamesTemp{iPhon});          
                    end
                    
                    

                    phonid = find(strcmp([PhonemeSequencingInfoS1.TOKEN],trialNames));

                    if(isempty(phonid))
                        trialNames                    
                    else
                        obj.phonoTactic(iTrial,:) = table2array(PhonemeSequencingInfoS1(phonid,2:11));
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
                        [obj.syllableUnit(iTrial,iPhon),obj.phonemeClass(iTrial,iPhon),obj.phonemeUnit(iTrial,iPhon)] = phonemeEncoder(phonAll{iPhon,iTrial});
                        
                    end  
                         phonid = find(strcmp([PhonemeSequencingInfoS1.TOKEN],phonSequence{iTrial}));
                    
                        if(isempty(phonid))                            
                           phonSequence{iTrial} 
                        else
                            obj.phonoTactic(iTrial,:) = table2array(PhonemeSequencingInfoS1(phonid,2:11));
                        end
                     obj.tokenName{iTrial} =  phonSequence{iTrial} ;
                end
                
            end
           
        end
    
    end
end