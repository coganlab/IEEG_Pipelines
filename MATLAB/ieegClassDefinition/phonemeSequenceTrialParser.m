classdef phonemeSequenceTrialParser
    % The phonemeSequenceTrialParser class parses phoneme sequence trial information.
    
    properties
        syllableUnit % Syllable unit (1 - Consonant, 2 - Vowel)
        phonemeUnit % Phoneme unit (1 - 9: a, ae, i, u, b, p, v, g, k)
        phonemeClass % Phoneme class (1 - 4: low, high, labial, dorsal)
        phonoTactic % Phonotactic probabilities
        tokenIdentity % TrialInfo trigger
        tokenName % Token names
    end
    
    methods
        function obj = phonemeSequenceTrialParser(trialInfo)
            % Class constructor
            
            % Load phonotactic data
            load('phonotactic.mat');
            
            % Initialize properties
            obj.syllableUnit = nan(length(trialInfo), 3);
            obj.phonemeUnit = nan(length(trialInfo), 3);
            obj.phonemeClass = nan(length(trialInfo), 3);
            obj.phonoTactic = nan(length(trialInfo), 10);
            obj.tokenIdentity = nan(length(trialInfo), 1);
            obj.tokenName = cell(length(trialInfo), 1);
            
            if iscell(trialInfo)
                % Parsing for cell array input (multiple trials)
                
                for iTrial = 1:length(trialInfo)
                    % Extract trial names and preprocess
                    trialNames = trialInfo{iTrial}.sound(1:end-4);
                    trialNamesTemp = strrep(trialNames, 'ae', 'z');
                    trialNamesTemp = num2cell(trialNamesTemp);
                    trialNamesTemp = strrep(trialNamesTemp, 'z', 'ae');
                    
                    % Store token identity and token name
                    obj.tokenIdentity(iTrial) = trialInfo{iTrial}.Trigger;
                    obj.tokenName{iTrial} = trialNames;
                    
                    % Encode phonemes and syllables
                    for iPhon = 1:3
                        [obj.syllableUnit(iTrial, iPhon), obj.phonemeClass(iTrial, iPhon), obj.phonemeUnit(iTrial, iPhon)] = phonemeEncoder(trialNamesTemp{iPhon});
                    end
                    
                    % Find phonotactic probabilities
                    phonid = find(strcmp([PhonemeSequencingInfoS1.TOKEN], trialNames));
                    
                    if isempty(phonid)
                        trialNames
                    else
                        obj.phonoTactic(iTrial, :) = table2array(PhonemeSequencingInfoS1(phonid, 2:11));
                    end
                end
            else
                % Parsing for structure array input (single trial)
                
                % Extract phoneme information
                phon1 = [trialInfo.phon1];
                phon2 = [trialInfo.phon2];
                phon3 = [trialInfo.phon3];
                phonAll = [phon1; phon2; phon3];
                phonSequence = strcat(phon1, phon2, phon3);
                
                for iTrial = 1:size(phonAll, 2)
                    % Encode phonemes and syllables
                    for iPhon = 1:3
                        [obj.syllableUnit(iTrial, iPhon), obj.phonemeClass(iTrial, iPhon), obj.phonemeUnit(iTrial, iPhon)] = phonemeEncoder(phonAll{iPhon, iTrial});
                    end
                    
                    % Find phonotactic probabilities
                    phonid = find(strcmp([PhonemeSequencingInfoS1.TOKEN], phonSequence{iTrial}));
                    
                    if isempty(phonid)
                        phonSequence{iTrial}
                    else
                        obj.phonoTactic(iTrial, :) = table2array(PhonemeSequencingInfoS1(phonid, 2:11));
                    end
                    
                    % Store token name
                    obj.tokenName{iTrial} = phonSequence{iTrial};
                end
            end
        end
    end
end
