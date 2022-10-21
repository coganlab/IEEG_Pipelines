function [phonemeWord,phonemeCell] = phonemeDecoder(phonemeSequence)

    for iTrial = 1:size(phonemeSequence,1)
        wordTemp = ''; 
        for iLength = 1:size(phonemeSequence,2)
            switch(phonemeSequence(iTrial,iLength))
                case 1
                    phonemeLabel = 'a';
                case 2
                    phonemeLabel = 'z';
                case 3
                    phonemeLabel = 'i';
                case 4
                    phonemeLabel = 'u';
                case 5
                    phonemeLabel = 'b';
                case 6
                   phonemeLabel = 'p';
                case 7
                    phonemeLabel = 'v';
                case 8
                    phonemeLabel = 'g';
                case 9
                    phonemeLabel = 'k';
            end
            wordTemp = strcat(wordTemp,phonemeLabel);
            phonemeCell{iTrial,iLength} = phonemeLabel;
        end
       
        phonemeWord{iTrial} =  strrep(strcat('/',wordTemp,'/'),'z','ae');
    end
end