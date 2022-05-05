classdef phonemeDecoderClass  
   
    properties
       
        numFold
        varExplained
        
    end
    methods
        function obj = phonemeDecoderClass(numFold,varExplained)
           
            obj.numFold = numFold;
            obj.varExplained = varExplained;          
            
        end
        function decompStruct = tsneDecompose(obj,ieegStruct,phonemeStruct,name, d_time_window,nElec,nIter)
            
            
            
            switch name
                case 'phoneme'
                    decoderUnit = phonemeStruct.phonemeUnit;
                case 'class'
                    decoderUnit = phonemeStruct.phonemeClass;
                case 'syllable'
                    decoderUnit = phonemeStruct.syllableUnit;
            end
            size(decoderUnit)
            [decompStruct.silScore,decompStruct.silRatio] = tsneScoreExtract(ieegStruct.data,decoderUnit(:,1)',ieegStruct.tw,d_time_window,obj.varExplained,ieegStruct.chanMap,nElec,nIter);
        end
        function decodeResultStruct = baseDecoder(obj,ieegStruct,phonemeStruct,name, d_time_window,sigChannel)
            if(nargin<5)
                d_time_window = ieegStruct.tw;
                sigChannel = 1:size(ieegStruct.data,1);
            end
            if(nargin<6)
                sigChannel = 1:size(ieegStruct.data,1);
            end
            decodeResultStruct.accPhoneme = zeros(1,3);
            decodeResultStruct.accPhonemeUnBias = zeros(1,3);
            decodeResultStruct.phonError = zeros(1,3);
            
            switch name
                case 'phoneme'
                    decoderUnit = phonemeStruct.phonemeUnit;
                case 'class'
                    decoderUnit = phonemeStruct.phonemeClass;
                case 'syllable'
                    decoderUnit = phonemeStruct.syllableUnit;
            end
            decodeUnitUnique = unique(decoderUnit); 
            decodeResultStruct.cmat = zeros(3,length(decodeUnitUnique),length(decodeUnitUnique));
            ieegInput = ieegStruct.data(sigChannel,:,:);
            assert(size(ieegInput,2)==size(decoderUnit,1),'Input/output trial mismatch');
            for iPhon = 1:3 % Iterating through each phoneme position
                    
                    CmatPhoneme = zeros(length(decodeUnitUnique),length(decodeUnitUnique));
                    for iTer = 1
                        [~,ytestAll,ypredAll] = pcaLinearDecoderWrap(ieegInput,decoderUnit(:,iPhon)',ieegStruct.tw,d_time_window,obj.varExplained,obj.numFold,0);
                      
                        CmatAll = confusionmat(ytestAll,ypredAll);
                        CmatPhoneme = CmatPhoneme + CmatAll;
                    end
                    CmatCatNorm = CmatPhoneme./sum(CmatPhoneme,2);
                    decodeResultStruct.accPhonemeUnBias(iPhon) = trace(CmatPhoneme)/sum(CmatPhoneme(:));
                    decodeResultStruct.accPhoneme(iPhon) = trace(CmatCatNorm)/size(CmatCatNorm,1);  
                    if(strcmp(name,'phoneme'))
                        [decodeResultStruct.phonError(iPhon)] = phonemeDistanceError(CmatCatNorm,decodeUnitUnique);
                    end
                    decodeResultStruct.cmat(iPhon,:,:) = CmatCatNorm;
            end            
        end
        function accTime = temporalGeneralization(obj,sigChannel,timeRes,timeWin,name)
            timeRange = obj.ieegStruct.tw(1):timeRes:obj.ieegStruct.tw(2)-timeWin;
            accTime = zeros(3,length(timeRange));
            for iTime = 1:length(timeRange)
                decodeResultStruct = baseDecoder(obj,sigChannel,[timeRange(iTime) timeRange(iTime)+timeWin],name);
                accTime(:,iTime) = decodeResultStruct.accPhoneme';
            end
        end       
        function decoderChanStruct = individualChannelDecoder(obj,d_time_window,name)
            for iChan = 1:size(obj.ieegStruct.data,1)
                decoderChanStruct{iChan} = baseDecoder(obj,iChan,d_time_window,name);
            end
        end
        
        
        
        
    end
end