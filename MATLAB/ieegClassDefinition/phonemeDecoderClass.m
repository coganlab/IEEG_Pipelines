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
        function decodeResultStruct = baseClassify(obj,ieegStruct,phonemeStruct,name, d_time_window,sigChannel,isauc)
            if(nargin<5)
                d_time_window = ieegStruct.tw;
                sigChannel = 1:size(ieegStruct.data,1);
                isauc = 0;
            end
            if(nargin<6)
                sigChannel = 1:size(ieegStruct.data,1);
                isauc = 0;
            end
            if(nargin<7)                
                isauc = 0;
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
                        [~,ytestAll,ypredAll,~,aucAll] = pcaLinearDecoderWrap(ieegInput,decoderUnit(:,iPhon)',ieegStruct.tw,d_time_window,obj.varExplained,obj.numFold,isauc);
                      
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
                    decodeResultStruct.aucAll(iPhon,:) = mean(aucAll);
            end            
        end
        function decodeResultStruct = baseRegress(obj,ieegStruct,phonemeStruct,name, d_time_window,sigChannel, rTrials)
            if(nargin<4)
                d_time_window = ieegStruct.tw;             
            end
            if(nargin<5)
                sigChannel = 1:size(ieegStruct.data,1);                
            end 
            if(nargin<6)
                rTrials = 1:size(ieegStruct.data,2);                
            end 
            trainTestDiff = 0;
            if(sum(size(d_time_window))==4)
                trainTestDiff = 1;
            end
                
            
            switch name
                case 'POS1'
                    decoderUnit = phonemeStruct.phonoTactic(rTrials,1)';
                case 'POS2'
                    decoderUnit = phonemeStruct.phonoTactic(rTrials,2)';
                case 'POS3'
                    decoderUnit = phonemeStruct.phonoTactic(rTrials,3)';
                case 'BiP1'
                    decoderUnit = phonemeStruct.phonoTactic(rTrials,4)';
                case 'BiP2'
                    decoderUnit = phonemeStruct.phonoTactic(rTrials,5)';
                case 'Pfwd1'
                    decoderUnit = phonemeStruct.phonoTactic(rTrials,6)';
                case 'Pfwd2'
                    decoderUnit = phonemeStruct.phonoTactic(rTrials,7)';
                case 'Pbkwd1'
                    decoderUnit = phonemeStruct.phonoTactic(rTrials,8)';
                case 'Pbkwd2'
                    decoderUnit = phonemeStruct.phonoTactic(rTrials,9)';
                case 'BLICK'
                    decoderUnit = phonemeStruct.phonoTactic(rTrials,10)';
            end
            
            
            ieegInput = ieegStruct.data(sigChannel,rTrials,:);
            assert(size(ieegInput,2)==length(decoderUnit),'Input/output trial mismatch');
            if(trainTestDiff==0)
                [ytestAll,ypredAll] = pcaLinearRegressDecoderWrap(ieegInput,decoderUnit,ieegStruct.tw,d_time_window,obj.varExplained,obj.numFold);
            else
                [ytestAll,ypredAll] = pcaLinearRegressDecoderWrapTrainTest(ieegInput,decoderUnit,ieegStruct.tw,d_time_window(1,:),d_time_window(2,:),obj.varExplained,obj.numFold);
            end
            distMod = fitlm(ytestAll,ypredAll);
            r2 = distMod.Rsquared.Ordinary;
            pVal = distMod.Coefficients.pValue(2);
            
            decodeResultStruct.r2 = r2;
            decodeResultStruct.pVal = pVal;            
        end            
        function accTime = tempGenClassify(obj,ieegStruct,phonemeStruct,name,timeRes,timeWin,sigChannel)
            timeRange = ieegStruct.tw(1):timeRes:ieegStruct.tw(2)-timeWin;
            accTime = zeros(3,length(timeRange));
            for iTime = 1:length(timeRange)
                decodeResultStruct = baseClassify(obj,ieegStruct,phonemeStruct,name,[timeRange(iTime) timeRange(iTime)+timeWin],name,sigChannel);
                accTime(:,iTime) = decodeResultStruct.accPhoneme';
            end
        end  
        function decodeTimeStruct = tempGenRegress1D(obj,ieegStruct,phonemeStruct,name,timeRes,timeWin,sigChannel, rTrials)
            timeRange = ieegStruct.tw(1):timeRes:ieegStruct.tw(2)-timeWin;
            r2Time = zeros(1,length(timeRange));
            pvalTime = nan(1,length(timeRange));
            for iTime = 1:length(timeRange)
                decodeResultStruct = baseRegress(obj,ieegStruct,phonemeStruct,name,[timeRange(iTime) timeRange(iTime)+timeWin],sigChannel, rTrials);
                r2Time(iTime) = decodeResultStruct.r2;
                pvalTime(iTime) = decodeResultStruct.pVal;
            end
            decodeTimeStruct.r2Time = r2Time;
            decodeTimeStruct.pvalTime = pvalTime;
            decodeTimeStruct.timeRange = timeRange;
        end  

        function decodeTimeStruct = tempGenRegress2D(obj,ieegStruct,phonemeStruct,name,timeRes,timeWin,sigChannel, rTrials)
            timeRange = ieegStruct.tw(1):timeRes:ieegStruct.tw(2)-timeWin;
            r2Time = zeros(length(timeRange),length(timeRange));
            pvalTime =nan(length(timeRange),length(timeRange));
            for iTimeTrain = 1:length(timeRange)    
                tsTrain = [timeRange(iTimeTrain) timeRange(iTimeTrain)+timeWin];
                for iTimeTest = 1:length(timeRange)        
                    tsTest = [timeRange(iTimeTest) timeRange(iTimeTest)+timeWin];
                    decodeResultStruct = baseRegress(obj,ieegStruct,phonemeStruct,name,[tsTrain; tsTest],sigChannel, rTrials);
                    r2Time(iTimeTrain,iTimeTest) = decodeResultStruct.r2;
                    pvalTime(iTimeTrain,iTimeTest) = decodeResultStruct.pVal;
                end
            end
            decodeTimeStruct.r2Time = r2Time;
            decodeTimeStruct.pvalTime = pvalTime;
            decodeTimeStruct.timeRange = timeRange;
        end

        function decoderChanStruct = indChanClassify(obj,ieegStruct,phonemeStruct,name,d_time_window)
            for iChan = 1:size(ieegStruct.data,1)
                iChan
                decoderChanStruct{iChan} = baseClassify(obj,ieegStruct,phonemeStruct,name,d_time_window,iChan,1);
            end
        end
        
    end
end