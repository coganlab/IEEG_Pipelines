classdef decoderClass  
   
    properties
       
        numFold double = 20
        varExplained (1,:) double = 80
        nIter double = 3
    end
    methods
        function obj = decoderClass(numFold,varExplained, nIter)
           
            obj.numFold = numFold;
            obj.varExplained = varExplained;      
            obj.nIter = nIter;     
            
        end
        function decompStruct = tsneDecompose(obj,ieegStruct,decoderUnit ,chanMap ,d_time_window,nElec,nIter)
            % only for micro-ECoG
            arguments
                obj {mustBeA(obj,'decoderClass')} % decoder class object
                ieegStruct {mustBeA(ieegStruct,'ieegStructClass')} % ieeg class object
                decoderUnit double {mustBeVector} % Decoder labels
                chanMap double % 2D channel Map 
                d_time_window double = ieegStruct.tw; % decoder time window; Defaults to epoch time-window
                nElec double = size(ieegStruct.data,1);% number of electrodes for analysis; Defaults to total
                nIter double = 50 % Number of iterations; Defaults to 50
            end
            assert(size(ieegStruct.data,2)==length(decoderUnit),'Input/output trial mismatch');    
            [decompStruct.silScore,decompStruct.silRatio] = tsneScoreExtract(ieegStruct.data,decoderUnit,ieegStruct.tw,chanMap,d_time_window,obj.varExplained ,nElec,nIter);
        end
        
        function decodeResultStruct = baseClassify(obj,ieegStruct,decoderUnit, d_time_window,selectChannel,selectTrial,isAuc)
            arguments
                obj {mustBeA(obj,'decoderClass')} % decoder class object
                ieegStruct {mustBeA(ieegStruct,'ieegStructClass')} % ieeg class object
                decoderUnit double {mustBeVector} % Decoder labels
                d_time_window double = ieegStruct.tw; % decoder time window; Defaults to epoch time-window
                selectChannel double = 1:size(ieegStruct.data,1); % select number of electrodes for analysis; Defaults to all
                selectTrial double = 1:size(ieegStruct.data,2); % select number of trials for analysis; Defaults to all
                isAuc logical = 0; % Select for AUC metric. Defaults to 0
            end
            
            
            trainTestDiff = 0;
            if(sum(size(d_time_window))==4)
                trainTestDiff = 1;
            end
            assert(size(ieegStruct.data,2)==length(decoderUnit),'Input/output trial mismatch');
            ieegInput = ieegStruct.data(selectChannel,selectTrial,:);
            decoderUnit = decoderUnit(selectTrial);
            


            decodeResultStruct.accPhoneme = 0;
            decodeResultStruct.accPhonemeUnBias = 0;         
            
            
            decodeUnitUnique = unique(decoderUnit); 
            decodeResultStruct.cmat = zeros(1,length(decodeUnitUnique),length(decodeUnitUnique));          
            
                    
            CmatAll = zeros(length(decodeUnitUnique),length(decodeUnitUnique));
            ytestall = [];
            for iTer = 1:obj.nIter
                if(trainTestDiff==0)
                    [~,ytest,ypred,~,~,modelWeightsAll] = pcaLinearDecoderWrap(ieegInput,decoderUnit,ieegStruct.tw,d_time_window,obj.varExplained,obj.numFold,isAuc);
                    %[~,ytest,ypred] = stmfDecodeWrap(ieegInput,decoderUnit,ieegStruct.tw,d_time_window,obj.numFold,isauc);
                else
                    [~,ytest,ypred,~,~,modelWeightsAll] = pcaLinearDecoderWrapTrainTest(ieegInput,decoderUnit,ieegStruct.tw,d_time_window(1,:), d_time_window(2,:), obj.varExplained,obj.numFold,isAuc);
                end
                ytestall = [ytestall ytest];
                Cmat = confusionmat(ytest,ypred);
                CmatAll = CmatAll + Cmat;
            end
            
            CmatCatNorm = CmatAll./sum(CmatAll,2);
            
            decodeResultStruct.accPhonemeUnBias = trace(CmatAll)/sum(CmatAll(:));
            decodeResultStruct.accPhoneme = trace(CmatCatNorm)/size(CmatCatNorm,1);  

            decodeResultStruct.cmat = CmatCatNorm;
            
            decodeResultStruct.p = StatThInv(ytestall,decodeResultStruct.accPhoneme.*100);           
            decodeResultStruct.modelWeights = modelWeightsAll;
        end

        function decodeResultStruct = baseRegress(obj,ieegStruct,decoderUnit, d_time_window,selectChannel,selectTrial)
           arguments
                obj {mustBeA(obj,'decoderClass')} % decoder class object
                ieegStruct {mustBeA(ieegStruct,'ieegStructClass')} % ieeg class object
                decoderUnit double {mustBeVector} % Decoder labels
                d_time_window double = ieegStruct.tw; % decoder time window; Defaults to epoch time-window
                selectChannel double = 1:size(ieegStruct.data,1); % select number of electrodes for analysis; Defaults to all
                selectTrial double = 1:size(ieegStruct.data,2); % select number of trials for analysis; Defaults to all
               
            end
            
            
            trainTestDiff = 0;
            if(sum(size(d_time_window))==4)
                trainTestDiff = 1;
            end
            assert(size(ieegStruct.data,2)==length(decoderUnit),'Input/output trial mismatch');
            ieegInput = ieegStruct.data(selectChannel,selectTrial,:);
            decoderUnit = decoderUnit(selectTrial);

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
        function decodeTimeStruct = tempGenClassify1D(obj,ieegStruct,decoderUnit,options)
            arguments
                obj {mustBeA(obj,'decoderClass')} % decoder class object
                ieegStruct {mustBeA(ieegStruct,'ieegStructClass')} % ieeg class object
                decoderUnit double {mustBeVector} % Decoder labels
                options.timeRes double = 0.02; % decoder time resolution; Defaults to 0.02
                options.timeWin double = 0.2;
                options.selectChannels double = 1:size(ieegStruct.data,1); % select number of electrodes for analysis; Defaults to all
                options.selectTrials double = 1:size(ieegStruct.data,2); % select number of trials for analysis; Defaults to all
               
            end

            timeRes = options.timeRes;
            timeWin = options.timeWin;
            selectChannels = options.selectChannels;
            selectTrials = options.selectTrials;
            timeRange = ieegStruct.tw(1):timeRes:ieegStruct.tw(2)-timeWin;
            accTime = zeros(1,length(timeRange));
            pValTime = zeros(1,length(timeRange));
            nTime = length(timeRange);
            modelweightTime = cell(1,length(timeRange));
            
            parfor iTime = 1:nTime
                decodeResultStruct = baseClassify(obj,ieegStruct,decoderUnit,[timeRange(iTime) timeRange(iTime)+timeWin],selectChannels,selectTrials,0);
                accTime(iTime) = decodeResultStruct.accPhoneme;
                pValTime(iTime) = decodeResultStruct.p;
                modelweightTime{iTime} = extractPcaLdaModelWeights(decodeResultStruct.modelWeights,...
                    size(decodeResultStruct.cmat,1),length(selectChannels),timeWin.*ieegStruct.fs);
            end
            decodeTimeStruct.accTime = accTime;
            decodeTimeStruct.timeRange = timeRange;
            decodeTimeStruct.pValTime = pValTime;
            decodeTimeStruct.modelweightTime = modelweightTime;
        end
        function decodeTimeStruct = tempGenClassify2D(obj,ieegStruct,decoderUnit,options)
             arguments
                obj {mustBeA(obj,'decoderClass')} % decoder class object
                ieegStruct {mustBeA(ieegStruct,'ieegStructClass')} % ieeg class object
                decoderUnit double {mustBeVector} % Decoder labels
                options.timeRes double = 0.02; % decoder time resolution; Defaults to 0.02
                options.timeWin double = 0.2;
                options.selectChannels double = 1:size(ieegStruct.data,1); % select number of electrodes for analysis; Defaults to all
                options.selectTrials double = 1:size(ieegStruct.data,2); % select number of trials for analysis; Defaults to all
               
             end
            timeRes = options.timeRes;
            timeWin = options.timeWin;
            selectChannels = options.selectChannels;
            selectTrials = options.selectTrials;
            timeRange = ieegStruct.tw(1):timeRes:ieegStruct.tw(2)-timeWin;
            accTime = zeros(length(timeRange),length(timeRange));
            pValTime = zeros(length(timeRange),length(timeRange));
            %f = waitbar(0, 'Starting');
            nTime = length(timeRange);
            
%             modelweightTime = [];
            parfor iTimeTrain = 1:nTime
                tsTrain = [timeRange(iTimeTrain) timeRange(iTimeTrain)+timeWin];
                for iTimeTest = 1:nTime        
                    tsTest = [timeRange(iTimeTest) timeRange(iTimeTest)+timeWin];                  
                    decodeResultStruct = baseClassify(obj,ieegStruct,decoderUnit,[tsTrain; tsTest],selectChannels,selectTrials,0);
                    accTime(iTimeTrain,iTimeTest) = decodeResultStruct.accPhoneme;
                    pValTime(iTimeTrain,iTimeTest) = decodeResultStruct.p;
%                     modelweightTimeTemp = extractPcaLdaModelWeights(decodeResultStruct.modelWeights,...
%                     size(decodeResultStruct.cmat,1),length(selectChannels),timeWin.*ieegStruct.fs);
%                     modelweightTime{iTimeTrain,iTimeTest} = modelweightTimeTemp.modelweightChan;
                end
%                 waitbar(iTimeTrain/length(timeRange), f, sprintf('Progress: %d %%', floor(iTimeTrain/length(timeRange)*100)));
%                 pause(0.1);
            end
            %close(f);
            decodeTimeStruct.accTime = accTime;   
            decodeTimeStruct.pValTime = pValTime; 
            decodeTimeStruct.timeRange = timeRange;
%             decodeTimeStruct.modelweightTime = modelweightTime;
        end
        
        function decodeTimeStruct = tempGenRegress1D(obj,ieegStruct,decoderUnit,options)
            arguments
                obj {mustBeA(obj,'decoderClass')} % decoder class object
                ieegStruct {mustBeA(ieegStruct,'ieegStructClass')} % ieeg class object
                decoderUnit double {mustBeVector} % Decoder labels
                options.timeRes double = 0.02; % decoder time resolution; Defaults to 0.02
                options.timeWin double = 0.2;
                options.selectChannels double = 1:size(ieegStruct.data,1); % select number of electrodes for analysis; Defaults to all
                options.selectTrials double = 1:size(ieegStruct.data,2); % select number of trials for analysis; Defaults to all
               
             end
            timeRes = options.timeRes;
            timeWin = options.timeWin;
            selectChannels = options.selectChannels;
            selectTrials = options.selectTrials;
            timeRange = ieegStruct.tw(1):timeRes:ieegStruct.tw(2)-timeWin;
            r2Time = zeros(1,length(timeRange));
            pValTime = nan(1,length(timeRange));
            nTime = length(timeRange);
           
            
            parfor iTime = 1:nTime
                decodeResultStruct = baseRegress(obj,ieegStruct,decoderUnit,[timeRange(iTime) timeRange(iTime)+timeWin],selectChannels,selectTrials);
                r2Time(iTime) = decodeResultStruct.r2;
                pValTime(iTime) = decodeResultStruct.pVal;
            end
            decodeTimeStruct.r2Time = r2Time;
            decodeTimeStruct.pValTime = pValTime;
            decodeTimeStruct.timeRange = timeRange;
        end  
        function decodeTimeStruct = tempGenRegress2D(obj,ieegStruct,decoderUnit,options)
            arguments
                obj {mustBeA(obj,'decoderClass')} % decoder class object
                ieegStruct {mustBeA(ieegStruct,'ieegStructClass')} % ieeg class object
                decoderUnit double {mustBeVector} % Decoder labels
                options.timeRes double = 0.02; % decoder time resolution; Defaults to 0.02
                options.timeWin double = 0.2;
                options.selectChannels double = 1:size(ieegStruct.data,1); % select number of electrodes for analysis; Defaults to all
                options.selectTrials double = 1:size(ieegStruct.data,2); % select number of trials for analysis; Defaults to all
               
             end
            timeRes = options.timeRes;
            timeWin = options.timeWin;
            selectChannels = options.selectChannels;
            selectTrials = options.selectTrials;
            timeRange = ieegStruct.tw(1):timeRes:ieegStruct.tw(2)-timeWin;
            nTime = length(timeRange);
            r2Time = zeros(length(timeRange),length(timeRange));
            pValTime =nan(length(timeRange),length(timeRange));
            %f = waitbar(0, 'Starting');
           
            
            parfor iTimeTrain = 1:nTime   
                tsTrain = [timeRange(iTimeTrain) timeRange(iTimeTrain)+timeWin];
                for iTimeTest = 1:nTime        
                    tsTest = [timeRange(iTimeTest) timeRange(iTimeTest)+timeWin];
                    decodeResultStruct = baseRegress(obj,ieegStruct,decoderUnit,[tsTrain; tsTest],selectChannels,selectTrials);
                    r2Time(iTimeTrain,iTimeTest) = decodeResultStruct.r2;
                    pValTime(iTimeTrain,iTimeTest) = decodeResultStruct.pVal;                    
                end
                %waitbar(iTimeTrain/length(timeRange), f, sprintf('Progress: %d %%', floor(iTimeTrain/length(timeRange)*100)));
            end
            %close(f);
            decodeTimeStruct.r2Time = r2Time;
            decodeTimeStruct.pValTime = pValTime;
            decodeTimeStruct.timeRange = timeRange;
        end
        
    end
end