classdef ieegStructMicro < ieegStructClass
    properties 
        chanMap
    end
    
    methods
        function obj = ieegStructMicro(data, fs, tw, fBand, name, chanMap)% class constructor
            obj@ieegStructClass(data, fs, tw, fBand, name);
            obj.chanMap = chanMap;
        end
        
        
        
        function ieegSpatialSmooth = spatialSmoothMicro(obj,window)
            dataTemp = obj.data;
            chanMapTemp = obj.chanMap;
            dataSmooth = zeros(size(dataTemp));
            for iTrial = 1:size(dataTemp,2)                
                dataSmooth(:,iTrial,:) = spatialSmooth(squeeze(dataTemp(:,iTrial,:)),chanMapTemp,window);                
            end
            ieegSpatialSmooth = ieegStructMicro(dataSmooth,obj.fs,obj.tw,obj.fBand,[obj.name '_' num2str(window(1)) 'x' num2str(window(2)) '_spatially_smoothed'], chanMapTemp);
            
        end
        
        function [ieegSpatialAverage,matrixPoints] = spatialAverage(obj,window,isOverlap)
            matrixPoints = matrixSubSample(obj.chanMap,window,isOverlap);
            chanMapAverage = zeros(floor(size(obj.chanMap,1)/window(1)),floor(size(obj.chanMap,2)/window(2)));
            dataAverage = zeros(size(matrixPoints,1),size(obj.data,2),size(obj.data,3));
            for iSA = 1:size(matrixPoints,1)
                dataAverage(iSA,:,:) = mean(obj.data(matrixPoints(iSA,:),:,:),1);
                chanMapAverage(iSA) = iSA;
            end
            ieegSpatialAverage = ieegStructMicro(dataAverage,obj.fs,obj.tw,obj.fBand,[obj.name '_' num2str(window(1)) 'x' num2str(window(2)) '_spatially_averaged'], chanMapAverage);
            
        end
        
        function ieegHiGammaNorm = extractHiGammaNorm(obj1,obj2,fDown,gtw1,gtw2) % extracting normalized high-gamma            
            ieegHiGammaTemp = extractHiGammaNorm@ieegStructClass(obj1,obj2,fDown,gtw1,gtw2);
            ieegHiGammaNorm = ieegStructMicro(ieegHiGammaTemp.data, ieegHiGammaTemp.fs, ieegHiGammaTemp.tw, ieegHiGammaTemp.fBand, ieegHiGammaTemp.name, obj1.chanMap);
        end
    end
end