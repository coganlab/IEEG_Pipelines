function [ieegall,microphone,trigger] = intanWrap(path,channel,fileNum,isDecimate,isTask,isCar)

arguments
    path {mustBeTextScalar}
    channel double
    fileNum double
    isDecimate logical = 0
    isTask logical = 0

    isCar logical = 0

end
    d = dir([path '*.rhd']);
    d.name
    ieegall = []; microphone = []; trigger = [];
    for iFile = fileNum
        disp(strcat('Recording ',num2str(iFile)));
        fullPathToFile = [path d(iFile).name]        
        ieegsamp = [];
        [amplifier_data] = read_Intan_RHD2000_file_path_update(path,iFile,channel);
        if(isDecimate)
            ieegsamp = (resample(amplifier_data',2000,20000))';
        else
            ieegsamp = amplifier_data;
        end

        if(isCar)
            ieegsamp = carFilter(ieegsamp);
        end


%         for iChan = 1:size(channel,2)   
%             
%             %[amplifier_data] = read_Intan_RHD2000_file_path_update_EDIT_ELIM_OVERHEAD(fullPathToFile,channel(iChan));
%             if(isDecimate)
%                 ieegsamp(iChan,:) = decimate(amplifier_data(iChan,:),10);
%                 %ieegsamp(iChan,:) = decimate(amplifier_data,10);
%             else
%                 ieegsamp(iChan,:) = amplifier_data(iChan,:);
%                 %ieegsamp(iChan,:) = amplifier_data;
%             end
%         end
        ieegall =[ieegall ieegsamp];
        if(isTask)
            [~,board_adc_data] = read_Intan_RHD2000_file_path_update_EDIT_ELIM_OVERHEAD(fullPathToFile,channel(1));
            trigger = [trigger board_adc_data(1,:)];
            microphone = [microphone board_adc_data(2,:)];
        end
    end
end