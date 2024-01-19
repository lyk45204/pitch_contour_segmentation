function data = readAudioFn(filePath)
%READAUDIOFUNCTION Summary of this function goes here
%   Detailed explanation goes here
    global data;
    %input audio : note the last '\' in the audiopath
%     [fileNameSuffix,filePath] = uigetfile({'*.wav';'*.mp3'},'Select File');
    br = batchreading;
    fileInf = dir2(filePath);
    n_files = length(fileInf);
    for i = 1:n_files
        fileNameSuffix = fileInf(i).name;
        if isnumeric(fileNameSuffix) == 0
            %if the user doesn't cancel, then read the audio
            fullPathName = strcat(filePath,fileNameSuffix);
            [audio,data(i).fs] = audioread(fullPathName);
            
            %sum the two channels into one channel
            channels = size(audio,2);
            if channels > 1
                audio = sum(audio,2);
            end
            data(i).time = (1:size(audio,1))/data(i).fs;
            data(i).dur = (size(audio,1))/data(i).fs;
            splitResults = strsplit(fileNameSuffix,'.');
            data(i).fileName  = splitResults{1};
            suffix = splitResults{2};
            data(i).audio = audio;
            data(i).filePath = filePath;
            data(i).fileNameSuffix = fileNameSuffix;
            
            %         plotAudio(data.time,data.audio,data.axeWave,data.fileNameSuffix);
        end
        
    end

end

