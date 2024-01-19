function data = readFileNameFn(filePath)
%READAUDIOFUNCTION Summary of this function goes here
%   Detailed explanation goes here
    global data;
    %input audio : note the last '\' in the audiopath
%     [fileNameSuffix,filePath] = uigetfile({'*.wav';'*.mp3'},'Select File');
    br = batchreading;
    fileInf = dir2(filePath);
    n_files = length(fileInf);

    %% ignore folders and DS_Store file
    IgnoreFiles = [];
    for i = 1:n_files
        if strcmp(fileInf(i).name,'.DS_Store')
            IgnoreFiles = [IgnoreFiles;i];
            
        end
        if fileInf(i).isdir
            IgnoreFiles = [IgnoreFiles;i];

        end

    end
    fileInf([IgnoreFiles]) = [];

    %% read
    n_files = length(fileInf);

    for i = 1:n_files
        [~, Filename, ~] = fileparts(fileInf(i).name);;
        if isnumeric(Filename) == 0
            data(i).Filename = Filename;
        end
    end
