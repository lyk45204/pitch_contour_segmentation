function [A] = readfiles(filePath)    

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

    %%
    n_files = length(fileInf);
    A = cell(n_files,1);
    for i = 1:n_files
        fileNameSuffix = fileInf(i).name;
        if isnumeric(fileNameSuffix) == 0
            %if the user doesn't cancel, then read the audio
            fullPathName = strcat(filePath,fileNameSuffix);
            [M] = readmatrix(fullPathName);
            A{i} = M;
        end
    end