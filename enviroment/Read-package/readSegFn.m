function data = readSegFn(filePath,data)
% input: M : onset, offset, duration
% output: onset offset duration
global data;
fileInf = dir2(filePath);
 %% ignore folders and DS_Store file
    IgnoreFiles = [];
    n_files = length(fileInf);
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
for i_files = 1:n_files
    fileNameSuffix = fileInf(i_files).name;
    if isnumeric(fileNameSuffix) == 0
        fullPathName = strcat(filePath,fileNameSuffix);
        [filepath,name,extension] = fileparts(fullPathName);
        [PorSegs] = readmatrix(fullPathName);

        % check if any state is not labelled with the correct number
        labelNum = [-1, 0, 1, -2];

        % 找到不在labelNum中的元素
        not_in_labelNum = ~ismember(PorSegs(:,2), labelNum);

        % 获取这些元素的位置
        positions = find(not_in_labelNum);
        if length(positions)>0
            % 输出位置
            for i_positions = 1:length(positions)
                fprintf('the %d seg in the %d track is labelled with wrong number\n', positions(i_positions), i_files);
            end
            error('there are some segs with wrong label ')

        end
        
      
        data(i_files).State = PorSegs;
        
    end
end
