function data = readVibratoGTFn(filePathRoot,data)
% input: M : onset, not related, duration
% output: onset offset duration
global data;
filePath = strcat(filePathRoot, 'Vibrato_GT/');
fileInf = dir2(filePath);

n_files = length(fileInf);
for i = 1:n_files
    fileNameSuffix = fileInf(i).name;
    if isnumeric(fileNameSuffix) == 0
        fullPathName = strcat(filePath,fileNameSuffix);
        [filepath,name,extension] = fileparts(fullPathName);
        switch extension
            case '.txt'
                [ groundTruthData ] = transferColerTruthData( fullPathName );
            case '.csv'
                [M] = readmatrix(fullPathName);
                if isempty(M)
                    groundTruthData = M;
                else
                    groundTruthData = [M(:,1) M(:,1)+M(:,3) M(:,3)]; % onset offset duration
                end
        end
        data(i).VibratoGT.Notelevel = groundTruthData;
        

        data(i).VibratoGT.Framelevel = VibrotoNote2Frame(groundTruthData, data(i).FDMframetime);
    end
end
    
   