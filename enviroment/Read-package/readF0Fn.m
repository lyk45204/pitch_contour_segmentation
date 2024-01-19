function data = readF0Fn(filePathRoot,indicator)
% this version of readF0 function is used to get different version of F0
% input:
%     unvoiced pitch indicator: 1 means no; 2 means have
% output:
%     original F0 (time,continous pitch, voiced/unvoiced indicator)

 
    global data;
%%
%---------Original pitch---------
switch indicator
    case 1
        filePath1 = strcat(filePathRoot, 'F0/');
        A1 = readfiles(filePath1);
        n_files = length(A1);
        for i_files = 1:length(A1)
            M1 = A1{i_files};
            data(i_files).OriPitch.time  = M1(:,1);
            data(i_files).OriPitch.pitch  = freqToMidi( M1(:,2) );
            data(i_files).OriPitch.voicedIndicator  = double(M1(:,2)>0);
            data(i_files).FsOri  = 1/(M1(2,1)-M1(1,1));
            data(i_files).F0voicing = data(i_files).OriPitch.voicedIndicator;
        end
        
        
    case 2
        filePath1 = strcat(filePathRoot, 'F0/');
        filePath2 = strcat(filePathRoot, 'F0_unvoiced/');
        A1 = readfiles(filePath1);
        n_files = length(A1);
        A2 = readfiles(filePath2);
        if length(A2) == n_files
            for i_files = 1:length(A1)
                M1 = A1{i_files}; M2 = A2{i_files};
                data(i_files).OriPitch.time  = M1(:,1);
                data(i_files).OriPitch.pitch  = freqToMidi( M1(:,2) );
                data(i_files).OriPitch.unvoicedPitch  = freqToMidi( M2(:,2) );
                data(i_files).OriPitch.voicedIndicator  = double(M2(:,2)>0);
                data(i_files).F0voicing = data(i_files).OriPitch.voicedIndicator;
            end
        else
            disp('The number of F0 files and F0_unvoiced files are different');
        end
        
end