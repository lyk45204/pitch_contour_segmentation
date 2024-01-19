function data = readMultiF0Fn(filePathRoot,indicator)
% this version of readF0 function is used to get different version of F0
% input:
%     unvoiced pitch indicator: 1 means no; 2 means have
% output:
%     original F0 (time,continous pitch, voiced/unvoiced indicator)
%     interpolated F0 with spline (time,continous pitch, voiced/unvoiced indicator)
%     smoothed F0 using moving average
%     smoothed + interpolated F0
%     interpolated + smoothed F0

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


%---------Original pitch---------

% check if there is duplicate ponits mistakely added on the time
        % track
    for i_files = 1:n_files
        time = data(i_files).OriPitch.time;
        % Find unique points and their first occurrence indices
        [uniquePoints, ia, ic] = unique(time, 'stable');
        
        % Find indices of all occurrences
        indices = 1:length(time);
        
        % Find duplicate indices
        duplicateIndices = find(ismember(indices, ia) == 0);
        

        if length(duplicateIndices)>0
            % Display duplicate data points along with their indices
            disp('Duplicate points and their indices:');
            for i = 1:length(duplicateIndices)
                disp(['Point: ', num2str(time(duplicateIndices(i))), ...
                    ', Index: ', num2str(duplicateIndices(i)), ...
                    ', Filename: ', data(i_files).Filename]);
            end

            error('The time track must contain unique values.')
        end
    end


%%
%---------do interpolation for the pitch using spline function---------
%Since we need to get the point in which the slope is 0, it is better to do
%interpolation to make the pitch trace more continious and avoid the
%straight lines
    InsertTimes = 5;
    for i_files = 1:n_files
        i_files
        time = data(i_files).OriPitch.time;
        pitch = data(i_files).OriPitch.pitch;

        [data(i_files).InterpoPitch.time,data(i_files).InterpoPitch.pitch] = interpolaFn(time,pitch,InsertTimes);
    end

%---------do interpolation for the pitch using spline function---------

%% 
%---------do smooth by using moving average---------
    for i_files = 1:n_files
        time = data(i_files).OriPitch.time;
        pitch = data(i_files).OriPitch.pitch;
        data(i_files).SmoothPitch.time = time;
        data(i_files).SmoothPitch.pitch = smooth(pitch,10);
    end
%---------do smooth by using moving average---------

%% 
%---------smoothed + interpolated F0---------
    InsertTimes = 5;
    for i_files = 1:n_files
        time = data(i_files).SmoothPitch.time;
        pitch = data(i_files).SmoothPitch.pitch;
        [Newtime,Newpitch] = interpolaFn(time,pitch,InsertTimes);
        data(i_files).SmoothInterpoPitch.time = Newtime;
        data(i_files).SmoothInterpoPitch.pitch = Newpitch;
    end
%---------smoothed + interpolated F0---------

%% 
%---------interpolated + smoothed F0---------
    for i_files = 1:n_files
        time = data(i_files).InterpoPitch.time;
        pitch = data(i_files).InterpoPitch.pitch;
        data(i_files).InterpoSmoothPitch.time = time;
        data(i_files).InterpoSmoothPitch.pitch = smooth(pitch,10);
    end
%---------interpolated + smoothed F0---------

%% plot five versions pitch curve
i_files = 2;
figure();
plot(data(i_files).OriPitch.time,data(i_files).OriPitch.pitch);
hold on;
plot(data(i_files).InterpoPitch.time,data(i_files).InterpoPitch.pitch);
hold on;
plot(data(i_files).SmoothPitch.time,data(i_files).SmoothPitch.pitch);
hold on;
plot(data(i_files).SmoothInterpoPitch.time,data(i_files).SmoothInterpoPitch.pitch);
hold on;
plot(data(i_files).InterpoSmoothPitch.time,data(i_files).InterpoSmoothPitch.pitch);
hold off;
legend('OriPitch','InterpoPitch','SmoothPitch','SmoothInterpoPitch','InterpoSmoothPitch');
title('Different versions of pitch trace')
xlabel('time/s')
ylabel('pitch/MIDI')



