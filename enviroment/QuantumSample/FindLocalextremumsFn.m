function [peaks troughs extremums] = FindLocalextremumsFn(pitch,time,MinPeakDistance,EndPoints)
% @ this function is used to find the local extremums by using findpeaks
% function.
%   input： 
%       pitch and time vector of the pitch curve.
%       MinPeakDistance: the min distance of two peaks when using findpeaks
%       EndPoints: 0 is not include; 1 is include
%   output：
%       peaks and troughs

% 
% peaks
[peaks.pitch,peaks.time] = findpeaks(pitch,time,'MinPeakDistance',MinPeakDistance);
tempPeaks = [peaks.time,peaks.pitch];
peaks.slope = zeros(length(peaks.pitch),1);
peaks.type = ones(length(peaks.pitch),1);
% troughs
[troughs.pitch,troughs.time] = findtroughsFn(pitch,time,MinPeakDistance);
troughs.pitch = -troughs.pitch;
tempTroughs = [troughs.time,troughs.pitch];
troughs.slope = zeros(length(troughs.pitch),1);
troughs.type = zeros(length(troughs.pitch),1);

% plot
% hold on;
% plot(PeakLocs,pks,'*k');
% hold on;
% plot(TroughLocs,trghs,'*y');

% sort peaks and troughs
[extremums] = sortPeaksTroughsFn(tempPeaks,tempTroughs); % time pitch indicator of peak or trough
% get the indexes
% extre_idx_time = zeros(length(extremums),1)
% for i = 1:length(extremums)
%     [v,extre_idx_time(i)] = min(abs(extremums(i)-time))
% end

% Include the start point and end point if they are not same with the first
% extrem or the last
if EndPoints == 1

        if size(extremums,1)>0
            extremums_first_pitch = extremums(1,2);
            extremums_first_indi = extremums(1,3);
            if extremums_first_pitch ~= pitch(1)
                % transfer the indicator
                first_indi = 1 - extremums_first_indi;
                extremums = [time(1) pitch(1) first_indi; extremums ];
            end
            extremums_last_pitch = extremums(end,2);
            extremums_last_indi = extremums(end,3);
            if extremums_last_pitch ~= pitch(end)
                % transfer the indicator
                last_indi = 1 - extremums_last_indi;
                extremums = [ extremums; time(end) pitch(end) last_indi];
            end
        else
            direction = sign(pitch(1)-pitch(end));
            switch direction
                case -1 % up
                    first_indi = 0;
                case 1  % down
                    first_indi = 1;
                case 0  % flat
                    first_indi = 0;
            end
            last_indi = 1-first_indi;

            extremums = [time(1) pitch(1) first_indi; time(end) pitch(end) last_indi];
        end

end

extremums = [extremums(:,1:2)  zeros(size(extremums,1),1) extremums(:,3)];













