function [trghs ,TroughLocs] = findtroughsFn(pitch,time,MinPeakDistance)

    [trghs ,TroughLocs] = findpeaks(-pitch,time,'MinPeakDistance',MinPeakDistance);
%     % find the first trough
%     D = find( diff(pitch)>0 );
%     trghs = [pitch(D(1));-trghstemp];
%     TroughLocs = [time(D(1));TroughLocstemp];