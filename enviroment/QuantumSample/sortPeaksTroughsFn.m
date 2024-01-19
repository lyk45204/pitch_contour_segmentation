function [Extremum] = sortPeaksTroughsFn(tempPeaks,tempTroughs)
% this fuction aims to sort the peaks and troughs, which are found by the upstream
% findpeaks function. 
%(don't need into one peak one trough alternatively, because sometimes the trough of previous curve is the peak of the next
% curve. The only aim is to keep monotonicity of the curve between two
% local extremum)

% input and output format:
%   eg. tempPeaks: [time, pitchofPeaks]...


TempPeaks = [tempPeaks,ones( size(tempPeaks,1), 1)]; % use 1 to represent peak
TempTroughs = [tempTroughs,zeros( size(tempTroughs,1), 1)]; % use 0 to represent trough
TempExtremum = [TempPeaks;TempTroughs];
[Extremum] = sortrows(TempExtremum,1);
% check if anywhere there are two seccessive peaks or troughs
if any( diff( Extremum(:,3) ) == 0 )
    display('there are two seccessive peaks or troughs');
    return;
end
% peaks = Extremum( find(Extremum(:,3)==1),: );
% troughs = Extremum( find(Extremum(:,3)==0),: );
    
    
    
end

