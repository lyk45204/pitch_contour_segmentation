function [InflectionPoint] = FindInflectionPointFn(pitch, time, extremums)
% @ this function is used to find the maximam of derivative between each peak and trough
%   input： 
%       pitch and time vector of the pitch curve.
%       
%   output：
%       Inflection Point where the slope get the maximam


Timeextremes = extremums(:,1);
Pitchextremes = extremums(:,2);
dt = time(2) - time(1);

Inflec_idx_pitch = zeros(length(Timeextremes)-1,1);
Inflec_Time = zeros(length(Timeextremes)-1,1);
Inflec_Pitch = zeros(length(Timeextremes)-1,1);
Inflec_Slope = zeros(length(Timeextremes)-1,1);
Inflec_Type = zeros(length(Timeextremes)-1,1);

for i = 1:length(Timeextremes)-1
    i
    Tstart = Timeextremes(i);
    Tend = Timeextremes(i+1);
    Pstart = Pitchextremes(i);
    Pend = Pitchextremes(i+1);
    
    [v, Tstart_idx_time] = min( abs(time-Tstart) );
    
    Portion_idx_time = find( time>Tstart & time<Tend );
    pitchPortion = [Pstart; pitch(Portion_idx_time); Pend];
    
    slope = diff( pitchPortion )./dt;
    [v, Maxslope_idx_pitchportion] = max( abs( slope ) );
    Inflec_idx_pitch(i) = (Tstart_idx_time - 1) + Maxslope_idx_pitchportion;
    Inflec_Time(i) = time(Inflec_idx_pitch(i));
    Inflec_Pitch(i) = pitch(Inflec_idx_pitch(i));
    Inflec_Slope(i) = slope(Maxslope_idx_pitchportion);
    Inflec_Type(i) = 0.5;
   
end
InflectionPoint = [Inflec_Time Inflec_Pitch Inflec_Slope Inflec_Type ];









