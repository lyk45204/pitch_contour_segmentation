function [interpolatedTime,interpolatedSpitch] = interpolaFn(originalTime,originalPitch,InsertTimes)
    f0Fs = 1/(originalTime(2)-originalTime(1));
    interpolationF = InsertTimes*f0Fs;  %interpolation frequency
    interpolatedTime = [originalTime(1):1/interpolationF:originalTime(end)]';
    interpolatedSpitch = spline(originalTime,originalPitch,interpolatedTime);