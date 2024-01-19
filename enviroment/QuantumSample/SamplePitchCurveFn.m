function data = SamplePitchCurveFn(data, methodsnum,PitchVersion)
% this function includes several different methods to sample pitch curve 
% methodsnum: 1(Method1) is diff; 2(Method2) is gradient
% PitchVersion: 
global data;

data = PitchVersionSelectFn(data,PitchVersion);
Nfiles = length(data);
for ifiles = 1:Nfiles
    switch methodsnum
        case 1
            %%
            %---------sample the pitch curve by finding the extremums and inflection points of the pitch trace---------
            MinPeakDistance = 0;
            % format: time pitch frame slope typeindicator
            [data(ifiles).peaks data(ifiles).troughs data(ifiles).extremums] = FindLocalextremumsFn(data(ifiles).F0pitch,data(ifiles).F0time,MinPeakDistance);
            data(ifiles).inflections = FindInflectionPointFn(data(ifiles).F0pitch, data(ifiles).F0time, data(ifiles).extremums);
            %---------sample the pitch curve by finding local extremums where the derivative is 0 and maximam of derivative between each peak and trough ---------
        case 2
            %% use the gradient(doesn't work well)
            time = data(ifiles).F0time;
            pitch = data(ifiles).F0pitch;
            [InflecTime, InflecPitch] = FindAllInflectionPointFn(time,pitch,1);
            figure();
            plot(data(ifiles).OriPitch.time,data(ifiles).OriPitch.pitch,'-y');
            hold on;
            plot(data(ifiles).time,data(ifiles).pitch,'-b');
%             hold on;
%             plot(InflecTime,InflecPitch,'or');
    end

end

%%
%---------find the local extremas of second derivative of pitch curve (1st-velocity; 2nd: Force; 3rd: intension)---------
% secondDpitch = diff( diff(pitch) );
% data(ifiles).secondDextremum = FindLocalextremumsFn( secondDpitch,time(2:end-1),0 );
% Textremums2 = data(ifiles).secondDextremum.extremums(:,1);
% Pextremums2 = pitch( data(ifiles).secondDextremum.extremums(:,3) );
%%
%---------find the local extremas of third derivative of pitch curve (1st-velocity; 2nd: Force; 3rd: intension)---------
% ThirdDpitch = diff( diff( diff(pitch) ) );
% data(ifiles).thirdDextremum = FindLocalextremumsFn( ThirdDpitch,time(2:end-2),0 );
% Textremums3 = data(ifiles).thirdDextremum.extremums(:,1);
% Pextremums3 = pitch( data(ifiles).thirdDextremum.extremums(:,3) );