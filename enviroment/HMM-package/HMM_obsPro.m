%% Observation probablity( Prob of observation given the state)
% input: 
%    pitchDistr: nS,nPPS,nSPP,minPitch, Obdistri, 
%    probablity of voiced: pitchtrack, voicedProb
%    calculateobsProb: yinTrust
% output: obsProb: row stores each frame and column stores each pitch step in three states
function ObsProb = HMM_obsPro(nS,nPPS,nSPP,minPitch, Obdistri, pitchtrack, voicedProb, yinTrust)
%% pitch state vector
iPitch = 0:1:nS*nPPS-1; %every step of pitch
minPitch = 35; %the lowest pitch
mu = minPitch + iPitch/nPPS; % a vector which include every mu of their pitch

%% calculate prior probablity of voiced
% set the voicedProb of every frame
n_frame = size(pitchtrack,1);
pitch_value = zeros(n_frame,1);
pitch_prob = zeros(n_frame,1);

pitch_value = pitchtrack(:,2);
pitch_prob(:) = voicedProb;
pitch_prob( find(pitch_value==0) ) = 0;

pIsPitched = pvoiced(pitch_prob); %row vector which stores each frame's  probality of voiced
%% calculate observation probablity
% get pitch_deva matrix
n = nS*nPPS*nSPP;
pitch_deva = zeros(n_frame,n);
% calculate the pitch_deva of frame with pitch
pitchtrack_rep = repmat(pitch_value,1,n);
mu_state = sort(repmat(mu,1,nSPP)); % 35,35,35,35.333,35.333,...
mu_rep = repmat(mu_state,n_frame,1);

pitch_deva = pitchtrack_rep - mu_rep;

loc_withoutpitch = find(pitch_value==0);
loc_withpitch = find(pitch_value~=0);
pitch_deva(loc_withoutpitch,:) = [];
%calculate observation probablity 
ObsProb_temp = zeros(n_frame,n); %row stores each frame and column stores each pitch step containing each state
n_obstate = size(Obdistri,1);

c_flip = 0; % to show if the pitch deviation should be flipped to fit the exp distribution
for i_state = 1:n_obstate
    
    if contains(Obdistri{i_state,1},"up",'IgnoreCase',true) % 
        c_flip = 1; % means need flip
    else
        c_flip = 0;
    end
%     k = pdf( Obdistri{i_state,2},(-1).^c_flip*pitch_deva(:,i_state:nSPP:end) );
%     kk = pitch_prob().^yinTrust;
    ObsProb_temp(loc_withpitch,i_state:nSPP:end) = pitch_prob(loc_withpitch).^yinTrust .* pdf( Obdistri{i_state,2},(-1).^c_flip*pitch_deva(:,i_state:nSPP:end) );
    ObsProb_temp(loc_withoutpitch,i_state:nSPP:end) = 1;
end

% normalization
ObsProb_nor = ObsProb_temp;
tempProbSum = sum(ObsProb_temp,2);
loc_withProb = find(tempProbSum>0); % avoid the prob equal to 0 and get NaN
ObsProb_nor(loc_withProb,:) = ObsProb_temp(loc_withProb,:)./tempProbSum(loc_withProb).*pIsPitched(loc_withProb); 

% insert observation probablity of silent states
ObsProb = ObsProb_nor;
ObsProb_sil = (1-pIsPitched) ./ (nPPS * nS); %silence state
ObsProb(:,nSPP:nSPP:end) = repmat(ObsProb_sil,1,n/nSPP);
end