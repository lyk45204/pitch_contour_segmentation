%% Viterbi_decoder
% description
% initial probablity£ºinit, (n,1)
% transition probablity: 
    % transProb_temp1, (5,nS*nPPS), A2A, A2S, S2S, S2s, s2s(S, stable; s, silence)
    % transProb_temp2, (75,nS*nPPS), 75 is max number of qualified distance
    % transProb_temp, (1, 15080)
    % from_temp, (1, 15080)
    % to_temp, (1, 15080)
% Observation probablity: obsProb, (n,1), one frame, each pitch step of
                                       % three states
% input: init, transProb_array, obsProb, nState, nFrame, nTrans
% output: path
function path = Viterbi_decoder(init, transProb_array, obsProb)
nState = size(transProb_array,1);
nFrame = size(obsProb,2);

% transMatrix must be square
checkTransition = size(transProb_array,2);
if checkTransition ~= nState
    error(message('stats:hmmviterbi:BadTransitions'));
end

% number of rows of observMatrix must be same as number of states
checkObservMatrix = size(obsProb,1);
if checkObservMatrix ~= nState
    error(message('stats:hmmviterbi:InputSizeMismatch'));
end
%% set parameters
delta = zeros(nState,1);
oldDelta = zeros(nState,1);
deltasum = 0;
scale = zeros(nFrame,1); % store the scale of every frame
psi = zeros(nFrame,nState); % store the best transitions of every frame 
path = zeros(nFrame,1); % store the best path for backward step, every element is a iState(a pitch of one state)
%% initialise first frame
oldDelta = init.* obsProb(1,:)';
% normalize oldDelta
deltasum = sum(oldDelta);
oldDelta = oldDelta/deltasum;
% scale of every frame
scale(1) = 1/deltasum;
psi = psi; % just correspond to the code of Tony
%% rest of forward step, observation is from 1 to 621(nState)£¬ transProb is every observ to others(nTrans), 
% oldDelta* transPro: rearrange the oldDelta by the index of from, multipy
% the transPro, and pick the max of every observation by the index of to,
% so that we can get the delta of next frame
% this idea make use of Matlab to avoid loop, so will modify the code of Tony
from = transProb_array(1,:);
to = transProb_array(2,:);
transProb = transProb_array(3,:);
h=waitbar(0,'please wait');
for iFrame = 2:nFrame
    str=['running',num2str(iFrame/nFrame*100),'%'];
    waitbar(iFrame/nFrame,h,str)
    deltasum = 0;
    psi = psi; %just correspond to the code of Tony
    oldDeltatemp = oldDelta(from);%rearrange the oldDelta by the index of from
    currentValue = oldDeltatemp.*transProb';%multipy the transPro
    %use the index of to get delta
    for i = 1:nState
        i_in_to = find(to==i); %the location of i in to
        [vm, lm] = max(currentValue(i_in_to));
        max_in_from = i_in_to(lm); % the location of max in to, which is equal to in from 
        delta(i) = vm; %max value
        psi(iFrame,i) = from(max_in_from); %the from value of the max value
    end
    delta = delta.*obsProb(iFrame,:)';
    deltasum = sum(delta);
    if deltasum>0
        oldDelta = delta/deltasum;
        delta(1:end) = 0;
        scale(iFrame) = 1/deltasum;
    else
        oldDelta(1:end) = 1/nState;
        delta(1:end) = 0;
        scale(iFrame) = 1
    end;
        
end

 %% initialise backward step
[bestvalue,iState] = max(oldDelta); %in the last frame, find the max value and the corresponding istate
currentValue = bestvalue;
path(nFrame) = iState;
%% rest of backward step
for iFrame = nFrame-1:-1:1
    path(iFrame) = psi(iFrame+1,path(iFrame+1));
end

end