% calculate transition probablity
% input: nSPP, nPPS, nS, minSemitoneDistance, maxJump, sigma2Note(the deviation of distantance distrubution); 
% output: transProb_array (from; to; transProb)
function transProb_array = HMM_transProb(nSPP, nPPS, nS, minSemitoneDistance, maxJump, sigma2Note)

%% transition probablity
%% Simple
transProb_temp1 = zeros(1,nS*nPPS*5);%every 5 elements are from a pitch step
pAttackSelftrans = 0.9; 
pAttack2Stable = 1-pAttackSelftrans;
pStableSelftrans = 0.99; 
pStable2Silent = 1-pStableSelftrans;
pSilentSelftrans = 0.9999;
transProb_temp1 = [pAttackSelftrans,pAttack2Stable,pStableSelftrans,pStable2Silent,pSilentSelftrans];

%% Complex, transition probablity of silence to attack(a vector)
%parameters
sd = 0:1/nPPS:maxJump; %according to the condition of distance
noteDistanceDistr = normpdf(sd, 0, sigma2Note); %obey Normal distribution, column vector
% calculate distance by collecting every ipitch and jpitch to replace calculating loops
iPitch = zeros(nS*nPPS,nS*nPPS);
jPitch = zeros(nS*nPPS,nS*nPPS);
iPitch(1,:) = 0:nS*nPPS-1;
iPitch = repmat(iPitch(1,:),nS*nPPS,1);
jPitch(:,1) = 0:nS*nPPS-1;
jPitch = repmat(jPitch(:,1),1,nS*nPPS);
% calculate semitoneDistance
fromPitch = iPitch;
toPitch = jPitch;
semitoneDistance = abs(fromPitch - toPitch) * 1.0 / nPPS;
% condition, find the distance satisfy the condition, row sequence is jPitch,
% column sequence is iPitch
[sd_r, sd_c] = find(semitoneDistance == 0 | (semitoneDistance > minSemitoneDistance & semitoneDistance < maxJump)); 
%% calculate 'from' and 'to' vector, from is the sequence of each pitch of states transition from, to is the sequence showing to what
% eg. 1 2 3 is three states of the first pitch 35,    4 5 6 is three states of
% the second pitch 35.333. 1to1 or 2, 2to2 or 3, 3to3 or 4 7 10.....; 
% from and to list all the possibilities of transition 
%find loop length of one iPitch
loopiendloc = find(sd_c(2:end)-sd_c(1:end-1)>0);
loopistartloc = [1;loopiendloc+1];
loopiendloc = [loopiendloc; length(sd_c)];
loopilen = loopiendloc-loopistartloc+1; %length of every loop
% calculate from
from_temp1 = [0,0,1,1,2]; %for simple transition when index = 0;
index_temp2 = iPitch(1,sd_c) * nSPP; 
from_temp2 = index_temp2+2; %for complex transition from = index+2
%insert from_temp1 to every loopstartloc of from_temp2
ft1len = length(from_temp1);
from = zeros(1, length(loopistartloc)*ft1len+length(from_temp2));
len_from = length(from); % store the length of from, because it will increase five every loop.
from(1:ft1len+length(from_temp2)) = [from_temp1, from_temp2];
for i = 1:length(loopiendloc)-1
    from = [from(1:ft1len*i+loopiendloc(i)),(from_temp1+nSPP*i),from(ft1len*(i)+loopiendloc(i)+1:len_from)];
end
from = from(1:len_from); % delete the last five zeros which is extruded in the last loop
%find jPitch loop boundary, which is equal to loop boundary of one iPitch
loopjendloc = find(sd_r(2:end)-sd_r(1:end-1)<0);
loopjstartloc = [1;loopjendloc+1];
loopjendloc = [loopjendloc; length(sd_r)];
loopjlen = loopjendloc-loopjstartloc+1; %length of every loop
% calculate to
to_temp1 = [0,1,1,2,2]; %for simple transition when index = 0;
toIndex__temp2 = jPitch(sd_r,1)' * nSPP; %toIndex = jPitch * par.nSPP
to_temp2 = toIndex__temp2; %for complex transition to = toIndex
%insert from_temp1 to every loopstartloc of from_temp2
tt1len = length(to_temp1);
to = zeros(1, length(loopjstartloc)*tt1len+length(to_temp2));
len_to = length(to);
to(1:tt1len+length(to_temp2)) = [to_temp1, to_temp2];
for i = 1:length(loopjendloc)-1
    to = [to(1:tt1len*i+loopjendloc(i)),(to_temp1+nSPP*i),to(tt1len*(i)+loopjendloc(i)+1:len_to)];
end
to = to(1:len_to);
%% calculate transProb
%pick up the qualified elements(satisfy the condition) using linear index
sd_l = sd_r+(sd_c-1)*nS*nPPS; %transfer index of row and column into linear 
idx_noteDistanceDistrtemp = round((semitoneDistance(sd_l)-0)/(1/nPPS))+1; %calculate the index in distribution of qualified elements to avoid beyond the range of horizontal distributon
tempWeightSilent = noteDistanceDistr(1,idx_noteDistanceDistrtemp);
tempTransProbSilent = zeros(max(loopilen),length(loopistartloc)); %store every tempWeightSilent vector, one column is for one iPitch loop
for i = 1:length(loopistartloc)
    tempTransProbSilent(1:loopilen(i),i) = tempWeightSilent(loopistartloc(i):loopiendloc(i))';%transfer tempWeightSilent into array to make it easy to get transProb_temp2
end
probSumSilent = sum(tempTransProbSilent); %sum each column, equal to sum each loop
transProb_temp2 = tempTransProbSilent./repmat(probSumSilent,max(loopilen),1)*(1-pSilentSelftrans);

%insert transProb_temp1 to transProb_temp2
transProb_temp1 = repmat(transProb_temp1',1,length(loopistartloc));
transProb_tempM = cat(1,transProb_temp1,transProb_temp2);
%transfer array to vector
transProb = reshape(transProb_tempM, 1, []);
transProb(find(transProb == 0)) = []; %delete all zeros which is reduious, produced from the process of making array
% modify from and to, since matlab squence is from 1
from = from+1;
to = to+1;
% synthesize a array to look transProb clearly
transProb_array = zeros(3,length(transProb));
transProb_array(1,:) = from;
transProb_array(2,:) = to;
transProb_array(3,:) = transProb;
end