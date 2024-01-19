function bestSequence = HMM(hiddenStates, hiddenStatesLabel, observations, initialProb, transitionProMatrix, emissionProDistribution)
    % Validate input dimensions and types
    % ... [Add any necessary input validation code here] ...

    % Number of hidden states and observations
    numStates = numel(hiddenStates);
    numObservations = size(observations,1);

    % Calculate the emission probability matrix
    emissionProMatrix = zeros(numStates, numObservations);
    for i = 1:numStates
        currentState = hiddenStates{i};
        currentStateEmissionDist = emissionProDistribution.(currentState);
        density = currentStateEmissionDist.density;
        X = currentStateEmissionDist.X; % feature 1 meshgrid
        Y = currentStateEmissionDist.Y; % feature 2 meshgrid
        
        for j = 1:numObservations
            currentObservation = observations(j,:);
            emissionProMatrix(i, j) = calculateDensityAtPoint(currentObservation(1), currentObservation(2), X, Y, density);

            % Check if the calculated density value is NaN
            if isnan(emissionProMatrix(i, j)) 
                error('NaN value detected at j = %d, currentObservation = [%f, %f]', j, currentObservation(1), currentObservation(2));
            end
            if emissionProMatrix(i, j)<0
                error('negative value detected at j = %d, currentObservation = [%f, %f]', j, currentObservation(1), currentObservation(2));
            end
        end

    end

    % Infer the best sequence of hidden states
    %Viterbi_decoder
    % input: init, transProb_array, obsProb, nState, nFrame, nTrans
    % output: the best path which contain state and pitch
   
    [bestSequence, p] = hmmViterbi_(emissionProMatrix, transitionProMatrix, initialProb); % use Mochen's code
    % Return the best sequence
    

end

function densityValue = calculateDensityAtPoint(xVal, yVal, X, Y, density)
    % Function to calculate the density at a given point (xVal, yVal)
    % using the provided density matrix and corresponding X and Y grids.
    
    
    % Interpolate the density matrix to find the density at (xVal,
    % yVal),extrapval set the outrange value is the min
    densityValue = interp2(X, Y, density, xVal, yVal, 'linear',min(density(:)));
end
