function [pred,LL] = predict_eeg_events_idx(eegs,stims,erp,nonlin,dly,idx,varargin)
% For an ERP model, compute the time-varying probability of an event based
% on the measured EEG. Use only the indexes specified.
% Inputs:
% - eegs = cell array of recorded EEG for each trial
% - stims = cell array of stimuli used for each trial. Each trial must be
% an array where each index is 1 = event, 0 otherwise
% - erp = ERP represented as a vector (channel x time)
% - nonlin = parameters for the sigmoidal nonlinearity
% - dly = delays of the ERP, in indexes
% Outputs:
% - pred = prediction, time-varying probability
% - LL = log-likelihood of the prediction
% Nate Zuk (2018)

% Initial variables
if nargin<6, % if idx isn't specified, use all indexes
    totidx = sum(cellfun(@(x) size(x,1),stims));
    idx = 1:totidx; 
end
parse_data = 1; % flag to specify if the EEG and stim should be indexed,
    % otherwise, it is assumed that eegs and stims are full matrices,
    % lagGen has already been applied to the eeg data, and the appropriate
    % values have been indexed (this is faster if the function is repeated
    % multiple times)

% Parse varargin
if ~isempty(varargin),
    for n = 2:2:length(varargin),
        eval([varargin{n-1} '=varargin{n};']);
    end
end

if parse_data,
    % Convert to cell array if the eeg and stim aren't already
    if ~iscell(eegs), eegs = {eegs}; end
    if ~iscell(stims), stims = {stims}; end

    % Create the X and Y matrices used
    fprintf('Creating X and Y matrices...');
    mattm = tic;
    X = cell(length(eegs),1);
    for jj = 1:length(eegs),
        X{jj} = lagGen(eegs{jj},dly);
    end
    x = zscore(cell_to_time_samples(X,idx));
    y = cell_to_time_samples(stims,idx);
    clear X
    fprintf('Completed @ %.3f s\n',toc(mattm));
else
%     warning('Parse_data was set to 0, assuming X and Y matrices are eegs and stims inputs respectively');
    x = eegs; y = stims;
end

% Compute the linear response
resp = x*erp;

% Compute the prediction
pred = glmval(nonlin,resp,'logit');

% Compute the log-likelihood
pbinom = pred.^(y).*(1-pred).^(1-y);
LL = sum(log(pbinom));