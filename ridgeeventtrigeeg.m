function [ERP,dly,X,Y] = ridgeeventtrigeeg(eeg,stim,Fs,mindly,maxdly,lambdas,idx,varargin)
% Compute the beat-triggered EEG signal (averaged correlation between beat
% times and the time-delayed EEG). Include a ridge parameter.
% Inputs:
% - eeg = cell array containing EEG per trial
% - stim = cell array of stimulus vector for each trial (must contain 1s or 0s)
% - Fs = sampling frequency (Hz)
% - maxdly = maximum possible delay of the model (ms)
% - lambda = ridge regression lambda value
% Outputs:
% - optmdl = optimized model, fit to all trials
% - stats = stats returned from lassoglm
% Nate Zuk (2017)

totidx = sum(cellfun(@(x) size(x,1),stim)); % get total idx
if nargin < 7 || isempty(idx), % if idx isn't specified...
    idx = 1:totidx; % use all indexes
end

verbose = 1;

% Parse varargin
if ~isempty(varargin),
    for n = 2:2:length(varargin),
        eval([varargin{n-1} '=varargin{n};']);
    end
end

dly = -floor(mindly/1000*Fs):-1:-ceil(maxdly/1000*Fs);

% Create the matrices for X and Y
% Y = stim; % just use the stimulus arrays for Y
if verbose, fprintf('Creating X and Y matrices...'); end
mattm = tic;
Xc = cell(length(eeg),1);
for jj = 1:length(eeg),
    Xc{jj} = zscore(lagGen(eeg{jj},dly)); % z-scoring added 10/11/18
end
X = cell_to_time_samples(Xc,idx);
Y = cell_to_time_samples(stim,idx);
clear Xc
if verbose, fprintf('Completed @ %.3f s\n',toc(mattm)); end
% Compute XTX and XTy
% [xtx,xty] = compute_linreg_matrices(X,Y,idx,[],'verbose',verbose);
[xtx,xty] = compute_linreg_matrices(X,Y,[],[],'verbose',verbose);
% Determine the number of events (for normalizing the ERP)
nevt = sum(cellfun(@(x) sum(x),stim));
% Compute the ERP model for each lambda value
M = eye(size(xtx));
ERP = NaN(size(xtx,1),length(lambdas));
if verbose, fprintf('Computing ERP model (%d iterations)',length(lambdas)); end
for l = 1:length(lambdas),
    if verbose, fprintf('.'); end
    ERP(:,l) = totidx/nevt*(xtx+M*lambdas(l))^(-1)*xty;
end
if verbose, fprintf('\n'); end