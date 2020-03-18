function [timeLL,t_center] = predacc_by_time(stims,preds,Fs,wnd,step)
% Compute the time-varying log-likelihood of an LNB model prediction. If
% stims and preds are cell arrays, they must be the same length, and timeLL
% will also be a cell array of the same length.
% Inputs:
% - stims = arrays of event onsets (1 for an event, 0 otherwise)
% - preds = model predictions of time-varying event probability
% - Fs = sampling rate (in Hz)
% - wnd = window size for computing log-likelihood (in ms)
% - step = step size for each successive window (in ms) (default: 0.5*wnd)
% Outputs:
% - timeLL = array of time-varying log-likelihood
% - t = center time of each window (in s)
% Nate Zuk (2020)

% Parse varargin
% if ~isempty(varargin),
%     for n = 2:2:length(varargin),
%         eval([varargin{n-1} '=varargin{n};']);
%     end
% end

% compute step size
if nargin<5, step = 0.5*wnd; end

% turn stims and preds into cell arrays, if they aren't already
if ~iscell(stims), stims = {stims}; end
if ~iscell(preds), preds = {preds}; end

% make sure stims and preds are the same length (same number of trials)
if length(stims)~=length(preds)
    error('Stims and preds must have the same number of trials')
end

% check each trial and make sure the stim and preds are the same length
% (same number of time samples)
for n = 1:length(stims)
    if length(stims{n})~=length(preds{n})
        error('All stims and preds arrays must have the same number of time samples');
    end
end

wnd_idx = wnd/1000*Fs; % compute the window size in indexes

timeLL = cell(length(stims),1);
t_center = cell(length(stims),1);
for n = 1:length(stims)
    % compute the starting indexes for each window when computing the
    % time-varying LL
    t = (0:length(stims{n})-1)/Fs; % time array for all indexes
    start_idx = round((0:(step/1000):t(end)-wnd/1000)*Fs)+1; % indexes where each window should start
    % preallocate the LL array
    timeLL{n} = NaN(length(start_idx),1);
    % compute the LL for each window
    for ii = 1:length(start_idx)
        idx_seg = start_idx(ii)+(0:wnd_idx-1);
        pbinom = preds{n}(idx_seg).^(stims{n}(idx_seg)).*(1-preds{n}(idx_seg)).^(1-stims{n}(idx_seg));
        timeLL{n}(ii) = sum(log(pbinom));
    end
    % save the center times for each window
    t_center{n} = (start_idx-1)/Fs+(wnd/1000)/2;
end