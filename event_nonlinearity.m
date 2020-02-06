function [KL,Pevtcndr,Prcndevt,Presp,edges] = event_nonlinearity(resp,events,varargin)
% Compute the nonlinearity between the linear projection of the response
% ('resp') and the timing of events (1 for events, 0 elsewhere). To
% compute the response: resp = X*bte, where X is the EEG lag matrix and bte
% is the associated beat-triggered EEG.
% Inputs:
% - resp = array of linear responses
% - events = array with a 1 for each event, 0 otherwise
% Outputs:
% - KL = KL divergence between P(resp|event) and P(resp)
% - Pevtcndr = P(event|resp), aka the nonlinearity
% - Prcndevt = P(resp|event)
% - Presp = P(resp)
% - edges = the edges of the histogram used to compute the probability
% distributions
% Nate Zuk (2018)

nbins = 20;

if ~isempty(varargin),
    for n = 2:2:length(varargin),
        eval([varargin{n-1} '=varargin{n};']);
    end
end

[H,edges] = histcounts(resp,nbins); % compute the linear response counts and create the histogram
% Compute the number of beats for each range of linear responses
Hbr = histcounts(resp(events==1),edges);
Presp = H'/sum(H); % P(resp)
Prcndevt = Hbr'/sum(Hbr); %P(resp|bt)
Pevt = sum(events)/length(events); %P(bt)
Pevtcndr = Prcndevt./Presp*Pevt; %P(bt|resp) = P(bt)P(resp|bt)/P(resp)

% Compute the KL divergence between P(resp|beat) and P(resp)
lgrt = log2(Prcndevt./Presp); % (in the logarithm term for KL)
lgrt(isinf(lgrt)|isnan(lgrt)) = 0; % set to zero if prjbt==0
KL = sum(Prcndevt.*lgrt); % KL divergence