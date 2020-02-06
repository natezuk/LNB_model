function [opterp,nonlin,optlambda,LL,dly,stats] = create_erp_regularize(eegs,stims,Fs,mindly,maxdly,lambdas,varargin)
% Create an ERP model in the form of a linear-nonlinear bernoulli model.
% Use ridge regression to regularize the ERP model.
% Inputs:
% - eeg = cell array of EEG trials
% - stim = cell array of stimuli for each trial.  Each stimulus must
% be a time array containing 1 for each event and 0 otherwise.
% - Fs = sampling rate of the EEG and stimulus (in Hz)
% - mindly = minimum delay in the ERP model (in ms)
% - maxdly = maximum delay in the ERP model (in ms)
% - lambdas = set of lambdas to use for regularization
% Outputs:
% - opterp = vector of the optimal ERP (channels x time elements)
% - nonlin = parameters for the sigmoidal nonlinearity [intercept; scaling]
% - optlambda = optimal regularization value
% - LL = log-likelihoods for cross-validation for each lambda and fold
% - dly = vector of the (index-based) delays in the ERP
% - stats = additional statistics associated with the nonlinearity
% computation
% Nate Zuk (2018)

% Initial variables
nfold = 10; % n-fold cross validation
verbose = 1;
    
% Parse varargin
if ~isempty(varargin),
    for n = 2:2:length(varargin),
        eval([varargin{n-1} '=varargin{n};']);
    end
end

% Convert to cell if needed
if ~iscell(eegs), eegs = {eegs}; end
if ~iscell(stims), stims = {stims}; end

% Use n-fold crossvalidation, randomly splitting the data into n folds for
% training and cross-val
totidx = sum(cellfun(@(x) size(x,1), stims));
rndidx = randperm(totidx); % randomly rearrange indexes
seg_folds = [1:floor(totidx/nfold):totidx totidx+1]; % define the range of indexes for each fold

% Create the lag matrices for each trial
dly = -floor(mindly/1000*Fs):-1:-ceil(maxdly/1000*Fs);
if verbose, disp('Create lag matrices...'); end
Xc = cell(length(eegs),1);
for ii = 1:length(eegs), Xc{ii} = lagGen(eegs{ii},dly); end

if length(lambdas)>1,
    LL = NaN(length(lambdas),nfold);
    for n = 1:nfold,
        if verbose, disp(['** Fold ' num2str(n) '/' num2str(nfold) ' **']); end
        foldtm = tic;
        % Get the training and testing indexes
        tstidx = rndidx(seg_folds(n):seg_folds(n+1)-1); % testing indexes
        trnidx = setxor(tstidx,1:totidx); % training indexes
        % Compute the ERP
        [erp,~,X,STIM] = ridgeeventtrigeeg(eegs,stims,Fs,mindly,maxdly,lambdas,trnidx,'verbose',verbose);
        % Get the design matrix and stimulus array for the testing data
        fprintf('Computing the X and Y matrices for the cross-validation data...');
        mattm = tic;
        Xtst = zscore(cell_to_time_samples(Xc,tstidx));
        STIMtst = cell_to_time_samples(stims,tstidx);
        fprintf('Completed @ %.3f s\n',toc(mattm));
        if verbose, fprintf('Computing nonlinearity and cross-validating each lambda (%d iterations)',length(lambdas)); end
        for l = 1:length(lambdas),
            if verbose, fprintf('.'); end
%             % Compute the linear response
            RESP = X*erp(:,l);
            nonlin = glmfit(RESP,STIM,'binomial');
            [~,LL(l,n)] = predict_eeg_events_idx(Xtst,STIMtst,erp(:,l),nonlin,dly,[],'parse_data',0);
        end
        if verbose, fprintf('\n'); end
        fprintf('Completed fold @ %.3f s\n',toc(foldtm));
        clear X STIM RESP Xtst STIMtst
    end
    % Determine the optimal lambda
    opt_llidx = find(sum(LL,2)==max(sum(LL,2)),1,'first');
    optlambda = lambdas(opt_llidx);
    if verbose, disp(['-- Optimal lambda = ' num2str(optlambda) ' --']); end
else
    LL = [];
    optlambda = lambdas;
end

% Fitting full model
if verbose, disp('Fitting full model...'); end
[opterp,~,X,STIM] = ridgeeventtrigeeg(eegs,stims,Fs,mindly,maxdly,optlambda,[],'verbose',verbose);
RESP = X*opterp;
[KL,posterior,likelihood,prior,edges] = event_nonlinearity(RESP,STIM);
evidence = sum(STIM)/length(STIM);
nonlin = glmfit(RESP,STIM,'binomial');
clear X STIM RESP

% Save some of the statistics
stats.KL = KL; % KL-divergence
stats.posterior = posterior; % histogram for the posterior P(event|resp) (aka the nonlinearity)
stats.likelihood = likelihood; % histogram of likelihood P(resp|event)
stats.prior = prior; % histogram of the prior P(resp);
stats.evidence = evidence; % P(event)
stats.cnts = edges(1:end-1)+diff(edges)/2; % center of the bins for the histograms
