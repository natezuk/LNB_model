function [nullLL,nulladjLL] = null_predict_events(eegs,stims,Fs,mindly,maxdly,lambdas,varargin)
% Create a null distribution of LL by iteratively randomly rearranging and time
% shifting the stimulus vectors, computing a new model using ridge
% regression, and determining the LL on a test set of data.
% Outputs:
% - nullLL = null distribution of LL
% - nulladjLL = null distribution of LL after subtracting the LL of a
% constant model
% Nate Zuk (2018)

nboot = 10; % number of iterations to compute the null distribution

if ~isempty(varargin),
    for n = 2:2:length(varargin),
        eval([varargin{n-1} '=varargin{n};']);
    end
end

dly = -floor(mindly/1000*Fs):-1:-ceil(maxdly/1000*Fs);

% Turn the lambda matrix into a vector, if necessary
lambdas = reshape(lambdas,[numel(lambdas) 1]);
ncnd = size(stims{1},2); % number of stimulus conditions

fprintf('Creating the null distribution (%d iterations)\n',nboot);
nullLL = NaN(nboot,1);
nulladjLL = NaN(nboot,1);
for n = 1:nboot,
    %fprintf('.');
    boottm = tic;
    xc = cell(length(stims),1); yc = cell(length(stims),1);
    % Rearrange the stimulus arrays
    Yc = stims(randperm(length(stims)));
    cnd_idx = randi(ncnd,length(stims),1); % randomly select stimulus conditions
    % Circularly shift each array randomly, then concatenate across trials
    for ii = 1:length(stims),
        Yc{ii} = circshift(Yc{ii}(:,cnd_idx(ii)),randperm(size(Yc{ii},1)));
        use_length = min([size(eegs{ii},1),size(Yc{ii},1)]); % use the same length for both EEG and stimulus
        xc{ii} = eegs{ii}(1:use_length,:); yc{ii} = Yc{ii}(1:use_length);
    end
    clear Yc
    % Randomly pick a lambda value
    lmb = lambdas(randi(length(lambdas)));
    % Compute the model on the randomized data
    [nerp,nonlin] = create_erp_regularize(xc(1:end-1),yc(1:end-1),Fs,mindly,maxdly,lmb,'verbose',0);
    % Compute the prediction for the left out trial
    Xtst = zscore(lagGen(xc{end},dly));
    lin_pred = Xtst*nerp;
    clear Xtst
    % Transform to probability, but account for the event probability for
    % the testing data (NZ, 8-1-2018)
    evtprob = sum(yc{end})/length(yc{end});
    nonlin(1) = log(evtprob/(1-evtprob));
    pred = glmval(nonlin,lin_pred,'logit');
    clear nerp nonlin
    % Compute the log-likelihood
    pbinom = pred.^(yc{end}).*(1-pred).^(1-yc{end});
    nullLL(n) = sum(log(pbinom));
    % Compute the log-likelihood of a constant model
    avg_p = sum(yc{end})/length(yc{end});
    base_pbinom = avg_p.^(yc{end}).*(1-avg_p).^(1-yc{end});
    nulladjLL(n) = nullLL(n) - sum(log(base_pbinom));
    fprintf('%d) %.3f s\n',n,toc(boottm));
    clear xc yc
end