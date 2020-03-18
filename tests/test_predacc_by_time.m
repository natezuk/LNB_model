% Test predacc_by_time by simulating a time-varying Bernoulli process
% Nate Zuk (2020)

addpath('..');

dur = 10; % duration of the trial
Fs = 128; % sampling rate (in Hz)
wnd = 1000; % window size for computing time-varying log-likelihood (in ms)
event_prob_step = [0.6 0.1]; % probability of event before and after step

% Generate a Bernoulli process with a step change in event probability
% 1/3 time at prob 1, 1/3 time at prob 2, 1/3 time back to prob 1
prob_step_idx = round((0:2)*dur*Fs/3)+1; % indexes where the changes should occur
event_prob = [event_prob_step(1)*ones(prob_step_idx(2)-prob_step_idx(1),1);...
    event_prob_step(2)*ones(prob_step_idx(3)-prob_step_idx(2),1);...
    event_prob_step(1)*ones(round(dur*Fs)-prob_step_idx(3)+1,1)];
stim = binornd(1,event_prob); % generate the sequence of events, based on those probabilities

% Compute a prediction using the original probabilities, and a prediction
% using the average probability over the entire trial
[timeLL,t_center] = predacc_by_time(stim,event_prob,Fs,wnd);
avg_prob = ones(length(stim),1)*sum(stim)/length(stim); % compute the average probability of an event
[avgLL,~] = predacc_by_time(stim,avg_prob,Fs,wnd);

% Plot
figure
% plot the original probabilities and events
subplot(2,1,1);
hold on
t = (0:dur*Fs-1)/Fs;
plot(t(stim==1),zeros(sum(stim),1),'k.','MarkerSize',12);
plot(t,event_prob,'m','LineWidth',2);
plot(t,avg_prob,'b','LineWidth',2);
set(gca,'FontSize',14);
xlabel('Time (s)');
ylabel('Event probability');
legend('Event times','Time-varying probability','Average probability');

% plot the log-likelihoods
subplot(2,1,2);
hold on
plot(t_center{1},avgLL{1},'b','LineWidth',2);
plot(t_center{1},timeLL{1},'r','LineWidth',2);
set(gca,'FontSize',14,'XLim',[0 dur]);
xlabel('Time (s)');
ylabel('Log-likelihood');
legend('Using average prob','Using time-varying prob');