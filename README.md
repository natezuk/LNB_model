# LNB_model
Generates a linear-nonlinear Bernoulli model for predicting discrete events, useful for EEG decoding

This code can be used to generate a regularized event-related potential (ERP) model, which can be used to create a time-varying event probability based on EEG data.  The model assumes that the likelihood of events is determined by dot produced between the ERP and the EEG followed by a nonlinearity defined by a sigmoid function to make the probability Bernoulli-distributed.  The fit of the time-varying probability is quantified using the log-likelihood, which can be compared to a null distribution based on shuffled and permuted data to evaluate the goodness of fit.

* `create_erp_regularize` -- uses 10-fold cross-validation to create an regularized ERP model with ridge regularization
* `predict_eeg_events_idx` -- computes a time-varying probability of events in a trial using a given ERP model and nonlinearity

This model was used to quantify the predictability of phonemes, vowels, and consonants from EEG recorded during continuous listening to speech.  A conference paper based on this work can be found here:

Zuk NJ, Di Liberto GL, Lalor EC (2019). Linear-nonlinear Bernoulli modeling for quantifying temporal coding of phonemes in continuous speech. Conference on Cognitive Computational Neuroscience, 13-16 September, Berlin, Germany. doi: [10.32470/CCN.2019.1192-0](https://ccneuro.org/2019/Papers/ViewPapers.asp?PaperNum=1192)