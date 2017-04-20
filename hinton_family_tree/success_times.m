%% both together

success_epochs = [];

for rseed = 0:99
    x = load(sprintf('results/simul_learning_3layer/hinton_nhidden_12_rseed_%i_rep_tracks.csv',rseed));
    y = find(x < 0.05);
    success_epochs = [success_epochs y(1)];
end

mean(success_epochs)

%% one alone
alone_success_epochs = [];

for rseed = 0:99
    x = load(sprintf('results/pfl/hinton_nhidden_12_rseed_%i_rep_tracks.csv',rseed));
    y = find(x < 0.05);
    alone_success_epochs = [alone_success_epochs y(1)];
end

mean(alone_success_epochs)

%% second in PFL
second_success_epochs = [];

for rseed = 0:99
    x = load(sprintf('results/pfl/hinton_nhidden_12_rseed_%i_rep_tracks.csv',rseed));
    x = x(1001:end);
    y = find(x < 0.05);
    second_success_epochs = [second_success_epochs y(1)];
end

mean(second_success_epochs)

%% second in PFL with alt task
second_alt_success_epochs = [];

for rseed = 0:99
    x = load(sprintf('results/pfl/hinton_alt_nhidden_12_rseed_%i_rep_tracks.csv',rseed));
    x = x(1001:end);
    y = find(x < 0.05);
    if isempty(y)
        y = 1000;
    end
    second_alt_success_epochs = [second_alt_success_epochs y(1)];
end

mean(second_alt_success_epochs)

%% after running counting successes ON THE APPROPRIATE SIMUL_3LAYER FILES, this checks correlation between second success epochs and shared structure extraction 

[b,bint,r,rint,stats]  = regress(success_epochs.',[ones(100,1) either_sig_comp_counts]);

