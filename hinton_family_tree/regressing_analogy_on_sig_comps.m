nhidden = 1000;
eta = 0.02/sqrt(nhidden);
%weightsize = 2.0;
batch = 'batch_'; %'batch_' or '' 
init_type = 'weight_unif_'; %'weight_unif_' or 'weight_He_''

found_time = csvread(sprintf('results/analogy_finding/hinton_%snhidden_%i_eta_%f_%smomentum_0.000000_found_times.csv',batch,nhidden,eta,init_type));
found_at_settling = csvread(sprintf('results/analogy_finding/hinton_%snhidden_%i_eta_%f_%smomentum_0.000000_found_at_settling.csv',batch,nhidden,eta,init_type));
either_sig_comp_counts = csvread(sprintf('results/analogy_finding/hinton_%snhidden_%i_eta_%f_%smomentum_0.000000_either_sig_comp_counts.csv',batch,nhidden,eta,init_type));
sig_comp_counts = csvread(sprintf('results/analogy_finding/hinton_%snhidden_%i_eta_%f_%smomentum_0.000000_sig_comp_counts.csv',batch,nhidden,eta,init_type));
gf_sig_comp_counts = csvread(sprintf('results/analogy_finding/hinton_%snhidden_%i_eta_%f_%smomentum_0.000000_gf_sig_comp_counts.csv',batch,nhidden,eta,init_type));

[b,dev,stats] = glmfit(either_sig_comp_counts,found_at_settling,'binomial');
display(b)
display(stats.t)
display(stats.p)
[b,dev,stats] = glmfit(either_sig_comp_counts,found_time,'normal');
display(b)
display(stats.t)
display(stats.p)


%% Saving for posterity
%csvwrite(sprintf('results/analogy_finding/hinton_%snhidden_%i_eta_%f_%smomentum_0.000000_found_times.csv',batch,nhidden,eta,init_type),found_time)
%csvwrite(sprintf('results/analogy_finding/hinton_%snhidden_%i_eta_%f_%smomentum_0.000000_found_flipped.csv',batch,nhidden,eta,init_type),found_flipped)
%csvwrite(sprintf('results/analogy_finding/hinton_%snhidden_%i_eta_%f_%smomentum_0.000000_found_at_settling.csv',batch,nhidden,eta,init_type),found_at_settling)
%csvwrite(sprintf('results/analogy_finding/hinton_%snhidden_%i_eta_%f_%smomentum_0.000000_either_sig_comp_counts.csv',batch,nhidden,eta,init_type),either_sig_comp_counts)
%csvwrite(sprintf('results/analogy_finding/hinton_%snhidden_%i_eta_%f_%smomentum_0.000000_sig_comp_counts.csv',batch,nhidden,eta,init_type),sig_comp_counts)
%csvwrite(sprintf('results/analogy_finding/hinton_%snhidden_%i_eta_%f_%smomentum_0.000000_gf_sig_comp_counts.csv',batch,nhidden,eta,init_type),gf_sig_comp_counts)


