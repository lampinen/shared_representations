run = 96;

actual_l1_reps = [load(sprintf('results/sequential_noshared/SN_hinton_nhidden_12_rseed_%i_f1_pre_middle_reps.csv',run-1)); load(sprintf('results/sequential_noshared/SN_hinton_nhidden_12_rseed_%i_f2_pre_middle_reps.csv',run-1))];
l1_reps = max(actual_l1_reps,0);

figure
Q1 = squareform(pdist(l1_reps,'cosine'));
imagesc(Q1)

%% 3layer
nhidden = 1000;
eta = 0.000949;
weightsize = 2.0;
run = 3;

single_l1_reps = [load(sprintf('results/simul_learning_3layer_single_inputs/hinton_batch_nhidden_%i_eta_%f_momentum_0.000000_weightsize_%f_rseed_%i_f1_single_input_pre_middle_reps.csv',nhidden,eta,weightsize,run-1)); load(sprintf('results/simul_learning_3layer_single_inputs/hinton_batch_nhidden_%i_eta_%f_momentum_0.000000_weightsize_%f_rseed_%i_f2_single_input_pre_middle_reps.csv',nhidden,eta,weightsize,run-1))];
l1_reps = max(single_l1_reps,0);

figure
Q1 = squareform(pdist(l1_reps));
imagesc(Q1)

perm_l1_reps = l1_reps([12,4,8,2,11,10,9,3,7,6,5,1,14,13,16,15,18,17,20,19,22,21,24,23,25:48],:);
figure
Q1 = squareform(pdist(perm_l1_reps));
imagesc(Q1)


actual_l1_reps = [load(sprintf('results/simul_learning_3layer_single_inputs/hinton_batch_nhidden_%i_eta_%f_momentum_0.000000_weightsize_%f_rseed_%i_f1_pre_middle_reps.csv',nhidden,eta,weightsize,run-1)); load(sprintf('results/simul_learning_3layer_single_inputs/hinton_batch_nhidden_%i_eta_%f_momentum_0.000000_weightsize_%f_rseed_%i_f2_pre_middle_reps.csv',nhidden,eta,weightsize,run-1))];
l1_reps = max(actual_l1_reps,0);

figure
Q1 = squareform(pdist(l1_reps));
imagesc(Q1)

%% SVD stuff
input = load('hinton_x_data.csv');

lz_IO = input.'*actual_l1_reps;

lz_IO_c = lz_IO-ones(48,1)*mean(lz_IO,1);

[U_lz,S_lz,V_lz] = svd(lz_IO_c.');

V = V_lz(:,1:12);
permuted_V = V([12,4,8,2,11,10,9,3,7,6,5,1,14,13,16,15,18,17,20,19,22,21,24,23,25:48],:);
Q = squareform(pdist(V,'cosine'));
imagesc(Q)