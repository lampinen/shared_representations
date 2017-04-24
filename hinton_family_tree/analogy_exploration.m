run = 96;

actual_l1_reps = [load(sprintf('results/sequential_noshared/SN_hinton_nhidden_12_rseed_%i_f1_pre_middle_reps.csv',run-1)); load(sprintf('results/sequential_noshared/SN_hinton_nhidden_12_rseed_%i_f2_pre_middle_reps.csv',run-1))];
l1_reps = max(actual_l1_reps,0);

figure
Q1 = squareform(pdist(l1_reps,'cosine'));
imagesc(Q1)

%% 3layer
run = 1;

pre_l1_reps = load(sprintf('results/regular_3layer/hinton_nhidden_200_dropout_1.000000_rseed_%i_pre_middle_reps.csv',run-1)); 
l1_reps = max(pre_l1_reps,0);

figure
Q1 = squareform(pdist(l1_reps));
imagesc(Q1)

pre_l2_reps = load(sprintf('results/regular_3layer/hinton_nhidden_200_dropout_1.000000_rseed_%i_l2_reps.csv',run-1)); 
l2_reps = max(pre_l2_reps,0);

figure
Q2 = squareform(pdist(l2_reps));
imagesc(Q2)


%% 4layer
run = 1;

actual_l1_reps = [load(sprintf('results/simul_learning_4layer/hinton_nhidden_12_rseed_%i_f1_pre_middle_reps.csv',run-1)); load(sprintf('results/simul_learning_4layer/hinton_nhidden_12_rseed_%i_f2_pre_middle_reps.csv',run-1))];
l1_reps = max(actual_l1_reps,0);
actual_l2_reps = [load(sprintf('results/simul_learning_4layer/hinton_nhidden_12_rseed_%i_f1_l2_reps.csv',run-1)); load(sprintf('results/simul_learning_4layer/hinton_nhidden_12_rseed_%i_f2_l2_reps.csv',run-1))];
l2_reps = max(actual_l2_reps,0);
actual_l3_reps = [load(sprintf('results/simul_learning_4layer/hinton_nhidden_12_rseed_%i_f1_l3_reps.csv',run-1)); load(sprintf('results/simul_learning_4layer/hinton_nhidden_12_rseed_%i_f2_l3_reps.csv',run-1))];
l3_reps = max(actual_l3_reps,0);

figure
Q1 = squareform(pdist(l1_reps));
imagesc(Q1)
figure
Q2 = squareform(pdist(l2_reps));
imagesc(Q2)
figure
Q3 = squareform(pdist(l3_reps));
imagesc(Q3)

figure
imagesc(Q1+Q2+Q3)


%% SVD stuff
input = load('hinton_x_data.csv');

lz_IO = input.'*actual_l1_reps;

lz_IO_c = lz_IO-ones(48,1)*mean(lz_IO,1);

[U_lz,S_lz,V_lz] = svd(lz_IO_c.');

V = V_lz(:,1:12);
permuted_V = V([12,4,8,2,11,10,9,3,7,6,5,1,14,13,16,15,18,17,20,19,22,21,24,23,25:48],:);
Q = squareform(pdist(V,'cosine'));
imagesc(Q)