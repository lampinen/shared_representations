
input = load('hinton_x_data.csv');
output = load('hinton_y_data.csv');

nl_IO = input.'*output;

nl_IO_c = nl_IO-ones(48,1)*mean(nl_IO,1);

[U_nl,S_nl,V_nl] = svd(nl_IO_c.');

% offset = nl_IO(1:24,1:12);
% offset = offset-max(max(offset));
%
%
% linearized_IO = nl_IO;
% linearized_IO(1:24,13:end) = offset;
% linearized_IO(25:end,1:12) = offset;
% 
% lz_IO_c = linearized_IO-ones(48,1)*mean(linearized_IO,1);
% 
% [U_lz,S_lz,V_lz] = svd(lz_IO_c.');

actual_pre_middle_reps = load('results/hinton_nonlinear_nhidden_12_rseed_0_pre_middle_reps.csv');
actual_middle_reps = max(actual_pre_middle_reps,0);
actual_pre_outputs = load('results/hinton_nonlinear_nhidden_12_rseed_0_pre_outputs.csv');

apmr_max_mag = max(max(abs(actual_pre_middle_reps)));
apo_max_mag = max(max(abs(actual_pre_outputs)));

imagesc(actual_pre_outputs,[-apo_max_mag,apo_max_mag])
colormap(redbluecmap)

imagesc(actual_pre_middle_reps,[-apmr_max_mag,apmr_max_mag])
colormap(redbluecmap)

%% Layer 1
lz_IO = input.'*actual_pre_middle_reps;

lz_IO_c = lz_IO-ones(48,1)*mean(lz_IO,1);

[U_lz,S_lz,V_lz] = svd(lz_IO_c.');

figure
V_lz_max_mag = max(max(abs(V_lz(1:end,1:10))));
imagesc(V_lz(1:end,1:10),[-V_lz_max_mag,V_lz_max_mag])
colormap(redbluecmap)

%% Layer 2
lz_IO_2 = actual_middle_reps.'*actual_pre_outputs;

lz_IO_2_c = lz_IO_2-ones(12,1)*mean(lz_IO_2,1);

[U_lz2,S_lz2,V_lz2] = svd(lz_IO_2_c.');

diag(S_lz2)

figure 

V_lz2_max_mag = max(max(abs(V_lz2(1:end,1:10))));
imagesc(V_lz2(1:end,1:10),[-V_lz2_max_mag,V_lz2_max_mag])
colormap(redbluecmap)
