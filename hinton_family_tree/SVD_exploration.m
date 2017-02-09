
input = load('hinton_x_data.csv');
output = load('hinton_y_data.csv');

nl_IO = input.'*output;

nl_IO_c = nl_IO-ones(48,1)*mean(nl_IO,1);

[U_nl,S_nl,V_nl] = svd(nl_IO_c.');


%% Load
actual_pre_middle_reps = load('results/hinton_nonlinear_nhidden_12_rseed_0_pre_middle_reps.csv');
actual_middle_reps = max(actual_pre_middle_reps,0);
actual_pre_outputs = load('results/hinton_nonlinear_nhidden_12_rseed_0_pre_outputs.csv');

apmr_max_mag = max(max(abs(actual_pre_middle_reps)));
apo_max_mag = max(max(abs(actual_pre_outputs)));

imagesc(actual_pre_middle_reps,[-apmr_max_mag,apmr_max_mag])
colormap(redbluecmap)

imagesc(actual_pre_outputs,[-apo_max_mag,apo_max_mag])
colormap(redbluecmap)



%% Layer 1 plot
lz_IO = input.'*actual_pre_middle_reps;

lz_IO_c = lz_IO-ones(48,1)*mean(lz_IO,1);

[U_lz,S_lz,V_lz] = svd(lz_IO_c.');

figure
V_lz_max_mag = max(max(abs(V_lz(1:end,1:12))));
imagesc(V_lz(1:end,1:12).',[-V_lz_max_mag,V_lz_max_mag])
xlabel('inputs','fontsize',16)
ylabel('modes','fontsize',16)
set(gca,'ytick',1:12)
colormap(redbluecmap)

%% Layer 2 plot
lz_IO_2 = actual_middle_reps.'*actual_pre_outputs;

lz_IO_2_c = lz_IO_2-ones(12,1)*mean(lz_IO_2,1);

[U_lz2,S_lz2,V_lz2] = svd(lz_IO_2_c.');

diag(S_lz2)

figure 

V_lz2_max_mag = max(max(abs(V_lz2(1:end,1:12))));
imagesc(V_lz2(1:end,1:12),[-V_lz2_max_mag,V_lz2_max_mag])
colormap(redbluecmap)

%% non-parametric test of whether cross domain projections are significantly higher than shuffled projections
display('V_lz')
V_lz_top = V_lz(1:24,1:12);
V_lz_bottom = V_lz(25:end,1:12);
V_lz_projections = sum(V_lz_top.*V_lz_bottom,1);

permuted_projections = zeros(1000,12);
display('shuffled_V_lz')
for i = 1:1000
    
    shuffled_V_lz= V_lz(randperm(48),1:12);
    shuffled_V_lz_top = shuffled_V_lz(1:24,1:12);
    shuffled_V_lz_bottom = shuffled_V_lz(25:end,1:12);
    permuted_projections(i,:) = sum(shuffled_V_lz_top.*shuffled_V_lz_bottom,1);
end

significance_cutoffs = prctile(abs(permuted_projections),95,1);

display(V_lz_projections);
display(significance_cutoffs);

display(abs(V_lz_projections) > significance_cutoffs);

%% ditto for output modes
display('U_lz2')
U_lz2_top = U_lz2(1:12,1:12);
U_lz2_bottom = U_lz2(13:end,1:12);
U_lz2_projections = sum(U_lz2_top.*U_lz2_bottom,1);

permuted_projections = zeros(1000,12);
display('shuffled_U_lz2')
for i = 1:1000
    
    shuffled_U_lz2= U_lz2(randperm(24),1:12);
    shuffled_U_lz2_top = shuffled_U_lz2(1:12,1:12);
    shuffled_U_lz2_bottom = shuffled_U_lz2(13:end,1:12);
    permuted_projections(i,:) = sum(shuffled_U_lz2_top.*shuffled_U_lz2_bottom,1);
end

significance_cutoffs = prctile(abs(permuted_projections),95,1);

display(U_lz2_projections);
display(significance_cutoffs);

display(abs(U_lz2_projections) > significance_cutoffs);


%% now input modes again, but gender flipped
V_lz_top = V_lz(1:24,1:12);
V_lz_bottom = V_lz(25:end,1:12);

%rearrange bottom according to switching gender
V_lz_bottom = V_lz_bottom([12,4,8,2,11,10,9,3,7,6,5,1,14,13,16,15,18,17,20,19,22,21,24,23],:);

V_lz_projections = sum(V_lz_top.*V_lz_bottom,1);

permuted_projections = zeros(1000,12);

for i = 1:1000
    
    shuffled_V_lz= V_lz(randperm(48),1:12);
    shuffled_V_lz_top = shuffled_V_lz(1:24,1:12);
    shuffled_V_lz_bottom = shuffled_V_lz(25:end,1:12);
    permuted_projections(i,:) = sum(shuffled_V_lz_top.*shuffled_V_lz_bottom,1);
end

significance_cutoffs = prctile(abs(permuted_projections),95,1);

display(V_lz_projections);
display(significance_cutoffs);

display(abs(V_lz_projections) > significance_cutoffs);

%%

figure

U_lz2_max_mag = max(max(abs(U_lz2(1:end,1:12))));
imagesc(U_lz2(1:end,1:12),[-U_lz2_max_mag,U_lz2_max_mag])
colormap(redbluecmap)

%% nonparametric test of projection of input modes onto the "domain" dimension
q = ones(48,1);
q(25:48) = -1;
A = q.'*V_lz;


maxBs = zeros(1000,1);

for i = 1:1000
    B = q(randperm(48)).'*V_lz; %lazy non-parametric estimate of how unusual the above projections are
    maxBs(i) = max(abs(B));
end

highest_value = (abs(A(3)))
first_component_value = abs(A(1))
significance_threshold = prctile(maxBs,5)

%% How do these project onto modes from offset version (even though it doesn't quite solve the problem)?

offset = nl_IO(1:24,1:12);
offset = offset-max(max(offset));


ideal_linearized_IO = nl_IO;
ideal_linearized_IO(1:24,13:end) = offset;
ideal_linearized_IO(25:end,1:12) = offset;

ilz_IO_c = ideal_linearized_IO-ones(48,1)*mean(ideal_linearized_IO,1);

[U_ilz,S_ilz,V_ilz] = svd(ilz_IO_c.');

%input
figure 
projection_strengths = abs(V_ilz(1:end,1:13).'*V_lz(1:end,1:12));
max_ps = max(max(projection_strengths));
imagesc(projection_strengths,[-max_ps,max_ps]);
colormap(redbluecmap);

%output
figure 
projection_strengths = abs(U_ilz(1:end,1:13).'*U_lz2(1:end,1:12));
max_ps = max(max(projection_strengths));
imagesc(projection_strengths,[-max_ps,max_ps]);
colormap(redbluecmap);

%% Looking at second layer?
imagesc(lz_IO*V_lz2)

%% crude check of gender flip indices:

sum(input(:,[12 4 8 2 11 10 9 3 7 6 5 1 14 13 16 15 18 17 20 19 22 21 24 23 25:48])) - sum(input)

%% Input modes, but looking at only people inputs

display('V_lz')
V_lz_top = V_lz(1:12,1:12);
V_lz_bottom = V_lz(25:36,1:12);
V_lz_projections = sum(V_lz_top.*V_lz_bottom,1);

permuted_projections = zeros(1000,12);
display('shuffled_V_lz')
for i = 1:1000
    
    shuffled_V_lz= V_lz(randperm(48),1:12);
    shuffled_V_lz_top = shuffled_V_lz(1:12,1:12);
    shuffled_V_lz_bottom = shuffled_V_lz(25:36,1:12);
    permuted_projections(i,:) = sum(shuffled_V_lz_top.*shuffled_V_lz_bottom,1);
end

significance_cutoffs = prctile(abs(permuted_projections),95,1);


display('People')
display(V_lz_projections);
display(significance_cutoffs);

display(abs(V_lz_projections) > significance_cutoffs);

%% "" Relations
display('V_lz')
V_lz_top = V_lz(13:24,1:12);
V_lz_bottom = V_lz(37:end,1:12);
V_lz_projections = sum(V_lz_top.*V_lz_bottom,1);

permuted_projections = zeros(1000,12);
display('shuffled_V_lz')
for i = 1:1000
    
    shuffled_V_lz= V_lz(randperm(48),1:12);
    shuffled_V_lz_top = shuffled_V_lz(13:24,1:12);
    shuffled_V_lz_bottom = shuffled_V_lz(37:end,1:12);
    permuted_projections(i,:) = sum(shuffled_V_lz_top.*shuffled_V_lz_bottom,1);
end

significance_cutoffs = prctile(abs(permuted_projections),95,1);


display('Relations')
display(V_lz_projections);
display(significance_cutoffs);

display(abs(V_lz_projections) > significance_cutoffs);

%% relations gender flipped 


display('V_lz')
V_lz_top = V_lz(13:24,1:12);
V_lz_bottom = V_lz(25:end,1:12);
V_lz_bottom = V_lz_bottom([12,4,8,2,11,10,9,3,7,6,5,1,14,13,16,15,18,17,20,19,22,21,24,23],:);
V_lz_bottom = V_lz_bottom(13:end,:)

V_lz_projections = sum(V_lz_top.*V_lz_bottom,1);

permuted_projections = zeros(1000,12);
display('shuffled_V_lz')
for i = 1:1000
    
    shuffled_V_lz= V_lz(randperm(48),1:12);
    shuffled_V_lz_top = shuffled_V_lz(13:24,1:12);
    shuffled_V_lz_bottom = shuffled_V_lz(37:end,1:12);
    permuted_projections(i,:) = sum(shuffled_V_lz_top.*shuffled_V_lz_bottom,1);
end

significance_cutoffs = prctile(abs(permuted_projections),95,1);


display('Relations')
display(V_lz_projections);
display(significance_cutoffs);

display(abs(V_lz_projections) > significance_cutoffs);
