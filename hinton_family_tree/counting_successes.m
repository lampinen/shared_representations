
input = load('hinton_x_data.csv');
output = load('hinton_y_data.csv');

nl_IO = input.'*output;

nl_IO_c = nl_IO-ones(48,1)*mean(nl_IO,1);

[U_nl,S_nl,V_nl] = svd(nl_IO_c.');

V_lz_projections_by_run = zeros(100,12);
V_lz_projection_tests_by_run = zeros(100,12);

U_lz2_projections_by_run = zeros(100,12);
U_lz2_projection_tests_by_run = zeros(100,12);

V_lz_gender_flipped_projection_tests_by_run = zeros(100,12);

U_lz2_gender_flipped_projection_tests_by_run = zeros(100,12);

rng(0) %Reproducibility

for run = 1:100

    %Load
    actual_pre_middle_reps = load(sprintf('results/hinton_nonlinear_nhidden_12_rseed_%i_pre_middle_reps.csv',run-1));
    actual_middle_reps = max(actual_pre_middle_reps,0);
    actual_pre_outputs = load(sprintf('results/hinton_nonlinear_nhidden_12_rseed_%i_pre_outputs.csv',run-1));


    lz_IO = input.'*actual_pre_middle_reps;

    lz_IO_c = lz_IO-ones(48,1)*mean(lz_IO,1);

    [U_lz,S_lz,V_lz] = svd(lz_IO_c.');



    lz_IO_2 = actual_middle_reps.'*actual_pre_outputs;

    lz_IO_2_c = lz_IO_2-ones(12,1)*mean(lz_IO_2,1);

    [U_lz2,S_lz2,V_lz2] = svd(lz_IO_2_c.');

    %% non-parametric test of whether cross domain projections are significantly higher than shuffled projections
    %display('V_lz')
    V_lz_top = V_lz(1:24,1:12);
    V_lz_bottom = V_lz(25:end,1:12);
    V_lz_projections = sum(V_lz_top.*V_lz_bottom,1);

    permuted_projections = zeros(1000,12);
    %display('shuffled_V_lz')
    for i = 1:1000

        shuffled_V_lz= V_lz(randperm(48),1:12);
        shuffled_V_lz_top = shuffled_V_lz(1:24,1:12);
        shuffled_V_lz_bottom = shuffled_V_lz(25:end,1:12);
        permuted_projections(i,:) = sum(shuffled_V_lz_top.*shuffled_V_lz_bottom,1);
    end

    significance_cutoffs = prctile(abs(permuted_projections),95,1);

    %display(V_lz_projections);
    %display(significance_cutoffs);
    V_lz_projections_by_run(run,:) = V_lz_projections;
    V_lz_projection_tests_by_run(run,:) = abs(V_lz_projections) > significance_cutoffs;

%% ditto for output modes
    %display('U_lz2')
    U_lz2_top = U_lz2(1:12,1:12);
    U_lz2_bottom = U_lz2(13:end,1:12);
    U_lz2_projections = sum(U_lz2_top.*U_lz2_bottom,1);

    permuted_projections = zeros(1000,12);
    %display('shuffled_U_lz2')
    for i = 1:1000

        shuffled_U_lz2= U_lz2(randperm(24),1:12);
        shuffled_U_lz2_top = shuffled_U_lz2(1:12,1:12);
        shuffled_U_lz2_bottom = shuffled_U_lz2(13:end,1:12);
        permuted_projections(i,:) = sum(shuffled_U_lz2_top.*shuffled_U_lz2_bottom,1);
    end

    significance_cutoffs = prctile(abs(permuted_projections),95,1);

    %display(U_lz2_projections);
    %display(significance_cutoffs);


    U_lz2_projections_by_run(run,:) = U_lz2_projections;
    U_lz2_projection_tests_by_run(run,:) = abs(U_lz2_projections) > significance_cutoffs;
    %% now input modes again, but gender flipped projection
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
    
    V_lz_gender_flipped_projection_tests_by_run(run,:) = abs(V_lz_projections) > significance_cutoffs;
%% ditto for output modes
    %display('U_lz2')
    U_lz2_top = U_lz2(1:12,1:12);
    U_lz2_bottom = U_lz2(13:end,1:12);
    
    %rearrange for gender flip
    U_lz2_bottom = U_lz2_bottom([12,4,8,2,11,10,9,3,7,6,5,1],:);
    
    U_lz2_projections = sum(U_lz2_top.*U_lz2_bottom,1);

    permuted_projections = zeros(1000,12);
    %display('shuffled_U_lz2')
    for i = 1:1000

        shuffled_U_lz2= U_lz2(randperm(24),1:12);
        shuffled_U_lz2_top = shuffled_U_lz2(1:12,1:12);
        shuffled_U_lz2_bottom = shuffled_U_lz2(13:end,1:12);
        permuted_projections(i,:) = sum(shuffled_U_lz2_top.*shuffled_U_lz2_bottom,1);
    end

    significance_cutoffs = prctile(abs(permuted_projections),95,1);

    %display(U_lz2_projections);
    %display(significance_cutoffs);


    U_lz2_gender_flipped_projection_tests_by_run(run,:) = abs(U_lz2_projections) > significance_cutoffs;
    
end

%How many times did each mode come out significant?
sum(V_lz_projection_tests_by_run,1)
sum(U_lz2_projection_tests_by_run,1)

%How many times did n or fewer modes come out significant?
sum(sum(V_lz_projection_tests_by_run,2) >= 3)
sum(sum(U_lz2_projection_tests_by_run,2) >= 3)

%Above, but for gender flipped
sum(V_lz_gender_flipped_projection_tests_by_run,1)
sum(U_lz2_gender_flipped_projection_tests_by_run,1)

sum(sum(V_lz_gender_flipped_projection_tests_by_run,2) >= 3)
sum(sum(U_lz2_gender_flipped_projection_tests_by_run,2) >= 3)

%and both?
sum(sum(V_lz_projection_tests_by_run + V_lz_gender_flipped_projection_tests_by_run - V_lz_projection_tests_by_run.*V_lz_gender_flipped_projection_tests_by_run ,2) >= 5)
sum(sum(V_lz_projection_tests_by_run + V_lz_gender_flipped_projection_tests_by_run - V_lz_projection_tests_by_run.*V_lz_gender_flipped_projection_tests_by_run ,2) >= 3)

sum(sum(U_lz2_projection_tests_by_run + U_lz2_gender_flipped_projection_tests_by_run - U_lz2_projection_tests_by_run.*U_lz2_gender_flipped_projection_tests_by_run ,2) >= 5)
sum(sum(U_lz2_projection_tests_by_run + U_lz2_gender_flipped_projection_tests_by_run - U_lz2_projection_tests_by_run.*U_lz2_gender_flipped_projection_tests_by_run ,2) >= 3)




sum((sum(V_lz_projection_tests_by_run,2) < 3) & (sum(V_lz_gender_flipped_projection_tests_by_run,2) < 3))
sum((sum(U_lz2_projection_tests_by_run,2) < 3) & (sum(U_lz2_gender_flipped_projection_tests_by_run,2) < 3))
sum((sum(U_lz2_projection_tests_by_run,2) < 2) & (sum(U_lz2_gender_flipped_projection_tests_by_run,2) < 2))


%% Let's visualize this

sig_comp_counts = sum(V_lz_projection_tests_by_run,2);
gf_sig_comp_counts = sum(V_lz_gender_flipped_projection_tests_by_run,2);
either_sig_comp_counts = sum(V_lz_projection_tests_by_run + V_lz_gender_flipped_projection_tests_by_run - V_lz_projection_tests_by_run.*V_lz_gender_flipped_projection_tests_by_run ,2);

N = histc(sig_comp_counts,(0:12)-0.5).'
gf_N = histc(gf_sig_comp_counts,(0:12)-0.5).'
either_N = histc(either_sig_comp_counts,(0:12)-0.5).'


bar(0:12,[N; gf_N; either_N].','stacked');
set(gca,'fontsize',13)
legend('Regular','Flipped','Either');
ylim([0 80]);
l = xlabel({'Number of layer 1 input modes showing'; 'significant shared structure representation'},'fontsize',15);
ylabel('Frequency out of 100 runs','fontsize',15)
colormap('prism')
%fuck matlab
set(l,'units','normalized');
axpos = get(gca,'pos');
extent = get(l,'extent');
set(gca,'pos',[axpos(1) axpos(2)-0.2*extent(2) axpos(3) axpos(4)])

csvwrite('sl_input_mode_significance.csv',[sig_comp_counts;  gf_sig_comp_counts; either_sig_comp_counts].')

%% now output

sig_comp_counts = sum(U_lz2_projection_tests_by_run,2);
gf_sig_comp_counts = sum(U_lz2_gender_flipped_projection_tests_by_run,2);
either_sig_comp_counts = sum(U_lz2_projection_tests_by_run + U_lz2_gender_flipped_projection_tests_by_run - U_lz2_projection_tests_by_run.*U_lz2_gender_flipped_projection_tests_by_run ,2);

N = histc(sig_comp_counts,(0:12)-0.5).'
gf_N = histc(gf_sig_comp_counts,(0:12)-0.5).'
either_N = histc(either_sig_comp_counts,(0:12)-0.5).'


bar(0:12,[N; gf_N; either_N].','stacked');
set(gca,'fontsize',13)
legend('Regular','Flipped','Either');
ylim([0 80]);
xlabel({'Number of layer 2 output modes showing'; 'significant shared structure representation'},'fontsize',15);
ylabel('Frequency out of 1000 runs','fontsize',15)
colormap('prism')

csvwrite('sl_output_mode_significance.csv',[sig_comp_counts;  gf_sig_comp_counts; either_sig_comp_counts].')

