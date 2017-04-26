%% 

intended_analogy = 1:24;
flipped_analogy = [12,4,8,2,11,10,9,3,7,6,5,1,14,13,16,15,18,17,20,19,22,21,24,23];


nhidden = 200;
eta = 0.003536;
weightsize = 2.0;
run = 2;

single_l1_reps = [load(sprintf('results/simul_learning_3layer_single_inputs/hinton_nhidden_%i_eta_%f_momentum_0.000000_weightsize_%f_rseed_%i_f1_single_input_pre_middle_reps.csv',nhidden,eta,weightsize,run-1)); load(sprintf('results/simul_learning_3layer_single_inputs/hinton_nhidden_%i_eta_%f_momentum_0.000000_weightsize_%f_rseed_%i_f2_single_input_pre_middle_reps.csv',nhidden,eta,weightsize,run-1))];

dist = squareform(pdist(single_l1_reps));
dist = dist(1:24,25:end);
dist_size = size(dist);

[sorted_dist,rank] = sort(dist,1);

%imagesc(rank)

curr_assignment = rank(1,:)
curr_missed = setdiff(intended_analogy,curr_assignment);
if ~isempty(curr_missed) %not surjective, not a valid assignment
    [~,used_indices,~] = unique(curr_assignment);
    curr_repeated = setdiff(intended_analogy,used_indices);
    to_reassign = [curr_missed curr_repeated];
    curr_repeated_indices = find(ismember(curr_assignment,curr_repeated));

    if length(curr_repeated_indices) <= 8 %If short enough, search from here to find best of these permutations by heuristic
        P = perms(curr_repeated_indices);
        this_score = inf;
        for i = 1:length(P)
           X(curr_repeated_indices) = P(i,:);
           temp = score(intended_analogy,X,dist,dist_size);
           if temp < this_score
               curr_assignment = X;
               this_score = temp;
           end
        end
    else % just reassign to first permutation and begin main search
        X = curr_assignment;
        X(curr_repeated_indices) = to_reassign;
    end
    display('Starting permutation found');
    display(curr_assignment);
end

% search
visited = [];
maxsteps = 10000;
while step < maxsteps
    if any(curr_assignment ~= intended_analogy) || any(curr_assignment ~= flipped_analogy)
        display('Solution found!')
        display(curr_assignment)
        break
    end
    step = step + 1;
end