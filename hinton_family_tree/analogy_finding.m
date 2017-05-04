%% 

intended_analogy = 1:24;
flipped_analogy = [12,4,8,2,11,10,9,3,7,6,5,1,14,13,16,15,18,17,20,19,22,21,24,23];


nhidden = 1000;
eta = 0.000949;
weightsize = 2.0;
run = 3;
batch = 'batch_'; %'batch_' or '' 

single_l1_reps = [load(sprintf('results/simul_learning_3layer_single_inputs/hinton_%snhidden_%i_eta_%f_momentum_0.000000_weightsize_%f_rseed_%i_f1_single_input_pre_middle_reps.csv',batch,nhidden,eta,weightsize,run-1)); load(sprintf('results/simul_learning_3layer_single_inputs/hinton_%snhidden_%i_eta_%f_momentum_0.000000_weightsize_%f_rseed_%i_f2_single_input_pre_middle_reps.csv',batch,nhidden,eta,weightsize,run-1))];

dist = squareform(pdist(single_l1_reps));
dist = dist(1:24,25:end);
dist_size = size(dist);

%% "settling" based permutation finding

% display('settling')
% dist_values = max(max(dist))-dist;
% % figure
% % imagesc(dist_values)
% 
% activity = zeros(size(dist));
% settling_update_rate = 0.01;
% for iteration = 1:50000
%     colsums = sum(activity,1);
%     rowsums = sum(activity,2);
%     
%     for i = 1:24
%         for j = 1:24
%             activity(i,j) = activity(i,j) + settling_update_rate*(2*activity(i,j)+dist_values(i,j)-(rowsums(i)+colsums(j)));
%         end
%     end
%     activity = max(activity,0);
% end
% % 
% % figure
% % imagesc(activity)

%% more sophisticated?
display('settling')

centered_l1_reps = [single_l1_reps(1:24,:)-ones(24,1)*sum(single_l1_reps(1:24,:),1); single_l1_reps(25:end,:)-ones(24,1)*sum(single_l1_reps(25:end,:),1)];
rel_dist = squareform(pdist(centered_l1_reps,'cosine'));
rel_dist = rel_dist(1:24,25:end);
rel_dist_values = max(max(rel_dist))-rel_dist;

% figure
% imagesc(dist_values)

activity = zeros(size(dist));
settling_update_rate = 0.01;
for iteration = 1:50000
    colsums = sum(activity,1);
    rowsums = sum(activity,2);
    
    for i = 1:24
        for j = 1:24
            activity(i,j) = activity(i,j) + settling_update_rate*(2*activity(i,j)+rel_dist_values(i,j)-(rowsums(i)+colsums(j)));
        end
    end
    activity = max(activity,0);
end

%%

[sorted_activity,rank] = sort(activity,1);

curr_assignment = rank(end,:)

if all(curr_assignment == intended_analogy) || all(curr_assignment == flipped_analogy)
     display('Solution found!')
     display(curr_assignment)
     display('found at settling')
     return
end
%%

curr_missed = setdiff(intended_analogy,curr_assignment);
if ~isempty(curr_missed) %not surjective, not a valid assignment
    display('not a permutation, fixing...')
    [~,used_indices,~] = unique(curr_assignment);
    curr_repeated = unique(curr_assignment(setdiff(intended_analogy,used_indices)));
    to_reassign = [curr_missed curr_repeated];
    curr_repeated_indices = find(ismember(curr_assignment,curr_repeated));

    if length(to_reassign) <= 8 %If short enough, search from here to find best of these permutations by heuristic
        P = perms(to_reassign);
        this_score = inf;
        X = curr_assignment;
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
        curr_assignment = X;
    end
    display('Starting permutation found');
    display(curr_assignment);
    
    if all(curr_assignment == intended_analogy) || all(curr_assignment == flipped_analogy)
         display('Solution found!')
         display(curr_assignment)
         display('found while permutation fixing')
         return
    end
end

%% search
transpositions = nchoosek(1:24,2);
trans_as = transpositions(:,1);
trans_bs = transpositions(:,2);

visited = [zeros(1,24)];
maxsteps = 10000;
curr_step = 1;
while curr_step <= maxsteps
    visited = [visited; curr_assignment];
    best = -1;
    best_score = inf;
    new_assignments = arrayfun( @(a,b) apply_transposition(curr_assignment,a,b),trans_as,trans_bs,'UniformOutput',false);
    for  i = 1:length(new_assignments)
        this_new_assignment = new_assignments{i};
        % is solution?
        if all(this_new_assignment == intended_analogy) || all(this_new_assignment == flipped_analogy)
            curr_assignment = this_new_assignment;
             display('Solution found!')
             display(curr_assignment)
             display('Number steps:')
             display(curr_step)
            return
        end    
        % rank
        this_score = score(intended_analogy,this_new_assignment,dist,dist_size);
        if this_score < best_score && ~ismember(this_new_assignment,visited,'rows') %new best! (and not yet visited)
            best_score = this_score;
            best = this_new_assignment;
%            insertion_index = find(top_10_scores > this_score,1);
%            top_10_scores = [top_10_scores(1:insertion_index-1) this_score top_10_scores(insertion_index:end-1)];
%            top_10 = [top_10(1:insertion_index-1,:) this_new_assignment top_10(insertion_index:end-1,:)]; 
        end
    end
    
    % crazy unlikely possibility, but who knows
    if best == -1
        display('visited all neighbors, bailing...')
        curr_assignment = -1;
        return
    end
    
    
    curr_assignment = best;

    
    curr_step = curr_step + 1;
end