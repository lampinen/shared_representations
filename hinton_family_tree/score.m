%% Helper function
function d = score(source,target,dists,dists_size)
    d = sum(dists(sub2ind(dists_size,source,target)));
end

