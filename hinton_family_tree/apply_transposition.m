%%% this function applies the transposition (a,b) to the permutation x,
%%% for example apply_transposition([5,4,3,2,1],1,2) = [4,5,3,2,1]
function x = apply_transposition(x,a,b)
    x([a b]) = x([b a]);
end