
actual_preoutputs = load('results/depth_comp_for_PNAS/linear_nlayer_2_nhidden_4_rseed_0_final_pre_outputs.csv')
ap_c = actual_preoutputs-ones(6,1)*mean(actual_preoutputs,1);
[U_ap,S_ap,V_ap] = svd(actual_preoutputs.')


actual_preoutputs = load('results/depth_comp_for_PNAS/linear_nlayer_3_nhidden_4_rseed_0_final_pre_outputs.csv')
ap_c = actual_preoutputs-ones(6,1)*mean(actual_preoutputs,1);
[U_ap,S_ap,V_ap] = svd(actual_preoutputs.')

%% 

for run = 1:1
   nlayer = 2;
   track2 = zeros(6,0);
   for epoch = 0:1:2999
       actual_preoutputs = load(sprintf('results/depth_comp_for_PNAS/original_linear_nlayer_%i_nhidden_4_rseed_%i_epoch_%i_pre_outputs.csv',nlayer,run-1,epoch));
       s = svd(actual_preoutputs.'); % the transpose changes nothing, I know
       track2 = [track2 s];
   end

   nlayer = 3;
   track3 = zeros(6,0);
   for epoch = 0:10:2990
       actual_preoutputs = load(sprintf('results/depth_comp_for_PNAS/original_linear_nlayer_%i_nhidden_4_rseed_%i_epoch_%i_pre_outputs.csv',nlayer,run-1,epoch));
       s = svd(actual_preoutputs.'); % the transpose changes nothing, I know
       track3 = [track3 s];
   end
   
   figure;
   plot(track2.')
   title('2-layer')
   figure;
   plot(track3.')
   title('3-layer')
end
