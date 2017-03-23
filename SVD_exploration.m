 
nonlinear_IO = [[1 1 0 0 0 0]; [1 0 1 0 0 0]; [0 0 0 1 1 0]; [0 0 0 1 0 1]];

linearized_IO = [[1 1 0 -1 0 -2]; [1 0 1 0 -1 0]; [-1 0 -2 1 1 0]; [0 -1 0 1 0 1]];

ideal_linearized_IO = [[1 1 0 0 0 -1]; [1 0 1 0 -1 0]; [0 0 -1 1 1 0]; [0 -1 0 1 0 1]];

nl_IO_c = nonlinear_IO-ones(4,1)*mean(nonlinear_IO,1);
lz_IO_c = linearized_IO-ones(4,1)*mean(linearized_IO,1);
ilz_IO_c = ideal_linearized_IO-ones(4,1)*mean(ideal_linearized_IO,1);

[U_nl,S_nl,V_nl] = svd(nl_IO_c.')
[U_lz,S_lz,V_lz] = svd(lz_IO_c.')
[U_ilz,S_ilz,V_ilz] = svd(ilz_IO_c.')




actual_preoutputs = load('nonlinear_nhidden_4_rseed_0_pre_outputs.csv')
ap_c = actual_preoutputs-ones(4,1)*mean(actual_preoutputs,1);
[U_ap,S_ap,V_ap] = svd(ap_c.')

middle_stage = [[1 0.5 0.5 0 0 0]; [1 0.5 0.5 0 0 0]; [0 0 0 1 0.5 0.5]; [0 0 0 1 0.5 0.5]];
ms_c = middle_stage-ones(4,1)*mean(middle_stage,1);
[U_ms,S_ms,V_ms] = svd(ms_c.')
 
[U_blah,S_blah,V_blah] = svd((nonlinear_IO-middle_stage).')

W1 = load('nonlinear_nhidden_4_rseed_0_epoch_250_W1.csv')
W2 = load('nonlinear_nhidden_4_rseed_0_epoch_250_W2.csv')

dW1dt = W2.'*(nonlinear_IO-W1.'*W2.').'

W1_l = load('linear_nhidden_4_rseed_0_epoch_150_W1.csv')
W2_l = load('linear_nhidden_4_rseed_0_epoch_150_W2.csv')

dW1_ldt = W2_l.'*(nonlinear_IO-W1_l.'*W2_l.').'

%SVD Component plotting
rseed = 2
epoch_list = 0:10:490;
nl_SVD_components_nl = zeros(6,4,0);
ilz_SVD_components_nl = zeros(6,4,0);
nl_SVD_components_l = zeros(6,4,0);
ilz_SVD_components_l = zeros(6,4,0);
for epoch = epoch_list
    W1 = load(sprintf('nonlinear_nhidden_4_rseed_%i_epoch_%i_W1.csv',rseed,epoch));
    W2 = load(sprintf('nonlinear_nhidden_4_rseed_%i_epoch_%i_W2.csv',rseed,epoch));
    nl_SVD_components_nl = cat(3,nl_SVD_components_nl,W2*W1*V_nl);
    ilz_SVD_components_nl = cat(3,ilz_SVD_components_nl,W2*W1*V_ilz);

    
    W1_l = load(sprintf('linear_nhidden_4_rseed_%i_epoch_%i_W1.csv',rseed,epoch));
    W2_l = load(sprintf('linear_nhidden_4_rseed_%i_epoch_%i_W2.csv',rseed,epoch));
    nl_SVD_components_l = cat(3,nl_SVD_components_l,W2_l*W1_l*V_nl);
    ilz_SVD_components_l = cat(3,ilz_SVD_components_l,W2_l*W1_l*V_ilz);

end

nl_SVD_projections_nl = [];
ilz_SVD_projections_nl = [];
nl_SVD_projections_l = [];
ilz_SVD_projections_l = [];


for i = 1:50
   nl_SVD_projections_nl = [nl_SVD_projections_nl; sum(nl_SVD_components_nl(:,1:3,i).*U_nl(:,1:3),1)]; 
   ilz_SVD_projections_nl = [ilz_SVD_projections_nl; sum(ilz_SVD_components_nl(:,1:3,i).*U_ilz(:,1:3),1)]; 

   nl_SVD_projections_l = [nl_SVD_projections_l; sum(nl_SVD_components_l(:,1:3,i).*U_nl(:,1:3),1)]; 
   ilz_SVD_projections_l = [ilz_SVD_projections_l; sum(ilz_SVD_components_l(:,1:3,i).*U_ilz(:,1:3),1)]; 
end

nl_vals = diag(S_nl);
nl_vals = nl_vals(1:3);
nl_vals = repmat(nl_vals.',50,1);

ilz_vals = diag(S_ilz);
ilz_vals = ilz_vals(1:3);
ilz_vals = repmat(ilz_vals.',50,1);

nl_SVD_projections_nl = nl_SVD_projections_nl./nl_vals;
nl_SVD_projections_l = nl_SVD_projections_l./nl_vals;


ilz_SVD_projections_nl = ilz_SVD_projections_nl./ilz_vals;
ilz_SVD_projections_l = ilz_SVD_projections_l./ilz_vals;

%regular SVD
plot(epoch_list,[nl_SVD_projections_nl nl_SVD_projections_l],'Linewidth',2);
legend('nonlinear net 1st comp.','nonlinear net 2nd comp.','nonlinear net 3rd comp.','linear net 1st comp.','linear net 2nd comp.','linear net 3rd comp.','Location','southeast');
title('Learning of components of the regular SVD')
xlabel('Epoch')
ylabel('Inner product (scaled) between current and target output mode')
%linearized SVD
plot(epoch_list,[ilz_SVD_projections_nl(:,1:2) ilz_SVD_projections_l(:,1:2)],'Linewidth',2);
legend('nonlinear net 1st comp.','nonlinear net 2nd comp.','linear net 1st comp.','linear net 2nd comp.','Location','southeast');
title('Learning of components of the linearized SVD')
xlabel('Epoch')
ylabel('Inner product (scaled) between current and target output mode')


%difference
plot(epoch_list,[ilz_SVD_projections_nl(:,2)-sum(nl_SVD_projections_nl(:,2:3),2)/2 ilz_SVD_projections_l(:,2)-sum(nl_SVD_projections_l(:,2:3),2)/2]);
legend('nonlinear net','linear net');


%Better way of thinking about it: how much do separate components from
%regular SVD project onto each other
cross_projections_nl = [];
cross_projections_l = [];

for i = 1:50
    cross_projections_nl = [cross_projections_nl; sum(nl_SVD_components_nl(:,2:3,i).*cat(2,U_nl(:,3),U_nl(:,2)),1)]; 
    cross_projections_l = [cross_projections_l; sum(nl_SVD_components_l(:,2:3,i).*cat(2,U_nl(:,3),U_nl(:,2)),1)]; 
end

plot(epoch_list,[sum(abs(cross_projections_nl),2)/2 sum(abs(cross_projections_l),2)/2],'Linewidth',2);
legend('nonlinear net','linear net','Location','east'); 
title('Average cross-projection of 2nd and 3rd components of regular SVD')
xlabel('Epoch')
ylabel('Inner product (scaled) between current and target cross output mode')

%% Generating Images

imagesc(nonlinear_IO.',[-1,1])
xlabel('inputs','fontsize',30)
ylabel('outputs','fontsize',30)
set(gca,'xtick',1:4)
colormap(redbluecmap)
pos = get(gca, 'Position');
pos(2) = pos(2)+0.02;
pos(4) = pos(4)+0.02;
set(gca, 'Position', pos)

temp = V_nl(:,1:3);
temp(:,3) = -temp(:,3); %Flip mode input and outputs for visual consistency
imagesc(temp.',[-1,1])
xlabel('inputs','fontsize',30)
ylabel('modes','fontsize',30)
set(gca,'xtick',1:4)
set(gca,'ytick',1:3)
colormap(redbluecmap)
hold on;
rectangle('Position',[0.53,0.53,3.96,0.98],...
         'LineWidth',5)
rectangle('Position',[0.53,1.5,3.96,1.0],...
         'LineWidth',5)   
rectangle('Position',[0.53,2.52,3.96,0.97],...
         'LineWidth',5)  
hold off;
pos = get(gca, 'Position');
pos(2) = pos(2)+0.02;
pos(4) = pos(4)+0.02;
set(gca, 'Position', pos)

imagesc(S_nl(1:3,1:3),[-1.7321,1.7321])
xlabel('modes','fontsize',30)
set(gca,'xtick',1:3)
set(gca,'ytick',1:3)
colormap(redbluecmap)
hold on;
rectangle('Position',[0.52,0.52,0.98,0.98],...
         'LineWidth',5)
rectangle('Position',[1.5,1.5,0.98,1.0],...
         'LineWidth',5)   
rectangle('Position',[2.5,2.52,0.98,0.98],...
         'LineWidth',5)  
hold off;
pos = get(gca, 'Position');
pos(2) = pos(2)+0.02;
pos(4) = pos(4)+0.02;
set(gca, 'Position', pos)


temp = U_nl(:,1:3);
temp(:,3) = -temp(:,3); %Flip mode input and outputs for visual consistency
imagesc(temp,[-1,1])
xlabel('modes','fontsize',30)
ylabel('outputs','fontsize',30)
set(gca,'ytick',1:6)
set(gca,'xtick',1:3)
colormap(redbluecmap)
hold on;
rectangle('Position',[0.52,0.52,0.98,6.0],...
         'LineWidth',5)
rectangle('Position',[1.5,0.52,1.0,6.0],...
         'LineWidth',5)   
rectangle('Position',[2.5,0.52,0.99,6.0],...
         'LineWidth',5)  
hold off;
pos = get(gca, 'Position');
pos(2) = pos(2)+0.02;
pos(4) = pos(4)+0.02;
set(gca, 'Position', pos)

imagesc(ideal_linearized_IO.',[-1,1])
xlabel('inputs','fontsize',30)
ylabel('pre-outputs','fontsize',30)
set(gca,'xtick',1:4)
colormap(redbluecmap)
pos = get(gca, 'Position');
pos(2) = pos(2)+0.02;
pos(4) = pos(4)+0.02;
set(gca, 'Position', pos)


imagesc(round(V_ilz(:,1:2).'*10)/10,[-1,1]) %round is to handle small floating point errors that are polluting plot
xlabel('inputs','fontsize',30)
ylabel('modes','fontsize',30)
set(gca,'xtick',1:4)
set(gca,'ytick',1:3)
colormap(redbluecmap)
hold on;
rectangle('Position',[0.53,0.52,3.96,0.98],...
         'LineWidth',5)
rectangle('Position',[0.53,1.5,3.96,0.99],...
         'LineWidth',5)   
hold off;
pos = get(gca, 'Position');
pos(2) = pos(2)+0.02;
pos(4) = pos(4)+0.02;
set(gca, 'Position', pos)

imagesc(S_ilz(1:2,1:2),[-2.4495,2.4495])
xlabel('modes','fontsize',30)
set(gca,'xtick',1:2)
set(gca,'ytick',1:2)
colormap(redbluecmap)
hold on;
rectangle('Position',[0.52,0.52,0.98,0.98],...
         'LineWidth',5)
rectangle('Position',[1.5,1.5,0.99,0.99],...
         'LineWidth',5)   
hold off;
pos = get(gca, 'Position');
pos(2) = pos(2)+0.02;
pos(4) = pos(4)+0.02;
set(gca, 'Position', pos)

imagesc(U_ilz(:,1:2),[-1,1])
xlabel('modes','fontsize',30)
ylabel('outputs','fontsize',30)
set(gca,'ytick',1:6)
set(gca,'xtick',1:2)
colormap(redbluecmap)
hold on;
rectangle('Position',[0.52,0.52,0.98,6.0],...
         'LineWidth',5)
rectangle('Position',[1.5,0.52,0.99,6.0],...
         'LineWidth',5)   
hold off;
pos = get(gca, 'Position');
pos(2) = pos(2)+0.02;
pos(4) = pos(4)+0.02;
set(gca, 'Position', pos)
