 
nonlinear_IO = [[1 1 0 0 0 0]; [1 0 1 0 0 0]; [0 0 0 1 1 0]; [0 0 0 1 0 1]];

linearized_IO = [[1 1 0 -1 0 -2]; [1 0 1 0 -1 0]; [-1 0 -2 1 1 0]; [0 -1 0 1 0 1]];

ideal_linearized_IO = [[1 1 0 -1 0 -1]; [1 0 1 -1 -1 0]; [-1 0 -1 1 1 0]; [-1 -1 0 1 0 1]];

nl_IO_c = nonlinear_IO-ones(4,1)*mean(nonlinear_IO,1);
lz_IO_c = linearized_IO-ones(4,1)*mean(linearized_IO,1);
ilz_IO_c = ideal_linearized_IO-ones(4,1)*mean(ideal_linearized_IO,1);

[U_nl,S_nl,V_nl] = svd(nonlinear_IO.')
[U_lz,S_lz,V_lz] = svd(linearized_IO.')
[U_ilz,S_ilz,V_ilz] = svd(ideal_linearized_IO.')

actual_preoutputs = load('nonlinear_nhidden_4_rseed_0_final_pre_outputs.csv')
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

temp = V_nl(:,1:4);
temp(:,3) = -temp(:,3); %Flip mode input and outputs for visual consistency
imagesc(temp.',[-1,1])
xlabel('inputs','fontsize',30)
ylabel('modes','fontsize',30)
set(gca,'xtick',1:4)
set(gca,'ytick',1:4)    
colormap(redbluecmap)
hold on;
rectangle('Position',[0.53,0.53,3.96,0.98],...
         'LineWidth',5)
rectangle('Position',[0.53,1.52,3.96,1.0],...
         'LineWidth',5)   
rectangle('Position',[0.53,2.52,3.96,1.0],...
         'LineWidth',5)  
rectangle('Position',[0.53,3.52,3.96,0.97],...
         'LineWidth',5)  
hold off;
pos = get(gca, 'Position');
pos(2) = pos(2)+0.02;
pos(4) = pos(4)+0.02;
set(gca, 'Position', pos)

imagesc(S_nl(1:4,1:4),[-1.7321,1.7321])
xlabel('modes','fontsize',30)
set(gca,'xtick',1:4)
set(gca,'ytick',1:4)
colormap(redbluecmap)
hold on;
rectangle('Position',[0.52,0.52,0.98,0.98],...
         'LineWidth',5)
rectangle('Position',[1.5,1.5,0.98,1.0],...
         'LineWidth',5)   
rectangle('Position',[2.5,2.52,0.98,0.98],...
         'LineWidth',5)  
rectangle('Position',[3.5,3.5,0.98,0.98],...
         'LineWidth',5)  
hold off;
pos = get(gca, 'Position');
pos(2) = pos(2)+0.02;
pos(4) = pos(4)+0.02;
set(gca, 'Position', pos)


temp = U_nl(:,1:4);
temp(:,3) = -temp(:,3); %Flip mode input and outputs for visual consistency
imagesc(temp,[-1,1])
xlabel('modes','fontsize',30)
ylabel('outputs','fontsize',30)
set(gca,'ytick',1:6)
set(gca,'xtick',1:4)
colormap(redbluecmap)
hold on;
rectangle('Position',[0.52,0.52,0.98,6.0],...
         'LineWidth',5)
rectangle('Position',[1.5,0.52,1.0,6.0],...
         'LineWidth',5)   
rectangle('Position',[2.5,0.52,1.0,6.0],...
         'LineWidth',5)  
rectangle('Position',[3.5,0.52,0.99,6.0],...
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

%% Pre and post

figure
initial_preoutputs = load('nonlinear_nhidden_4_rseed_4_initial_pre_outputs.csv')
initial_c = initial_preoutputs-ones(4,1)*mean(initial_preoutputs,1);
[U_in,S_in,V_in] = svd(initial_c.')

imagesc(V_in.',[-1,1])
xlabel('inputs','fontsize',16)
ylabel('modes','fontsize',16)
colormap(redbluecmap)

figure
final_preoutputs = load('nonlinear_nhidden_4_rseed_6_final_pre_outputs.csv')
final_c = final_preoutputs-ones(4,1)*mean(final_preoutputs,1);
[U_f,S_f,V_f] = svd(final_c.')

imagesc(V_f.',[-1,1])
xlabel('inputs','fontsize',16)
ylabel('modes','fontsize',16)
colormap(redbluecmap)


initial_preoutputs = load('nonlinear_nhidden_4_rseed_25_initial_pre_outputs.csv')
initial_c = initial_preoutputs-ones(4,1)*mean(initial_preoutputs,1);
[U_in,S_in,V_in] = svd(initial_c.')
figure
imagesc(V_in.',[-1,1])
xlabel('inputs','fontsize',16)
ylabel('modes','fontsize',16)
colormap(redbluecmap)
figure
imagesc(U_in,[-1,1])
xlabel('modes','fontsize',16)
ylabel('outputs','fontsize',16)
colormap(redbluecmap)



final_preoutputs = load('nonlinear_nhidden_4_rseed_25_final_pre_outputs.csv')
final_c = final_preoutputs-ones(4,1)*mean(final_preoutputs,1);
[U_f,S_f,V_f] = svd(final_c.')
figure
imagesc(V_f.',[-1,1])
xlabel('inputs','fontsize',16)
ylabel('modes','fontsize',16)
colormap(redbluecmap)
figure
imagesc(U_f,[-1,1])
xlabel('modes','fontsize',16)
ylabel('outputs','fontsize',16)
colormap(redbluecmap)

%% Detailed run results
prev_fv = figure();
prev_fs = figure();
prev_fu = figure();
fv = figure();
fs = figure();
fu = figure();
for epoch = 0:10:500
    initial_preoutputs = load(sprintf('results/detailed_run/linear_nhidden_4_rseed_4_epoch_%i_pre_outputs.csv',epoch))
    initial_c = initial_preoutputs-ones(4,1)*mean(initial_preoutputs,1);
    [U_in,S_in,V_in] = svd(initial_c.')
    figure(fv)
    imagesc(V_in.',[-1,1])
    xlabel('inputs','fontsize',16)
    ylabel('modes','fontsize',16)
    colormap(redbluecmap)
    figure(fs)
    imagesc(S_in,[0,3])
    xlabel('modes','fontsize',16)
    colormap(redbluecmap)
    figure(fu)
    imagesc(U_in,[-1,1])
    xlabel('modes','fontsize',16)
    ylabel('outputs','fontsize',16)
    colormap(redbluecmap)

    
    pause
    figure(prev_fv)
    imagesc(V_in.',[-1,1])
    xlabel('inputs','fontsize',16)
    ylabel('modes','fontsize',16)
    colormap(redbluecmap)
    figure(prev_fs)
    imagesc(S_in,[0,3])
    xlabel('modes','fontsize',16)
    colormap(redbluecmap)
    figure(prev_fu)
    imagesc(U_in,[-1,1])
    xlabel('modes','fontsize',16)
    ylabel('outputs','fontsize',16)
    colormap(redbluecmap)

end

%% Do solutions SVD modes lie in subspace spanned by initial SVD modes

%linear
initial_preoutputs = load(sprintf('results/detailed_run/linear_nhidden_4_rseed_4_initial_pre_outputs.csv',epoch))
initial_c = initial_preoutputs-ones(4,1)*mean(initial_preoutputs,1);
[U_in,S_in,V_in] = svd(initial_c.')

final_preoutputs = load(sprintf('results/detailed_run/linear_nhidden_4_rseed_4_final_pre_outputs.csv',epoch))
final_c = final_preoutputs-ones(4,1)*mean(final_preoutputs,1);
[U_f,S_f,V_f] = svd(final_c.')

Q = [V_in(:,1:3) V_f(:,1)];
det(Q)
Q = [V_in(:,1:3) V_f(:,2)];
det(Q)
Q = [V_in(:,1:3) V_f(:,3)];
det(Q)

%nonlinear
initial_preoutputs = load(sprintf('results/detailed_run/nonlinear_nhidden_4_rseed_4_initial_pre_outputs.csv',epoch))
initial_c = initial_preoutputs-ones(4,1)*mean(initial_preoutputs,1);
[U_in,S_in,V_in] = svd(initial_c.')

final_preoutputs = load(sprintf('results/detailed_run/nonlinear_nhidden_4_rseed_4_final_pre_outputs.csv',epoch))
final_c = final_preoutputs-ones(4,1)*mean(final_preoutputs,1);
[U_f,S_f,V_f] = svd(final_c.')

Q = [V_in(:,1:3) V_f(:,1)];
det(Q)
Q = [V_in(:,1:3) V_f(:,2)];
det(Q)
Q = [V_in(:,1:3) V_f(:,3)];
det(Q)




%% playing with rotations

 
nonlinear_IO = [[1 1 0 0 0 0]; [1 0 1 0 0 0]; [0 0 0 1 1 0]; [0 0 0 1 0 1]];


nl_IO_c = nonlinear_IO-ones(4,1)*mean(nonlinear_IO,1);

[U_nl,S_nl,V_nl] = svd(nl_IO_c.')

V_test = V_nl;
U_test = U_nl;

V_test(:,2) = 1/sqrt(2)*(V_nl(:,2)+V_nl(:,3));
V_test(:,3) = 1/sqrt(2)*(V_nl(:,2)-V_nl(:,3));
U_test(:,2) = 1/sqrt(2)*(U_nl(:,2)+U_nl(:,3));
U_test(:,3) = 1/sqrt(2)*(U_nl(:,2)-U_nl(:,3));

imagesc(U_test*S_nl*V_test.'); colormap('redbluecmap');


actual_preoutputs = load('results/onehundredruns/linear_nhidden_4_rseed_0_pre_outputs.csv')
ap_c = actual_preoutputs-ones(4,1)*mean(actual_preoutputs,1);
[U_ap,S_ap,V_ap] = svd(ap_c.')

%% No biases

[U,S,V] = svd(nonlinear_IO)

[Ui,Si,Vi] = svd(ideal_linearized_IO)

%% Redone task


actual_preoutputs = load('results/letter_redux/nonlinear_nhidden_4_rseed_0_final_pre_outputs.csv')
ap_c = actual_preoutputs-ones(6,1)*mean(actual_preoutputs,1);
[U_ap,S_ap,V_ap] = svd(actual_preoutputs.')




nonlinear_IO = [[1 1 0 0 0 0 0 0 0 0]; [1 1 0 0 0 0 0 0 0 0]; [1 0 1 1 1 0 0 0 0 0]; [0 0 0 0 0 1 1 0 0 0]; [0 0 0 0 0 1 1 0 0 0]; [0 0 0 0 0 1 0 1 1 1]];
ideal_linearized_IO = [[1 1 0 0 0 -1 0 -1 -1 -1]; [1 1 0 0 0 -1 0 -1 -1 -1]; [1 0 1 1 1 -1 -1 0 0 0]; [-1 0 -1 -1 -1 1 1 0 0 0]; [-1 0 -1 -1 -1 1 1 0 0 0]; [-1 -1 0 0 0 1 0 1 1 1]];


%nonlinear_IO = [[1 1 0 0 0 0 0 0 0 0];  [1 0 1 1 1 0 0 0 0 0]; [0 0 0 0 0 1 1 0 0 0]; [0 0 0 0 0 1 0 1 1 1]];
%ideal_linearized_IO = [[1 1 0 0 0 0 0 -1 -1 -1]; [1 0 1 1 1 0 -1 0 0 0]; [0 0 -1 -1 -1 1 1 0 0 0]; [0 -1 0 0 0 1 0 1 1 1]];


nl_IO_c = nonlinear_IO-ones(6,1)*mean(nonlinear_IO,1);
ilz_IO_c = ideal_linearized_IO-ones(6,1)*mean(ideal_linearized_IO,1);

[U_nl,S_nl,V_nl] = svd(nonlinear_IO.')
[U_ilz,S_ilz,V_ilz] = svd(ideal_linearized_IO.')

figure
imagesc(nonlinear_IO.',[-1 1]); colormap('redbluecmap')
figure
imagesc(ideal_linearized_IO.',[-1 1]); colormap('redbluecmap')


figure
imagesc(V_nl(:,1:4).',[-1 1]); colormap('redbluecmap')
figure
imagesc(V_ilz(:,1:2).',[-1 1]); colormap('redbluecmap')

m_S_nl = max(max(S_nl));
figure
imagesc(S_nl(1:4,1:4),[-m_S_nl, m_S_nl]); colormap('redbluecmap')

m_S_ilz = max(max(S_ilz));
figure
imagesc(S_ilz(1:2,1:2),[-m_S_ilz, m_S_ilz]); colormap('redbluecmap')


figure
imagesc(U_nl(:,1:4),[-1 1]); colormap('redbluecmap')
figure
imagesc(U_ilz(:,1:2),[-1 1]); colormap('redbluecmap')

%% blah
counts = zeros(100,1);
counts2 = zeros(100,1);
for run = 0:99
    actual_preoutputs = load(sprintf('nonlinear_nhidden_4_rseed_%i_final_pre_outputs.csv',run));  
    counts(run+1) = sum(actual_preoutputs(1:2,4) > min(actual_preoutputs(1:2,5:6),[],2)) + sum(actual_preoutputs(3:4,1) > min(actual_preoutputs(3:4,2:3),[],2));
    counts2(run+1) = sum(actual_preoutputs(1:2,4) < max(actual_preoutputs(1:2,5:6),[],2)) + sum(actual_preoutputs(3:4,1) < max(actual_preoutputs(3:4,2:3),[],2));

end
hist(counts)
[~,i] = sort(counts)
hist(counts2)