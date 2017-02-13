
avg_data = zeros(1,2000);
avg_alt_data = zeros(1,2000);

for run = 1:100
   data = load(sprintf('results/pfl/hinton_nhidden_12_rseed_%i_rep_tracks.csv',run-1)); 
   avg_data = avg_data + data;
   data = load(sprintf('results/pfl/hinton_alt_nhidden_12_rseed_%i_rep_tracks.csv',run-1)); 
   avg_alt_data = avg_alt_data + data;  
end

plot(1:1000,avg_data(1001:2000),1:1000,avg_alt_data(1001:2000))
legend('Sequential Learning -- Analogous','Sequential Learning -- Non-Analogous');
ylabel('MSE')
xlabel('Epoch')