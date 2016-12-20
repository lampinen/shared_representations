nonlinear_IO = [[1 1 0 0 0 0]; [1 0 1 0 0 0]; [0 0 0 1 1 0]; [0 0 0 1 0 1]];

linearized_IO = [[1 1 0 -1 0 -2]; [1 0 1 0 -1 0]; [-1 0 -2 1 1 0]; [0 -1 0 1 0 1]];

ideal_linearized_IO = [[1 1 0 0 0 -1]; [1 0 1 0 -1 0]; [0 0 -1 1 1 0]; [0 -1 0 1 0 1]];


nl_IO_c = nonlinear_IO-ones(4,1)*mean(nonlinear_IO,1);
lz_IO_c = linearized_IO-ones(4,1)*mean(linearized_IO,1);
ilz_IO_c = ideal_linearized_IO-ones(4,1)*mean(ideal_linearized_IO,1);


[U_nl,S_nl,V_nl] = svd(nl_IO_c.')
[U_lz,S_lz,V_lz] = svd(lz_IO_c.')
[U_ilz,S_ilz,V_ilz] = svd(ilz_IO_c.')

actual_preoutputs = load('nonlinear_nhidden_2_rseed_0_pre_outputs.csv')
ap_c = actual_preoutputs-ones(4,1)*mean(actual_preoutputs,1);
[U_ap,S_ap,V_ap] = svd(ap_c.')