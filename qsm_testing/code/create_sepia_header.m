%%% Code for creating a Header file (maybe later automatically done in
%%% converter). Needed for QSM PIPELINE testing with SEPIA

B0 = 3; % Magnetic Field Strength in Tesla
B0_dir = [0;0;1]; % Direction of B0
CF = 127740000; % Central Frequency
TE = [0.002 0.016 0.03  0.044 0.058];
% Echo time or list of echo times
matrixSize = [109, 129, 218]; % The "dimensions" of the image used
voxelSize = [0.976562, 0.976562, 2.344]; % Pixdim 

save("header_mrsim.mat",'B0','CF','B0_dir',"TE", "matrixSize","voxelSize");

%%
% Values for GRE acq from23-08
B0 = 3; % Magnetic Field Strength in Tesla
B0_dir = [0;0;1]; % Direction of B0
CF = 127740000; % Central Frequency
TE = [0.0033 0.0052 0.0071];
% Echo time or list of echo times
matrixSize = [144, 144, 20]; % The "dimensions" of the image used
voxelSize = [1.9792, 1.9792, 2]; % Pixdim 

save("header_torso.mat",'B0','CF','B0_dir',"TE", "matrixSize","voxelSize");