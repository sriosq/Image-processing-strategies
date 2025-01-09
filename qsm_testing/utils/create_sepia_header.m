%%% Code for creating a Header file (maybe later automatically done in
%%% converter). Needed for QSM PIPELINE testing with SEPIA
%%% Whole body is axially 512 isotropic and then it gets cropped
%%% Initially it's 828

B0 = 3; % Magnetic Field Strength in Tesla
B0_dir = [0;0;1]; % Direction of B0
CF = 127740000; % Central Frequency
TE = [0.008 0.016 0.024 0.032  0.040];
% Echo time or list of echo times
matrixSize = [101, 171, 141]; % The "dimensions" of the image used
voxelSize = [0.9766, 0.9766, 2.344]; % Pixdim 
outpath = "E:/msc_data/sc_qsm/data/wb/simulation/header_qsm_tsting_hcrop.mat"
save(outpath,'B0','CF','B0_dir',"TE", "matrixSize","voxelSize");

%%
% Values for GRE acq from 23-08 // db0-030
B0 = 3; % Magnetic Field Strength in Tesla
B0_dir = [0;0;1]; % Direction of B0
CF = 127740000; % Central Frequency
TE = [0.0033 0.0052 0.0071];
% Echo time or list of echo times
matrixSize = [144, 144, 20]; % The "dimensions" of the image used
voxelSize = [1.9792, 1.9792, 2]; % Pixdim 

save("header_torso.mat",'B0','CF','B0_dir',"TE", "matrixSize","voxelSize");

%% 
B0 = 3;
B0_dir = [0;0;1];
CF = 127740000;
TE = [0.0033 0.0052];
% Echo time or list of echo times
matrixSize = [192, 320, 320]; % The "dimensions" of the image used
voxelSize = [1, 1, 1]; % Pixdim 

save("2choes_dB0_030.mat",'B0','CF','B0_dir',"TE", "matrixSize","voxelSize");