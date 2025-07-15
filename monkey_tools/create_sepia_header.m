%%% Code for creating a Header file (maybe later automatically done in
%%% converter). Needed for QSM PIPELINE testing with SEPIA
%%% Whole body is axially 512 isotropic and then it gets cropped
%%% Initially it's 828

B0 = 3; % Magnetic Field Strength in Tesla
B0_dir = [0;0;1]; % Direction of B0
CF = 127740000; % Central Frequency
%TE = [0.001, 0.002, 0.003, 0.004, 0.005, 0.010, 0.015, 0.020, 0.030,
%0.040]; % For Weird Tes
TE = [0.004, 0.008, 0.012, 0.016, 0.020, 0.024]; % For more realistic TEs
% Echo time or list of echo times
matrixSize = [101, 171, 141]; % The "dimensions" of the image used
voxelSize = [0.9766, 0.9766, 2.344]; % Pixdim 
outpath = "E:/msc_data/sc_qsm/data/wb/simulation/header_qsm_tsting_hcrop2.mat";
outpath2 = "E:/msc_data/sc_qsm/data/cropped/piece-wise/simulation/TE_1_weird_40/header_qsm_tsting_hcrop2.mat";
outpath3 = "E:/msc_data/sc_qsm/data/cropped/piece-wise/simulation/TE_4_4_24/header_qsm_tsting_hcrop2.mat";
% Choose correct outpath AVOID overwriting!!
save(outpath3,'B0','CF','B0_dir',"TE", "matrixSize","voxelSize");

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

%% Swiss invivo Acquisition header
TE = [6.86, 13.14, 19.42, 25.7, 31.98];
B0 = 3;
B0_dir = [0;0;1];
CF = 4*42.58*1e6; % In Hz, B0 * gyromagnetic ration
% Echo time or list of echo times
matrixSize = [384, 384, 16]; % The "dimensions" of the image used
voxelSize = [0.5, 0.5, 5]; % Pixdim 

save("correct_swiss_qsm_sim.mat",'B0','CF','B0_dir',"TE", "matrixSize","voxelSize");
>>>>>>> b65df692141ffcee5ffbf0861d6d158704a671b3

%% Custom TEs with Swiss acq FOV
TE = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40] ;
B0 = 3;
B0_dir = [0;0;1];
CF = 127740000;
% Echo time or list of echo times
matrixSize = [301, 351, 128]; % The "dimensions" of the image used
voxelSize = [0.976562, 0.976562, 2.344]; % Pixdim 

save("custom_qsm_sim.mat",'B0','CF','B0_dir',"TE", "matrixSize","voxelSize");


%% Sepia Header for sct_100 in vivo QSM - test#1

TE = [2.63, 7.63, 12.63, 17.63, 22.63];
B0 = 3;
B0_dir = [0;0;1];
CF = 123.249136;
matrixSize = [304, 304, 20];
voxelSize = [0.523, 0.523, 5];

save("sct_100_qsm.mat", 'B0','CF','B0_dir',"TE", "matrixSize","voxelSize")



