%%% Code for creating a Header file (maybe later automatically done in
%%% converter). Needed for QSM PIPELINE testing with SEPIA
% Remember that Sepia expects echo times in seconds

%%% Whole body cropped (Initially it's 828 long)
B0 = 3; % Magnetic Field Strength in Tesla
B0_dir = [0;0;1]; % Direction of B0
CF = 127740000; % Central Frequency
%TE = [0.001, 0.002, 0.003, 0.004, 0.005, 0.010, 0.015, 0.020, 0.030,
%0.040]; % I call this our custom TEs
TE = [0.004, 0.008, 0.012, 0.016, 0.020, 0.024]; % For more realistic TEs
% Echo time or list of echo times
matrixSize = [101, 171, 141]; % The FOV of the image used
voxelSize = [0.9766, 0.9766, 2.344]; % Voxel size 
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

save("correct_swiss_qsm_invivo.mat",'B0','CF','B0_dir',"TE", "matrixSize","voxelSize");

%% Custom TEs with Swiss acq FOV - te's of first Swiss data for siumulation
TE = [0.00686, 0.01314, 0.01942, 0.0257, 0.03198] ;
B0 = 3;
B0_dir = [0;0;1];
CF = 127740000;
% Echo time or list of echo times
matrixSize = [301, 351, 128]; % The "dimensions" of the image used
voxelSize = [0.9766, 0.9766, 2.344]; % Pixdim 

save("swiss_qsm1_te_seconds_sim.mat",'B0','CF','B0_dir',"TE", "matrixSize","voxelSize");

%% Swiss TEs with Swiss acq FOV- SIMULATION
TE = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40] ;
B0 = 3;
B0_dir = [0;0;1];
CF = 127740000;
% Echo time or list of echo times
matrixSize = [301, 351, 128]; % The "dimensions" of the image used
voxelSize = [0.976562, 0.976562, 2.344]; % Pixdim 

save("custom_qsm_sim.mat",'B0','CF','B0_dir',"TE", "matrixSize","voxelSize");



%% Custom TEs with Swiss acq FOV TEs in seconds only to test in MEDI?
TE = [0.001, 0.002, 0.003, 0.004, 0.005, 0.010, 0.015, 0.020, 0.030, 0.040] ; % In seconds
B0 = 3; % In Tesla
B0_dir = [0;0;1];
CF = 127740000; % In Hz
% Echo time or list of echo times
matrixSize = [301, 351, 128]; % The "dimensions" of the image used
voxelSize = [0.976562, 0.976562, 2.344]; % Pixdim 

save("custom_qsm_sim_for_medi_CF_Hz.mat",'B0','CF','B0_dir',"TE", "matrixSize","voxelSize");

%% Sepia Header for sct_100 in vivo QSM - test#1

TE = [2.63, 7.63, 12.63, 17.63, 22.63];
B0 = 3;
B0_dir = [0;0;1];
CF = 123.249136;
matrixSize = [304, 304, 20];
voxelSize = [0.523, 0.523, 5];

save("sct_100_qsm.mat", 'B0','CF','B0_dir',"TE", "matrixSize","voxelSize")

%% Sepia Header for Swiss data mk2.
% We have 4 subjects, and each has 5 echoes
% For HC1, HC2 :
% RepetitionTime": 0.038,
% FlipAngle": 8

TE = [0.00685, 0.01085, 0.01485, 0.01885, 0.02285];
B0 = 3;
B0_dir = [0;0;1];
CF = 123.249;
matrixSize = [384, 384, 16];
voxelSize = [0.5833, 0.5833, 5];

save("swiss_header_mk2_te_sec.mat", 'B0','CF','B0_dir',"TE", "matrixSize","voxelSize")

%%
% Values for GRE acq for chi-fitting project db0-032, db0-033 and db0-035
B0 = 3; % Magnetic Field Strength in Tesla
B0_dir = [0; 0; 1]; % Direction of B0
CF = 127740000; % Central Frequency
TE = [3.27, 5.2, 7.13]; % In seconds 
% Echo time or list of echo times
matrixSize = [144, 144, 20]; % The "dimensions" of the image used
voxelSize = [1.9792, 1.9792, 2, 3]; % Pixdim 

save("db032_header.mat", 'B0', 'CF', 'B0_dir', "TE", "matrixSize", "voxelSize");

%% Swiss invivo Acquisition for Simulation Phantom mk2 (final version)
% Now that we know that Sepia expects echo times in seconds
TE = [0.00685, 0.01085, 0.01485, 0.01885, 0.02285];
B0 = 3;
B0_dir = [0;0;1];
CF = B0*42.58*1e6; % In Hz, B0 * gyromagnetic ratio
matrixSize = [301, 351, 128]; % The "dimensions" of the image used
voxelSize = [0.976562, 0.976562, 2.344]; % Pixdim 

save("qsm_sc_phantom_swiss_params.mat",'B0','CF','B0_dir',"TE", "matrixSize","voxelSize");

%% Ideal invivo Acquisition for Simulation Phantom mk2 (final version)
% Now that we know that Sepia expects echo times in seconds
TE = [0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04];
B0 = 3;
B0_dir = [0;0;1];
CF = 127740000; % In Hz, B0 * gyromagnetic ratio
matrixSize = [301, 351, 128]; % The "dimensions" of the image used
voxelSize = [0.976562, 0.976562, 2.344]; % Pixdim 

save("qsm_sc_phantom_custom_params.mat",'B0','CF','B0_dir',"TE", "matrixSize","voxelSize");

%% Testing headers
% Now that we know that Sepia expects echo times in seconds
TE = [6.85, 10.85, 14.85, 18.85, 22.85];
B0 = 3;
B0_dir = [0;0;1];
CF = 127740000; % In Hz, B0 * gyromagnetic ratio
matrixSize = [301, 351, 128]; % The "dimensions" of the image used
voxelSize = [0.976562, 0.976562, 2.344]; % Pixdim 

save("testing_headers2.mat",'B0','CF','B0_dir',"TE", "matrixSize","voxelSize");

%% Header for chi_003, we have 4 testing protocols:
% for m2: [0.00693, 0.01185, 0.01685, 0.02185, 0.02685], 
% matrixSize = [384, 384, 16], voxelSize = [0.4427, 0.4427, 5]
% for m4: [0.00661, 0.01067, 0.01473, 0.01879, 0.02285], 
% matrixSize = [320, 320, 16], voxelSize = [0.5312, 0.5312, 5];
% for m6: [0.00606, 0.01106, 0.01606, 0.02106, 0.02606], 
% matrixSize = [272, 272, 16], voxelSize = [0.625, 0.625, 5]
% Now that we know that Sepia expects echo times in seconds
TE = [0.00606, 0.01106, 0.01606, 0.02106, 0.02606];
B0 = 3;
B0_dir = [0;0;1];
CF = 123249367; % In Hz, B0 * gyromagnetic ratio 123.249367 127740000
matrixSize = [272, 272, 16]; % The "dimensions" of the image used
voxelSize = [0.625, 0.625, 5]; % Pixdim 

save("chi_003_m6_header.mat",'B0','CF','B0_dir',"TE", "matrixSize","voxelSize");

%% Header for chi_004, we have 3 acq. protocols:
% for m1, 2 &3 all have same echo train: [0.00693, 0.01185, 0.01685, 0.02185, 0.02685], 
% matrixSize = [384, 384, 16], voxelSize = [0.4427, 0.4427, 5]

TE = [0.00693, 0.01185, 0.01685, 0.02185, 0.02685];
B0 = 3;
B0_dir = [0;0;1];
CF = 123249367; % In Hz, B0 * gyromagnetic ratio 123.249367 127740000
matrixSize = [384, 384, 16]; % The "dimensions" of the image used
voxelSize = [0.4427, 0.4427, 5]; % Pixdim 

save("chi_004_m1_header.mat",'B0','CF','B0_dir',"TE", "matrixSize","voxelSize");

%% Header for Swiss 7T 
% 

TE = [0.00455, 0.00881, 0.01307, 0.01733, 0.02159];
B0 = 7;
B0_dir = [0;0;1];
% CF = 123249367; % In Hz, B0 * gyromagnetic ratio 123.249367 127740000 ->
% The above CF is for 3T, for Swiss 7T they used a 7T Siemens 
CF = 297189000;
matrixSize = [500, 510, 1]; % The "dimensions" of the image used
voxelSize = [0.256, 0.256, 6]; % Pixdim 

save("swiss_7T_m1_s1_header.mat",'B0','CF','B0_dir',"TE", "matrixSize","voxelSize");