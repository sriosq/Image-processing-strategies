% Running this before starting sepia processing

clear

path = 'R:/Poly_MSc_Code/libraries_and_toolboxes/sepia';
path2 = 'R:/Poly_MSc_Code/libraries_and_toolboxes/toolboxes';

data_path = '';
addpath(path);
addpath(genpath(path2));
cd(path);

%Run to open sepia 
sepia
%% 

phase = niftiread("no_gauss_sim_phase.nii.gz");
% This image does not have a header pre defined so we need to add it
% And we need to be able to use view_nii!!!!!!!!!!!!!!!!!!!!!!!!!
%%
% Traverse to code and load header_mrsim.mat
l_header = load("header_mrsim.mat");
header = l_header.header;
% Create a basic NIfTI header using niftiinfo (if you have the file)
info = phase;

% Update the header with your own header information
% This example assumes that your header.mat contains fields compatible with niftiinfo
fieldNames = fieldnames(header);

for i = 1:length(fieldNames)
    info.(fieldNames{i}) = header.(fieldNames{i});
end
niftiwrite(data, 'output_phase_with_header.nii', info);
%% 
% Analzying the Phase unwrapping steps
img_path = "Sepia_fieldmap.nii.gz";
fieldmap = niftiread(img_path);
unwrapped_path = "Sepia_part-phase_unwrapped.nii.gz";
unwrapped_phase = niftiread(unwrapped_path);

%% 

phase_residual = angle(exp(1i*phase) .* conj(exp(1i*unwrapped_phase)));


%% 
% This needs to be replaced by view_nii!!!!!!!!!1
slice_number = round(size(phase_residual, 3) / 2);
slice = phase_residual(:, :, slice_number);

figure;
imshow(slice, []);
imagesc(slice);
colormap gray;  % Set the colormap to grayscale
axis image;     % Set aspect ratio so that pixels are square
title(['Slice number ', num2str(slice_number)]);
colorbar;
%% 
% After loading the phase and the unwrapped phase lets go to analyze them

voxelGM = [51, 25, 110]; 
voxelWM= [49, 23, 110];
voxelCSF = [46, 24, 110];
voxelAIR = [16,42,110];

num_echoes = size(phase,4);

% Preallocate arrays to store voxel values
valuesGM = zeros(1, num_echoes);
valuesWM= zeros(1, num_echoes);
valuesCSF = zeros(1, num_echoes);
valuesAIR = zeros(1, num_echoes);

% Extract voxel values across echoes
for echo = 1:num_echoes
    valuesGM(echo) = phase(voxelGM(1), voxelGM(2), voxelGM(3), echo);
    valuesWM(echo) = phase(voxelWM(1), voxelWM(2), voxelWM(3), echo);
    valuesCSF(echo) = phase(voxelCSF(1), voxelCSF(2), voxelCSF(3), echo);
    valuesAIR(echo) = phase(voxelAIR(1), voxelAIR(2), voxelAIR(3), echo);
end

% Create a new figure for the plot
figure;

% Plot values for each voxel
hold on; % Hold on to plot multiple lines on the same graph
plot(1:num_echoes, valuesGM, '-o', 'DisplayName', 'Voxel GM'); 
plot(1:num_echoes, valuesWM, '-s', 'DisplayName', 'Voxel WM'); 
plot(1:num_echoes, valuesCSF, '-^', 'DisplayName', 'Voxel CSF'); 
plot(1:num_echoes, valuesAIR, '-v', 'DisplayName', 'Voxel AIR'); 

% Customize the plot
xlabel('Echo Number');
ylabel('Voxel Intensity');
title('Voxel Intensity Through Echoes');
legend; % Show the legend
grid on;
hold off; % Release the plot

%%
% Show the phase with the voxels 
% Select the slice to display (using the z-coordinate from the voxel points)
slice_number = voxelGM(3); % Example: using the z-coordinate of voxelGM
slice = phase(:, :, slice_number);

% Display the slice
figure;
imagesc(slice);
colormap gray;
axis image;
colorbar;
title(['Slice number ', num2str(slice_number)]);

% Hold on to overlay points
hold on;

% Plot the voxel points with a "+" sign
% Notice the swap of indices and adjustment for MATLAB's (x, y) = (column, row)
plot(voxelGM(2), voxelGM(1), 'r+', 'MarkerSize', 10, 'LineWidth', 2); % Plot for GM
plot(voxelWM(2), voxelWM(1), 'g+', 'MarkerSize', 10, 'LineWidth', 2); % Plot for WM
plot(voxelCSF(2), voxelCSF(1), 'b+', 'MarkerSize', 10, 'LineWidth', 2); % Plot for CSF
plot(voxelAIR(2), voxelAIR(1), 'y+', 'MarkerSize', 10, 'LineWidth', 2); % Plot for CSF
% Customize the plot
xlabel('X (Columns)');
ylabel('Y (Rows)');
legend('Voxel GM', 'Voxel WM', 'Voxel CSF');
hold off;







