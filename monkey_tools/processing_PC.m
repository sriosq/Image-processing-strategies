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




%%

slices = {
    'T20250625121020_NS_V_S22_i00001_ph.nii'
    'T20250625121020_NS_V_S22_i00002_ph.nii'
    'T20250625121020_NS_V_S22_i00004_ph.nii'
    'T20250625121020_NS_V_S22_i00005_ph.nii'
    'T20250625121020_NS_V_S22_i00007_ph.nii'
    'T20250625121020_NS_V_S22_i00009_ph.nii'
    'T20250625121020_NS_V_S22_i00011_ph.nii'
    };

% Read the first file to get NIFTI header + number of slices
first = niftiread(slices{1});
info = niftiinfo(slices{1});

% Size of first file (Nx, Ny, Nz_in_first)
[Nx, Ny, Nz1] = size(first);

% Count total slices across all files
total_slices = 0;
for k = 1:numel(slices)
    temp = niftiread(slices{k});
    total_slices = total_slices + size(temp,3);
end

% Allocate 3D volume
data = zeros(Nx, Ny, total_slices, class(first));

% Fill the volume
index = 1;
for k = 1:numel(slices)
    temp = niftiread(slices{k});
    zcount = size(temp,3);
    data(:,:,index:index+zcount-1) = temp;
    index = index + zcount;
end

% Save final 3D volume
identifier = "NS_V_S22_e1_stacked_ph.nii";
outpath = "stacked/" + identifier;

niftiwrite(data, outpath, 'Compressed', true);
disp("Created: " + identifier);

%%
ref_slice  = "T20250625121020_NS_V_S22_i00001_ph.nii";
stacked_file = "stacked/concatenated_echoes/NS_V_S22_ph.nii.gz";
fixed_file   = "stacked/concatenated_echoes/fixed_NS_V_S22_ph";

ref_hdr = niftiinfo(ref_slice);
stacked_hdr = niftiinfo(stacked_file);
stacked_data = niftiread(stacked_file);

% Copy all three voxel dimensions (dx, dy, dz)
stacked_hdr.PixelDimensions(1:3) = ref_hdr.PixelDimensions(1:3);

% Also fix raw pixdim (NIfTI convention: pixdim(2)=dx, pixdim(3)=dy, pixdim(4)=dz)
if isfield(stacked_hdr, 'Raw') && isstruct(stacked_hdr.Raw) && numel(stacked_hdr.Raw.pixdim) >= 4
    stacked_hdr.Raw.pixdim(2:4) = ref_hdr.PixelDimensions(1:3);
end

% Ensure correct image size
stacked_hdr.ImageSize = size(stacked_data);

% Fix datatype
stacked_hdr.Datatype = class(stacked_data);

% Save
niftiwrite(stacked_data, fixed_file, stacked_hdr, 'Compressed', true);

% Print
fixed_hdr = niftiinfo(fixed_file);
disp(fixed_hdr.PixelDimensions);
disp(fixed_hdr.ImageSize);

