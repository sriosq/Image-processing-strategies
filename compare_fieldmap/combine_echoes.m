% Go to the folder where the data is located

echo1_file = "ASPIRE_brain_three_echo_20240823134311_7_e1.nii.gz";
echo2_file = "ASPIRE_brain_three_echo_20240823134311_7_e2.nii.gz";
echo3_file = "ASPIRE_brain_three_echo_20240823134311_7_e3.nii.gz";

% Read the NIfTI images and their header info
info_echo1 = niftiinfo(echo1_file);
echo1_data = niftiread(echo1_file);

info_echo2 = niftiinfo(echo2_file);
echo2_data = niftiread(echo2_file);

info_echo3 = niftiinfo(echo3_file);
echo3_data = niftiread(echo3_file);

% Assuming the images are the same size, concatenate the echoes along a new dimension
combined_phase_data = cat(4, echo1_data, echo2_data, echo3_data);  % 4th dimension for echoes
% Then copy the header information from any of the echoes and update the
% information
combined_info = info_echo1;
combined_info.ImageSize = size(combined_phase_data);  % Update image size to 4D (including echoes)
combined_info.Datatype = class(combined_phase_data);  % Update the datatype if necessary
combined_info.PixelDimensions = [info_echo1.PixelDimensions, 3];  % Add echo dimension to pixel dimensions


niftiwrite(combined_phase_data, 'brain_complete_mag',combined_info);