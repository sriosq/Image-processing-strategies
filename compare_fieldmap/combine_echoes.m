% Go to the folder where the data is located
% Make sure you copy the 3 echoes!
% Important to not get ME data with repeated echoes!

echo1_file = "ASPIRE_cervical_three_echo_insp_20250228140637_4_e1_ph.nii.gz";
echo2_file = "ASPIRE_cervical_three_echo_insp_20250228140637_4_e2_ph.nii.gz";
echo3_file = "ASPIRE_cervical_three_echo_insp_20250228140637_4_e3_ph.nii.gz";

echo1_swiss = "phase_echo1.nii";
echo2_swiss = "phase_echo2.nii";
echo3_swiss = "phase_echo3.nii";
echo4_swiss = "phase_echo4.nii";
echo5_swiss = "phase_echo5.nii";

% Read the NIfTI images and their header info
info_echo1 = niftiinfo(echo1_swiss);
echo1_data = niftiread(echo1_swiss);

info_echo2 = niftiinfo(echo2_swiss);
echo2_data = niftiread(echo2_swiss);

info_echo3 = niftiinfo(echo3_swiss);
echo3_data = niftiread(echo3_swiss);

info_echo4 = niftiinfo(echo4_swiss);
echo4_data = niftiread(echo4_swiss);

info_echo5 = niftiinfo(echo5_swiss);
echo5_data = niftiread(echo5_swiss);

% Assuming the images are the same size, concatenate the echoes along a new dimension
combined_phase_data = cat(4, echo1_data, echo2_data, echo3_data, echo4_data, echo5_data);  % 4th dimension for echoes
% Then copy the header information from any of the echoes and update the
% information
combined_info = info_echo1;
combined_info.ImageSize = size(combined_phase_data);  % Update image size to 4D (including echoes)
combined_info.Datatype = class(combined_phase_data);  % Update the datatype if necessary
combined_info.PixelDimensions = [info_echo1.PixelDimensions, 3];  % Add echo dimension to pixel dimensions


niftiwrite(combined_phase_data, 'concat_phase_swiss',combined_info,'Compressed',true);