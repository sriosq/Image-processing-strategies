% Go to the folder where the data is located
% Make sure you copy all the echoes correctly!
% Important to not get ME data with repeated echoes!

echo1_swiss = "T20250625121020_NS_V_S18_i00001_ph.nii";
echo2_swiss = "T20250625121020_NS_V_S18_e2_i00001_ph.nii";
echo3_swiss = "T20250625121020_NS_V_S18_e3_i00001_ph.nii";
echo4_swiss = "T20250625121020_NS_V_S18_e4_i00001_ph.nii";
echo5_swiss = "T20250625121020_NS_V_S18_e5_i00001_ph.nii";

fn = 'slabs_alone_concat_echoes/NS_V_S18_phs';

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
combined_data = cat(4, echo1_data, echo2_data, echo3_data, echo4_data, echo5_data);  % 4th dimension for echoes
% Then copy the header information from any of the echoes and update the
% information
combined_info = info_echo1;
combined_info.ImageSize = size(combined_data);  % Update image size to 4D (including echoes)
combined_info.Datatype = class(combined_data);  % Update the datatype if necessary
combined_info.PixelDimensions = [info_echo1.PixelDimensions, 3];  % Add echo dimension to pixel dimensions

niftiwrite(combined_data, fn , combined_info, 'Compressed', true);