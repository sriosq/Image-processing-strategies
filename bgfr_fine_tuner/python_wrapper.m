%% python_wrapper(in1,in2,in3,in4, outpath, mask_path, paramStrut)
%
% 4 inputs, correspond to whats used in config files
% Output and Mask are paths 
% Finally, paramStruct is used to get the parameters for algorithms
%
% Remember that dictionaries get converted to structs in MATLAB from python
% only when all the keys are strings! Not supported otherwise
%
% Input/Output filenames
function python_wrapper(in1, in2, in3, in4, alg_name, outpath, maskpath, paramStruct)
sepia_addpath;

disp("Begin from Python");

input_py = struct();
input_py(1).name = in1;
input_py(2).name = in2;
input_py(3).name = in3;
input_py(4).name = in4;

disp("Input received");
% Make sure radius_array is MATLAB-compatible (already double from Python)
if strcmp(alg_name, 'VSHARP')
    if isfield(paramStruct, 'bfr') && isfield(paramStruct.bfr, 'radius')
        radius_array = paramStruct.bfr.radius;
        disp(radius_array)
        paramStruct.bfr.radius = radius_array;
        else
            error('VSHARP selected but no radius field found in paramStruct.bfr');
    end
else
    disp(['Using algorithm: ', alg_name, ' (no extra handling needed)'])
end

output_basename = outpath;
params = paramStruct;
disp("Algorithm Parameters received");
disp("Running ...");
mask = [maskpath];
sepiaIO(input_py, output_basename, mask, params);
disp("Done");
end




