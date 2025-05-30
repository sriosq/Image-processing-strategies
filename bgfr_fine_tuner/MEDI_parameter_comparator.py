import os
import json
import matplotlib.pyplot as plt
import re

def load_params_across_runs(iter_path, algo_name, param_name, json_fn = "sidecar_data.json"):

    param_values = []
    run_numbers = []

    pattern = re.compile(rf"{re.escape(algo_name)}_run(\d+)$")

    for folder in sorted(os.listdir(iter_path)):
        match = pattern.match(folder)

        if match:
            run_index = int(match.group(1))
            json_path = os.path.join(iter_path, folder, json_fn)

            if os.path.isfile(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    if param_name in data:
                        param_values.append(data[param_name])
                        run_numbers.append(run_index)
                    else:
                        print(f"⚠️ Parameter '{param_name}' not found in {json_path}")
            else:
                print(f"⚠️ JSON file not found in {json_path}")

    return run_numbers, param_values

def plot_parameter(run_numbers, param_values, param_name, algo_name):
    plt.figure(figsize=(8, 5))
    plt.plot(run_numbers, param_values, marker='o', linestyle='None')
    plt.xlabel("Run Number")
    plt.ylabel(param_name)
    plt.title(f"{param_name} across runs for {algo_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

##########################################################################################################################################################################

iteration_folder = r"E:\msc_data\sc_qsm\new_gauss_sims\mrsim_outpus\cropped_ideal\dipole_inversion_tests\iter_MEDI\VNS_on_smv_merit_off\RMSE_test1_with_mag_500_evals"
algo_name = "MEDI"
param_to_plot = "Lambda"

runs, values = load_params_across_runs(iteration_folder, algo_name, param_to_plot)
plot_parameter(runs, values, param_to_plot, algo_name)