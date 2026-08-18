# QSM Processing Pipeline GUI

This is the first application shell for combining the validated QSM automation in
`qsm_fine_tuner/automated_pipeline_running.py` with useful processing ideas from
`sc_qsm`.

The first milestone deliberately separates **machine configuration** from
**pipeline logic**. Paths are saved in `qsm_pp_gui/config.json`. That file is
visible and easy to edit, but ignored by Git because it contains paths specific
to one computer. Copy `config.example.json` when configuring a new computer.

## Installation

MATLAB Engine for Python must be installed into the same Conda environment
that runs the GUI. MATLAB R2024b supports Python 3.9 through 3.12, so this
project uses Python 3.11.

From the `Image-processing-strategies` repository root, create and activate the
environment:

```powershell
conda env create -f qsm_pp_gui\environment.yml
conda activate qsm_gui
```

The environment includes Tkinter and the Python packages needed for the GUI,
NIfTI processing, visualization, numerical processing, and tests.

Next, install MATLAB Engine into the active `qsm_gui` environment. For MATLAB
R2024b installed in its default Windows location:

```powershell
Push-Location "C:\Program Files\MATLAB\R2024b\extern\engines\python"
python -m pip install .
Pop-Location
```

For the default MATLAB R2024b location on macOS:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qsm_gui
python -m pip install "/Applications/MATLAB_R2024b.app/extern/engines/python"
```

If a different MATLAB release or installation directory is used, replace that
path with `<matlabroot>\extern\engines\python`. The MATLAB release must support
the Python version selected in `environment.yml`.

Verify that the active environment can import the Engine and start MATLAB:

```powershell
python -c "import matlab.engine; eng = matlab.engine.start_matlab(); print(eng.version()); eng.quit()"
```

Do not run the GUI until this command succeeds. Installing MATLAB Engine into
`base` or another environment does not make it available inside `qsm_gui`.

Julia and Spinal Cord Toolbox are separate applications. Install them normally
and make sure `julia` and `sct_deepseg` can be called from a newly opened
terminal:

```powershell
julia --version
sct_deepseg -h
sct_deepseb -h
```

See MathWorks' documentation for [installing MATLAB Engine for
Python](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)
and its [MATLAB/Python compatibility
table](https://www.mathworks.com/support/requirements/python-compatibility.html)
when preparing a computer with a different MATLAB release.

## Running the GUI

Activate the environment and run from the repository root:

```powershell
conda activate qsm_gui
python -m qsm_pp_gui
```

The module form above is preferred. Running the package directory directly is
also supported for convenience:

```powershell
python qsm_pp_gui
```

### Loading and resuming a participant

Use **Load project…** at the top of the application to select an existing
`*_qsm_project.json`. The GUI repopulates the acquisition and masking fields,
validates the saved SEPIA JSON and MATLAB headers, reconstructs milestone state
from existing outputs, and reports both completed milestones and the next step.
You do not need to rerun Inputs & header after restarting the application.

The coloured bar is persistent across all tabs:

- green with a check mark means the milestone is complete;
- blue with an arrow identifies the next pipeline stage;
- gray means the stage is still pending.

Each important operation is recorded in the project JSON with its completion
state and UTC timestamp. Older project files without milestone entries are
updated automatically after their files are validated.

Start on **Overview**, then use **Inputs & header** to initialize a participant.
The application creates a participant directory and prefixes its SEPIA header
and project file with the participant identifier. Clicking **Validate and create
SEPIA headers** writes the JSON metadata, launches MATLAB through MATLAB Engine
for Python, runs `utils/create_sepia_header.m`, and verifies the resulting `.mat`
file before marking the step complete.

Julia and Spinal Cord Toolbox commands must be available on the operating
system's `PATH`. They are requirements rather than GUI settings. On the
**Settings** tab, select:

- ROMEO's `romeo.jl` script;
- the SEPIA root directory.

The Settings tab shows the current operating system, Python executable, active
Conda environment, MATLAB Engine import status, and platform-appropriate setup
commands. Environment changes take effect only after restarting the GUI from
the corrected environment.

Enable **Developer mode** on the Settings tab to show a subprocess log beneath
the workflow. Before every SCT or ROMEO subprocess starts, the GUI prints the
exact command with all paths and arguments. Normal progress messages remain
visible whether developer mode is enabled or not. This preference is saved in
the local `config.json`.

## Acquisition units

Echo-time and central-frequency units are selected explicitly; the GUI never
guesses from numeric magnitude. Echo times may be entered in `ms` or `s`, and
central frequency in `Hz` or `MHz`. The SEPIA JSON/MAT headers always receive
seconds and Hz, while ROMEO receives milliseconds. The project JSON records the
original values and selected units so reopening a project restores exactly what
was entered. Older project files without unit metadata retain the historical
defaults of milliseconds and Hz.

## Masking workflow

The Masks tab creates `<output-root>/<participant-id>/masking/` and runs each
operation as a separate checkpoint:

1. Extract the first echo from the selected 3D/4D meGRE magnitude and create SC
   and GM masks using `sct_deepseg`.
2. Create WM as SC minus GM using `sct_maths`.
3. Segment the spinal cord in the selected T1-weighted image.
4. Choose the first and last disc label, then open the interactive SCT viewer.
5. Use those labels to create a T1-space vertebral-level segmentation.
6. Manually open, inspect, and if necessary correct every SC, GM, WM, T1 SC,
   and T1-space vertebral-level output. Explicitly record QC approval in the GUI.
7. Register the first-echo meGRE to T1w using the meGRE and T1w SC masks, then
   apply the inverse T1w-to-meGRE warp to the vertebral-level segmentation.
8. Manually inspect and, if necessary, correct the final meGRE-space vertebral
   labels before recording registration QC approval.

The final outputs use participant-prefixed names and are recorded under the
`masking` section of the participant project JSON.

For a multi-echo 4D magnitude, **Also create a second GM mask from the
echo-averaged magnitude** creates a voxelwise arithmetic mean across the echo
dimension and saves:

- `<participant-id>_echo-avg_magnitude.nii.gz`
- `<participant-id>_echo-avg_desc-GM_mask.nii.gz`

The original first-echo GM segmentation is always retained. The WM mask remains
SC minus the first-echo GM mask, so enabling the optional comparison does not
silently change existing downstream processing. The option requires at least two
echoes in a 4D NIfTI; leave it off for a 3D acquisition. Both GM masks must be
manually inspected and corrected when necessary before masking QC is approved.

Existing non-empty outputs are reused so checked work is not silently overwritten.

Enable **Force rerun selected step** to deliberately replace the output of the
chosen action. Re-running the T1 cord mask or disc labels also removes the old
vertebral-level result because it is no longer valid. The GUI always extracts
and verifies a 3D first-echo magnitude before calling SCT; SCT segmentation
models do not accept the original 4D multi-echo magnitude.

More specifically:

- forcing meGRE masks regenerates the first echo and the SC, GM, and WM masks,
  plus the echo-average image and GM mask when that option is enabled;
- forcing the T1 SC mask invalidates the vertebral-level result;
- forcing disc labels replaces the labels and invalidates vertebral levels;
- forcing vertebral levels replaces only the vertebral-level result.

After a rerun, milestones are refreshed from the files on disk so an invalidated
downstream result is shown as incomplete.

Manual QC is mandatory. File creation alone does not complete the masking stage,
and registration cannot start until meGRE and T1w mask QC is approved. The GUI
cannot determine whether anatomical boundaries or vertebral labels are correct;
the NIfTI outputs must be opened in an appropriate viewer and corrected when
needed. The masking stage becomes green only after the warped meGRE-space labels
receive their own manual QC approval.

Registration outputs are stored in `masking/register/`:

```text
warp_megre2t1w.nii.gz
warp_t1w2megre.nii.gz
<id>_space-T1w_desc-meGRE_registered.nii.gz
<id>_space-meGRE_desc-SC_vertlevels_dseg.nii.gz
```

The pre-registration label image is named
`<id>_space-T1w_desc-SC_vertlevels_dseg.nii.gz`. The final transformation uses
nearest-neighbor interpolation (`-x nn`) to preserve discrete vertebral labels.

## Field mapping

The Field map tab uses the original 4D magnitude and phase NIfTIs. It verifies
that their shapes match, that the number of volumes equals the number of echo
times, and that the 3D meGRE SC mask matches their spatial dimensions. The
SEPIA header retains echo times in seconds, but the GUI multiplies them by 1000
because ROMEO requires echo times in **milliseconds**.

Phase-offset correction can be `bipolar`, `on`, or `off`. Both ROMEO variants
receive the meGRE mask through `-k`:

- **masked fieldmap** adds `-u`;
- **unmasked fieldmap** omits `-u`.

Outputs are stored alongside `masking/` under the participant directory:

```text
fieldmap/
├── masked_fieldmap/
│   ├── B0.nii
│   ├── corrected_phase.nii
│   ├── <id>_desc-b0_fieldmap.nii.gz
│   └── <id>_desc-corrected_phase.nii.gz
└── unmasked_fieldmap/
    ├── B0.nii
    ├── corrected_phase.nii
    ├── <id>_desc-b0_fieldmap.nii.gz
    └── <id>_desc-corrected_phase.nii.gz
```

ROMEO's `-o` argument receives the variant directory, not a filename. ROMEO
writes the fixed raw names `B0.nii` and `corrected_phase.nii`; the GUI preserves
those files and creates gzip-compressed, participant-prefixed copies. The Field
map milestone is complete only when both compressed products exist for both
variants.

For quantitative visual QC, enter `cmin`, `cmax`, and a Matplotlib colorbar
name such as `bwr`, then select **Create masked and unmasked B0 QC PNGs**. The
GUI runs the reusable `utils.deepseb.call_deepseb` helper once for each B0 map,
using the meGRE mask as an outline. Corrected phase is not rendered. Each PNG
has the same basename as its participant-prefixed B0 map:

```text
<id>_desc-b0_fieldmap.nii.gz
<id>_desc-b0_fieldmap.png
```

Changing `cmin`, `cmax`, or `cbar` automatically regenerates both PNGs, even if
PNG files already exist. With unchanged settings, existing PNGs are reused
unless **Force rerun fieldmap or PNG output** is enabled. In Developer mode,
confirm the effective values in the exact logged `sct_deepseb` command.

## Noise SD and weights

After field-map QC, open **4 Noise & weights**. The GUI loads the original 4D
magnitude, the meGRE SC mask, and the echo times in seconds from the saved SEPIA
header. **Create noise SD and DI weights** writes both maps to the participant's
`noise_weights` folder. Existing non-empty maps are reused unless **Force rerun
both maps** is selected. A rerun invalidates noise/weights QC and downstream BGFR
approval. Open both NIfTI maps, inspect/correct the inputs if needed, select the
manual-QC checkbox, and record approval before BGFR.

## Background-field removal

The **5 BGFR** procedure menu provides:

- `comp_bgfr`: all five default and all five optimized algorithms
- `default`: PDF, LBV, SHARP, RESHARP, and VSHARP default parameters
- `optimized`: the corresponding optimized parameters

BGFR uses the selected B0 fieldmap, MATLAB SEPIA header, meGRE SC mask, and noise
SD map. The SEPIA directory comes from Settings; MATLAB Engine is loaded only
when the run starts. Each algorithm receives its own folder containing the local
field, a portable parameter JSON, and a QC PNG. SEPIA initially generates its
fixed output name, after which the GUI performs a cross-platform rename to
`<participant-id>_desc-localfield.nii.gz`; its JSON and PNG use the same stem.
The local-field display range is
taken directly from the BGFR `cmin`, `cmax`, and `cbar` controls. Enable developer
mode to see each conceptual `python_wrapper(...)` call and every exact
`sct_deepseb` subprocess command. Force rerun recreates the selected procedure
and invalidates BGFR QC. Manually inspect every local field/PNG before recording
BGFR approval.

The reusable `utils.deepseb.call_deepseb` function is intentionally independent
of map type: fieldmaps, BGFR local fields, and later chi maps can each supply
their own input, output, display limits, colorbar, and optional outline mask.

## SEPIA external toolboxes

- Download the external QSM toolboxes required by the algorithms you intend to
  run, including MEDI, STI Suite, FANSI, SEGUE, MRI Susceptibility Calculation,
  and mritools where applicable. Preserve each toolbox's original directory
  structure and configure its location in SEPIA's
  `SpecifyToolboxesDirectory.m`/universal-variables configuration. Add only the
  SEPIA home directory to MATLAB's path: `python_wrapper.m` calls
  `sepia_addpath`, which lets SEPIA activate the configured dependency for the
  selected algorithm. Do not rely on machine-specific toolbox paths embedded in
  Python. Check each external toolbox's own license before redistribution.

See the official [SEPIA installation and dependency instructions](https://sepia-documentation.readthedocs.io/en/latest/getting_started/Installation.html).

## Dipole inversion

After BGFR QC, select a local-field result in **6 Dipole inversion**. The GUI
passes the original magnitude as SEPIA input 2 and the generated DI weights as
input 3, matching `python_wrapper.m`. Procedure choices are `comp_di`, `default`,
`optimized`, and `automatic`. Each selected algorithm writes
`<participant-id>_desc-chimap.nii.gz`, a JSON and PNG with the same stem, under
its algorithm folder in `dipole_inversion`. The
default chi-map display range is `-0.04` to `0.04` with `bwr`. Force reruns
invalidate final DI approval; manually inspect every chi map and PNG before
recording the final pipeline milestone.

When a project is loaded, the DI local-field input prefers the saved `opt_pdf`
result because this is the spinal-cord QSM default. If it is unavailable, the
first existing BGFR result is shown instead. The path remains browseable for
intentional method comparisons; no filename or numeric unit is guessed.
Milestone validation performed when a button is clicked does not reload or
overwrite values currently typed into the field-map form.

Finally, manually inspect both B0 maps and PNGs and select **Record fieldmap QC
approval**. This is a separate `fieldmap_qc` milestone and is the readiness gate
for BGFR and DI. Re-running either ROMEO variant invalidates the PNG and final
QC milestones.

The MATLAB executable is intentionally not requested. The current Python code
uses MATLAB Engine for Python, so the relevant check is whether `matlab.engine`
can be imported. SEPIA paths will be added to each MATLAB engine session from
this configuration.

`sct_deepseg` belongs to Spinal Cord Toolbox (SCT), not to SEPIA. It creates
spinal-cord or gray-matter segmentations and is relevant to the masking workflow
borrowed from `sc_qsm`. It can remain as `sct_deepseg` when SCT is available on
`PATH`. It is optional until those masking stages are enabled.

## MATLAB utility

`utils/create_sepia_header.m` converts the JSON metadata produced by the GUI
into the MATLAB `.mat` header consumed by SEPIA. It accepts either the generated
`*_sepia_header.json` or `*_qsm_project.json`:

```matlab
header_path = create_sepia_header("sub-001_sepia_header.json");
```

An explicit output path is optional:

```matlab
header_path = create_sepia_header("sub-001_qsm_project.json", ...
    "sub-001_custom_header.mat");
```
