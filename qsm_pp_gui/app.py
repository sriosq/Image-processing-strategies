"""Guided Tkinter interface for the QSM processing pipeline."""

from __future__ import annotations

import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .acquisition import Acquisition, parse_numbers
from .config import ToolConfig, default_config_path
from .fieldmap import FieldmapInputs, create_fieldmap_pngs, run_fieldmap, update_project_fieldmap
from .matlab_header import MatlabHeaderError, create_matlab_header
from .masking import (
    MaskingError,
    MaskingInputs,
    create_disc_labels,
    create_megre_masks,
    create_t1_sc_mask,
    create_vertebral_levels,
    register_megre_to_t1w,
    run_command,
    update_project_masking,
)
from .project import ProjectError, load_project, milestone_complete, milestone_summary, set_project_milestone
from .validation import validate_tools


class QsmApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("QSM Processing Pipeline")
        self.geometry("1060x760")
        self.minsize(900, 650)
        self.tool_config = ToolConfig.load()
        self.developer_mode = tk.BooleanVar(value=self.tool_config.developer_mode)
        self.developer_enabled = self.tool_config.developer_mode
        self.values: dict[str, tk.StringVar] = {}
        self.current_project_path: Path | None = None
        self.current_project: dict = {}
        self._build()

    def _build(self) -> None:
        style = ttk.Style(self)
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Section.TLabel", font=("Segoe UI", 12, "bold"))
        outer = ttk.Frame(self, padding=18)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="QSM Processing Pipeline", style="Title.TLabel").pack(anchor="w")
        header_row = ttk.Frame(outer)
        header_row.pack(fill="x", pady=(2, 8))
        ttk.Label(header_row, text="A guided, participant-centred workflow with validation and QC at every stage.").pack(side="left")
        ttk.Button(header_row, text="Load project…", command=self._load_project_dialog).pack(side="right")
        self.progress_frame = tk.Frame(outer, background="#d9dde3")
        self.progress_frame.pack(fill="x", pady=(0, 12))
        self.progress_labels: dict[str, tk.Label] = {}
        for key, label in (("inputs_header", "Inputs"), ("masks", "Masks"), ("field_map", "Field map"), ("noise_weights", "Noise & weights"), ("bgfr", "BGFR"), ("dipole_inversion", "Inversion")):
            item = tk.Label(self.progress_frame, text=label, background="#d9dde3", foreground="#30343b", padx=12, pady=7)
            item.pack(side="left", fill="x", expand=True, padx=1)
            self.progress_labels[key] = item
        self._update_progress_bar()
        self.tabs = ttk.Notebook(outer)
        self.tabs.pack(fill="both", expand=True)
        pages = [("Overview", self._overview), ("1  Inputs & header", self._inputs), ("2  Masks", self._masks), ("3  Field map", self._fieldmap), ("4  Noise & weights", self._placeholder), ("5  BGFR", self._placeholder), ("6  Dipole inversion", self._placeholder), ("Settings", self._settings)]
        for title, builder in pages:
            page = ttk.Frame(self.tabs, padding=18)
            self.tabs.add(page, text=title)
            builder(page, title)
        self.command_log_frame = ttk.LabelFrame(outer, text="Developer subprocess log", padding=6)
        self.command_log = tk.Text(self.command_log_frame, height=7, wrap="word", state="disabled", font=("Consolas", 9))
        self.command_log.pack(fill="x", expand=True)
        if self.developer_mode.get():
            self.command_log_frame.pack(fill="x", pady=(10, 0))

    def _overview(self, parent: ttk.Frame, _title: str) -> None:
        ttk.Label(parent, text="Pipeline overview", style="Section.TLabel").pack(anchor="w")
        ttk.Label(parent, text="Start with your meGRE acquisition, then move through the tabs from left to right. Each generated file is prefixed with the participant ID.", wraplength=850).pack(anchor="w", pady=(5, 18))
        stages = ["✓ Select magnitude and phase data; enter acquisition metadata; create the SEPIA header", "→ Create or select masks and inspect them", "Calculate the total field with ROMEO", "Calculate noise SD and inversion weights", "Run background-field removal", "Run dipole inversion", "Inspect outputs and record QC"]
        for number, stage in enumerate(stages, 1):
            ttk.Label(parent, text=f"{number}.  {stage}", padding=(8, 9)).pack(fill="x", anchor="w")
        ttk.Button(parent, text="Start with participant inputs →", command=lambda: self.tabs.select(1)).pack(anchor="w", pady=(20, 0))
        self.resume_status = ttk.Label(parent, text="No project loaded. Start a new participant or use Load project above.", foreground="#555", wraplength=850, justify="left")
        self.resume_status.pack(anchor="w", pady=(18, 0))

    def _set(self, name: str, default: str = "") -> tk.StringVar:
        variable = tk.StringVar(value=default)
        self.values[name] = variable
        return variable

    def _entry_row(self, parent: ttk.Frame, row: int, label: str, name: str, default: str = "", hint: str = "", browse: str | None = None) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=6)
        ttk.Entry(parent, textvariable=self._set(name, default)).grid(row=row, column=1, sticky="ew", pady=6)
        if browse:
            ttk.Button(parent, text="Browse…", command=lambda: self._browse(name, browse)).grid(row=row, column=2, padx=(8, 0), pady=6)
        if hint:
            ttk.Label(parent, text=hint, foreground="#555").grid(row=row, column=3, sticky="w", padx=(10, 0))

    def _inputs(self, parent: ttk.Frame, _title: str) -> None:
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text="Participant and meGRE acquisition", style="Section.TLabel").grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Label(parent, text="Echo times are entered in milliseconds and converted to seconds in the SEPIA header. Central frequency is entered in Hz.", wraplength=850).grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 14))
        rows = [
            ("Participant ID", "participant_id", "", "Example: sub-001", None),
            ("Magnitude NIfTI", "magnitude_path", "", "3D or multi-echo 4D", "file"),
            ("Phase NIfTI", "phase_path", "", "3D or multi-echo 4D", "file"),
            ("Echo times (ms)", "echo_times", "", "Example: 4.55, 8.81, 13.07", None),
            ("B0 strength (T)", "b0", "3", "Example: 3 or 7", None),
            ("B0 direction", "b0_direction", "0, 0, 1", "Three scanner-axis components", None),
            ("Central frequency (Hz)", "central_frequency", "", "Example: 123250000", None),
            ("Matrix size", "matrix_size", "", "Three integers: X, Y, Z", None),
            ("Voxel size (mm)", "voxel_size", "", "Three values: X, Y, Z", None),
            ("Output root", "output_directory", "", "A participant folder is created here", "directory"),
        ]
        for row, values in enumerate(rows, 2):
            self._entry_row(parent, row, *values)
        ttk.Button(parent, text="Validate and create SEPIA headers (.json + .mat)", command=self._save_acquisition).grid(row=12, column=0, columnspan=2, sticky="w", pady=(18, 8))
        self.input_status = ttk.Label(parent, text="No header created yet.", foreground="#555", wraplength=850)
        self.input_status.grid(row=13, column=0, columnspan=4, sticky="w")

    def _masks(self, parent: ttk.Frame, _title: str) -> None:
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text="Mask creation and vertebral levels", style="Section.TLabel").grid(row=0, column=0, columnspan=4, sticky="w")
        explanation = (
            "The meGRE spinal-cord (SC) and gray-matter (GM) masks are created from the first magnitude echo. "
            "The white-matter (WM) mask is SC minus GM. The T1 is segmented separately; disc labels are placed "
            "manually in the SCT viewer and then used to calculate vertebral levels. All outputs go into the "
            "participant's masking folder. Every mask and both vertebral-level segmentations must be opened, "
            "inspected manually, and corrected when necessary before approval."
        )
        ttk.Label(parent, text=explanation, wraplength=900).grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 14))
        self._entry_row(parent, 2, "Magnitude NIfTI", "mask_magnitude_path", "", "Defaults to the acquisition magnitude", "file")
        self._entry_row(parent, 3, "T1-weighted NIfTI", "mask_t1_path", "", "Required for vertebral levels", "file")

        range_frame = ttk.Frame(parent)
        range_frame.grid(row=4, column=0, columnspan=4, sticky="w", pady=(4, 8))
        ttk.Label(range_frame, text="Disc labels from").pack(side="left")
        ttk.Entry(range_frame, textvariable=self._set("disc_first", "1"), width=6).pack(side="left", padx=6)
        ttk.Label(range_frame, text="to").pack(side="left")
        ttk.Entry(range_frame, textvariable=self._set("disc_last", "10"), width=6).pack(side="left", padx=6)
        self.force_mask_rerun = tk.BooleanVar(value=False)
        ttk.Checkbutton(range_frame, text="Force rerun selected step", variable=self.force_mask_rerun).pack(side="left", padx=(24, 0))

        actions = ttk.LabelFrame(parent, text="Run and check each step", padding=12)
        actions.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(8, 8))
        action_specs = [
            ("1. Create meGRE SC, GM, and WM masks", self._start_megre_masks),
            ("2. Create T1 spinal-cord mask", self._start_t1_mask),
            ("3. Open viewer and label selected disc range", self._start_disc_labels),
            ("4. Create T1-space vertebral-level segmentation", self._start_vertebral_levels),
        ]
        self.mask_buttons = []
        for row, (text, command) in enumerate(action_specs):
            button = ttk.Button(actions, text=text, command=command)
            button.grid(row=row, column=0, sticky="w", pady=4)
            self.mask_buttons.append(button)
        self.mask_qc_confirmed = tk.BooleanVar(value=False)
        ttk.Checkbutton(actions, text="I manually inspected/corrected the SC, GM, WM, T1 SC, and T1-space vertebral labels", variable=self.mask_qc_confirmed).grid(row=4, column=0, sticky="w", pady=(12, 3))
        qc_button = ttk.Button(actions, text="5. Record mask meGRE and T1w QC approval", command=self._approve_mask_qc)
        qc_button.grid(row=5, column=0, sticky="w", pady=4)
        self.mask_buttons.append(qc_button)
        registration_button = ttk.Button(actions, text="6. Register meGRE to T1w and warp vertebral levels to meGRE", command=self._start_registration)
        registration_button.grid(row=6, column=0, sticky="w", pady=4)
        self.mask_buttons.append(registration_button)
        self.registration_qc_confirmed = tk.BooleanVar(value=False)
        ttk.Checkbutton(actions, text="I manually inspected/corrected the warped meGRE-space vertebral labels", variable=self.registration_qc_confirmed).grid(row=7, column=0, sticky="w", pady=(12, 3))
        registration_qc_button = ttk.Button(actions, text="7. Record registration QC approval", command=self._approve_registration_qc)
        registration_qc_button.grid(row=8, column=0, sticky="w", pady=4)
        self.mask_buttons.append(registration_qc_button)
        ttk.Label(actions, text="Existing non-empty outputs are reused unless Force rerun is selected. Any rerun invalidates the relevant QC approval and downstream registration.", foreground="#555", wraplength=760).grid(row=9, column=0, sticky="w", pady=(10, 0))
        self.mask_status = ttk.Label(parent, text="Complete the Inputs & header tab before running masking.", foreground="#555", wraplength=900)
        self.mask_status.grid(row=6, column=0, columnspan=4, sticky="w", pady=(8, 0))

    def _fieldmap(self, parent: ttk.Frame, _title: str) -> None:
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text="Total-field mapping with ROMEO", style="Section.TLabel").grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Label(parent, text="ROMEO uses the original 4D magnitude and phase, the approved meGRE SC mask, and echo times converted from the SEPIA header into milliseconds. Both runs use -k; the masked variant additionally uses -u.", wraplength=900).grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 14))
        self._entry_row(parent, 2, "4D magnitude NIfTI", "field_magnitude_path", "", browse="file")
        self._entry_row(parent, 3, "4D phase NIfTI", "field_phase_path", "", browse="file")
        self._entry_row(parent, 4, "meGRE SC mask", "field_mask_path", "", browse="file")
        self._entry_row(parent, 5, "Echo times (milliseconds)", "field_echo_times", "", "SEPIA seconds × 1000")
        ttk.Label(parent, text="Phase-offset correction").grid(row=6, column=0, sticky="w", padx=(0, 12), pady=6)
        self.field_phase_correction = tk.StringVar(value="bipolar")
        ttk.Combobox(parent, textvariable=self.field_phase_correction, values=("bipolar", "on", "off"), state="readonly", width=18).grid(row=6, column=1, sticky="w", pady=6)
        display_frame = ttk.Frame(parent)
        display_frame.grid(row=7, column=0, columnspan=4, sticky="w", pady=6)
        ttk.Label(display_frame, text="B0 PNG: cmin").pack(side="left")
        ttk.Entry(display_frame, textvariable=self._set("field_cmin", "-3"), width=8).pack(side="left", padx=5)
        ttk.Label(display_frame, text="cmax").pack(side="left")
        ttk.Entry(display_frame, textvariable=self._set("field_cmax", "3"), width=8).pack(side="left", padx=5)
        ttk.Label(display_frame, text="colorbar").pack(side="left")
        ttk.Entry(display_frame, textvariable=self._set("field_cbar", "bwr"), width=12).pack(side="left", padx=5)
        self.force_fieldmap_rerun = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text="Force rerun fieldmap or PNG output", variable=self.force_fieldmap_rerun).grid(row=8, column=0, columnspan=2, sticky="w", pady=(8, 8))
        button_row = ttk.Frame(parent)
        button_row.grid(row=9, column=0, columnspan=4, sticky="w", pady=6)
        self.fieldmap_buttons = [
            ttk.Button(button_row, text="Run masked fieldmap (-k, -u)", command=lambda: self._start_fieldmap(True)),
            ttk.Button(button_row, text="Run unmasked fieldmap (-k)", command=lambda: self._start_fieldmap(False)),
        ]
        for index, button in enumerate(self.fieldmap_buttons):
            button.pack(side="left", padx=(0 if index == 0 else 8, 0))
        png_button = ttk.Button(parent, text="Create masked and unmasked B0 QC PNGs", command=self._start_fieldmap_pngs)
        png_button.grid(row=10, column=0, columnspan=2, sticky="w", pady=(10, 4))
        self.fieldmap_buttons.append(png_button)
        self.fieldmap_qc_confirmed = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text="I manually inspected the B0 fieldmaps and QC PNGs and they are ready for BGFR/DI", variable=self.fieldmap_qc_confirmed).grid(row=11, column=0, columnspan=4, sticky="w", pady=(10, 4))
        qc_button = ttk.Button(parent, text="Record fieldmap QC approval", command=self._approve_fieldmap_qc)
        qc_button.grid(row=12, column=0, columnspan=2, sticky="w", pady=4)
        self.fieldmap_buttons.append(qc_button)
        self.fieldmap_status = ttk.Label(parent, text="Load or create a participant project first.", foreground="#555", wraplength=900)
        self.fieldmap_status.grid(row=13, column=0, columnspan=4, sticky="w", pady=(12, 0))

    def _placeholder(self, parent: ttk.Frame, title: str) -> None:
        ttk.Label(parent, text=title, style="Section.TLabel").pack(anchor="w")
        ttk.Label(parent, text="This stage will be connected after the preceding stage is implemented and checked.").pack(anchor="w", pady=(5, 0))

    def _settings(self, parent: ttk.Frame, _title: str) -> None:
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text="Tool locations", style="Section.TLabel").grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(parent, text="Julia and Spinal Cord Toolbox are expected on PATH. Only script/toolbox directories are stored here.").grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 14))
        self._entry_row(parent, 2, "ROMEO script (romeo.jl)", "romeo_script", self.tool_config.romeo_script, browse="file")
        self._entry_row(parent, 3, "SEPIA directory", "sepia_directory", self.tool_config.sepia_directory, browse="directory")
        ttk.Button(parent, text="Save settings", command=self._save_settings).grid(row=4, column=0, sticky="w", pady=(14, 0))
        ttk.Button(parent, text="Validate tools", command=self._validate_tools).grid(row=4, column=1, sticky="w", pady=(14, 0))
        self.tool_status = ttk.Label(parent, text=f"Configuration: {default_config_path()}", wraplength=850)
        self.tool_status.grid(row=5, column=0, columnspan=3, sticky="w", pady=(14, 0))
        ttk.Separator(parent).grid(row=6, column=0, columnspan=3, sticky="ew", pady=18)
        ttk.Label(parent, text="Python and MATLAB Engine setup", style="Section.TLabel").grid(row=7, column=0, columnspan=3, sticky="w")
        ttk.Label(parent, text="The GUI cannot install packages into the Python process that is currently running. Use the commands below in a terminal, then restart the GUI from qsm_gui.", wraplength=850).grid(row=8, column=0, columnspan=3, sticky="w", pady=(4, 8))
        ttk.Button(parent, text="Check current environment", command=self._check_environment).grid(row=9, column=0, sticky="w")
        self.environment_status = ttk.Label(parent, text=self._environment_summary(), wraplength=900, justify="left")
        self.environment_status.grid(row=10, column=0, columnspan=3, sticky="w", pady=(8, 8))
        setup = self._setup_commands()
        setup_box = tk.Text(parent, height=9, wrap="none")
        setup_box.grid(row=11, column=0, columnspan=3, sticky="ew")
        setup_box.insert("1.0", setup)
        setup_box.configure(state="disabled")
        ttk.Checkbutton(parent, text="Developer mode: show exact subprocess commands", variable=self.developer_mode, command=self._toggle_developer_mode).grid(row=12, column=0, columnspan=3, sticky="w", pady=(14, 0))

    def _browse(self, name: str, kind: str) -> None:
        selected = filedialog.askdirectory() if kind == "directory" else filedialog.askopenfilename(filetypes=[("NIfTI or scripts", "*.nii *.nii.gz *.jl *.exe"), ("All files", "*.*")])
        if selected:
            self.values[name].set(selected)

    def _acquisition(self) -> Acquisition:
        return Acquisition(
            participant_id=self.values["participant_id"].get().strip(),
            magnitude_path=self.values["magnitude_path"].get().strip(),
            phase_path=self.values["phase_path"].get().strip(),
            output_directory=self.values["output_directory"].get().strip(),
            echo_times_ms=parse_numbers(self.values["echo_times"].get()),
            b0_tesla=float(self.values["b0"].get()),
            b0_direction=parse_numbers(self.values["b0_direction"].get(), 3),
            central_frequency_hz=float(self.values["central_frequency"].get()),
            matrix_size=parse_numbers(self.values["matrix_size"].get(), 3, integer=True),
            voxel_size_mm=parse_numbers(self.values["voxel_size"].get(), 3),
        )

    def _save_acquisition(self) -> None:
        try:
            header, project = self._acquisition().save()
        except (ValueError, OSError) as exc:
            messagebox.showerror("Invalid acquisition", str(exc))
            return
        self.input_status.configure(
            text="JSON metadata created. Starting MATLAB to create the MAT header…",
            foreground="#7a5700",
        )
        self.update_idletasks()
        try:
            matlab_header = create_matlab_header(header, project)
        except MatlabHeaderError as exc:
            self.input_status.configure(
                text=f"Incomplete: JSON metadata exists, but the MAT header was not created.\n{exc}",
                foreground="#9b1c1c",
            )
            messagebox.showerror("MATLAB header creation failed", str(exc))
            return
        self.input_status.configure(
            text=f"Created:\n{header}\n{matlab_header}\n{project}",
            foreground="#176b2c",
        )
        self.current_project_path = project
        self._refresh_current_project()
        messagebox.showinfo(
            "Participant initialized",
            "The JSON metadata, MATLAB SEPIA header, and participant project file were created.",
        )

    def _save_settings(self) -> None:
        self.tool_config = ToolConfig(romeo_script=self.values["romeo_script"].get().strip(), sepia_directory=self.values["sepia_directory"].get().strip(), developer_mode=self.developer_mode.get())
        path = self.tool_config.save()
        self.tool_status.configure(text=f"Saved: {path}")

    def _validate_tools(self) -> None:
        config = ToolConfig(romeo_script=self.values["romeo_script"].get().strip(), sepia_directory=self.values["sepia_directory"].get().strip(), developer_mode=self.developer_mode.get())
        checks = validate_tools(config)
        self.tool_status.configure(text="\n".join(f"{'✓' if check.ok else '✗'} {check.name}: {check.detail}" for check in checks))

    def _masking_inputs(self) -> MaskingInputs:
        magnitude = self.values["mask_magnitude_path"].get().strip() or self.values["magnitude_path"].get().strip()
        t1_value = self.values["mask_t1_path"].get().strip()
        output_root = self.values["output_directory"].get().strip()
        if not output_root:
            raise MaskingError("Select an output root on the Inputs & header tab first.")
        return MaskingInputs(
            participant_id=self.values["participant_id"].get().strip(),
            magnitude_path=Path(magnitude),
            t1_path=Path(t1_value) if t1_value else None,
            output_root=Path(output_root),
        )

    def _run_mask_task(self, description: str, milestone: str, operation) -> None:
        try:
            inputs = self._masking_inputs()
        except MaskingError as exc:
            self._finish_mask_task(False, str(exc))
            return
        for button in self.mask_buttons:
            button.configure(state="disabled")
        self.mask_status.configure(text=f"Running: {description}…", foreground="#7a5700")

        def worker() -> None:
            try:
                outputs = operation(inputs)
                update_project_masking(inputs, outputs, milestone)
            except (MaskingError, OSError, ValueError) as exc:
                self.after(0, lambda: self._finish_mask_task(False, str(exc)))
                return
            self.after(0, lambda: self._finish_mask_task(True, "\n".join(str(path) for path in outputs.values())))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_mask_task(self, successful: bool, detail: str) -> None:
        for button in self.mask_buttons:
            button.configure(state="normal")
        if successful:
            self.mask_status.configure(text=f"Completed:\n{detail}", foreground="#176b2c")
            self._refresh_current_project()
        else:
            self.mask_status.configure(text=f"Masking step failed:\n{detail}", foreground="#9b1c1c")
            messagebox.showerror("Masking step failed", detail)

    def _start_megre_masks(self) -> None:
        force = self.force_mask_rerun.get()
        self._run_mask_task("meGRE SC, GM, and WM masks", "megre_masks", lambda inputs: create_megre_masks(inputs, runner=self._gui_runner, force=force))

    def _start_t1_mask(self) -> None:
        force = self.force_mask_rerun.get()
        self._run_mask_task("T1 spinal-cord mask", "t1_sc_mask", lambda inputs: {"t1_sc_mask": create_t1_sc_mask(inputs, runner=self._gui_runner, force=force)})

    def _start_disc_labels(self) -> None:
        try:
            first_label = int(self.values["disc_first"].get())
            last_label = int(self.values["disc_last"].get())
        except ValueError:
            self._finish_mask_task(False, "Disc label start and end must be integers.")
            return
        force = self.force_mask_rerun.get()
        self._run_mask_task("manual disc labels", "disc_labels", lambda inputs: {"disc_labels": create_disc_labels(inputs, first_label, last_label, runner=self._gui_runner, force=force)})

    def _start_vertebral_levels(self) -> None:
        force = self.force_mask_rerun.get()
        self._run_mask_task("vertebral-level segmentation", "vertebral_levels", lambda inputs: {"vertebral_levels": create_vertebral_levels(inputs, runner=self._gui_runner, force=force)})

    def _approve_mask_qc(self) -> None:
        if not self.mask_qc_confirmed.get():
            messagebox.showerror("Manual QC required", "Open and inspect every T1-space mask and vertebral-level file, correct them if needed, then select the confirmation checkbox.")
            return
        self._refresh_current_project()
        required = ("megre_masks", "t1_sc_mask", "disc_labels", "vertebral_levels")
        if not self.current_project_path or not all(milestone_complete(self.current_project, name) for name in required):
            messagebox.showerror("Masking incomplete", "All mask and T1-space vertebral-level outputs must exist before QC can be approved.")
            return
        set_project_milestone(self.current_project_path, "mask_qc")
        self._refresh_current_project()
        self.mask_status.configure(text="meGRE and T1w mask QC approved. Registration is the next step.", foreground="#176b2c")

    def _start_registration(self) -> None:
        self._refresh_current_project()
        if not milestone_complete(self.current_project, "mask_qc"):
            messagebox.showerror("Manual QC required", "Approve the T1-space masking QC milestone before registration.")
            return
        force = self.force_mask_rerun.get()
        self._run_mask_task("meGRE-to-T1w registration and label warping", "registration", lambda inputs: register_megre_to_t1w(inputs, runner=self._gui_runner, force=force))

    def _approve_registration_qc(self) -> None:
        if not self.registration_qc_confirmed.get():
            messagebox.showerror("Manual QC required", "Open the meGRE-space vertebral-level segmentation, correct it if needed, then select the confirmation checkbox.")
            return
        self._refresh_current_project()
        if not self.current_project_path or not milestone_complete(self.current_project, "registration"):
            messagebox.showerror("Registration incomplete", "Complete registration and label warping before approving registration QC.")
            return
        set_project_milestone(self.current_project_path, "registration_qc")
        self._refresh_current_project()
        self.mask_status.configure(text="Registration QC approved. The masking stage is complete.", foreground="#176b2c")

    def _fieldmap_inputs(self) -> FieldmapInputs:
        output_root = self.values["output_directory"].get().strip()
        if not output_root:
            raise MaskingError("Load or create a participant project first.")
        return FieldmapInputs(
            participant_id=self.values["participant_id"].get().strip(),
            magnitude_path=Path(self.values["field_magnitude_path"].get().strip()),
            phase_path=Path(self.values["field_phase_path"].get().strip()),
            mask_path=Path(self.values["field_mask_path"].get().strip()),
            echo_times_ms=parse_numbers(self.values["field_echo_times"].get()),
            output_root=Path(output_root),
            phase_offset_correction=self.field_phase_correction.get(),
        )

    def _start_fieldmap(self, masked: bool) -> None:
        self._refresh_current_project()
        if not milestone_complete(self.current_project, "registration_qc"):
            messagebox.showerror("Masking QC required", "Complete and approve the final meGRE-space registration QC before field mapping.")
            return
        try:
            inputs = self._fieldmap_inputs()
        except (MaskingError, ValueError) as exc:
            messagebox.showerror("Invalid fieldmap inputs", str(exc))
            return
        config = ToolConfig(
            romeo_script=self.values["romeo_script"].get().strip(),
            sepia_directory=self.values["sepia_directory"].get().strip(),
            developer_mode=self.developer_mode.get(),
        )
        for button in self.fieldmap_buttons:
            button.configure(state="disabled")
        variant = "masked" if masked else "unmasked"
        self.fieldmap_status.configure(text=f"Running {variant} ROMEO fieldmap…", foreground="#7a5700")
        force = self.force_fieldmap_rerun.get()

        def worker() -> None:
            try:
                outputs = run_fieldmap(inputs, config, masked, runner=self._gui_runner, force=force)
                update_project_fieldmap(inputs, masked_output=outputs if masked else None, unmasked_output=outputs if not masked else None)
            except (MaskingError, OSError, ValueError) as exc:
                self.after(0, lambda: self._finish_fieldmap(False, str(exc)))
                return
            self.after(0, lambda: self._finish_fieldmap(True, "\n".join(str(path) for path in outputs.values())))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_fieldmap(self, successful: bool, detail: str) -> None:
        for button in self.fieldmap_buttons:
            button.configure(state="normal")
        if successful:
            self.fieldmap_status.configure(text=f"Completed:\n{detail}", foreground="#176b2c")
            self._refresh_current_project()
        else:
            self.fieldmap_status.configure(text=f"ROMEO failed:\n{detail}", foreground="#9b1c1c")
            messagebox.showerror("ROMEO fieldmap failed", detail)

    def _start_fieldmap_pngs(self) -> None:
        self._refresh_current_project()
        if not milestone_complete(self.current_project, "field_map"):
            messagebox.showerror("Fieldmaps incomplete", "Create both masked and unmasked fieldmaps before generating QC PNGs.")
            return
        try:
            inputs = self._fieldmap_inputs()
            cmin = float(self.values["field_cmin"].get())
            cmax = float(self.values["field_cmax"].get())
            cbar = self.values["field_cbar"].get().strip()
        except (MaskingError, ValueError) as exc:
            messagebox.showerror("Invalid PNG settings", str(exc))
            return
        for button in self.fieldmap_buttons:
            button.configure(state="disabled")
        self.fieldmap_status.configure(text="Creating masked and unmasked B0 QC PNGs…", foreground="#7a5700")
        force = self.force_fieldmap_rerun.get()

        def worker() -> None:
            try:
                outputs = create_fieldmap_pngs(inputs, cmin, cmax, cbar, runner=self._gui_runner, force=force)
            except (MaskingError, OSError, ValueError) as exc:
                self.after(0, lambda: self._finish_fieldmap(False, str(exc)))
                return
            self.after(0, lambda: self._finish_fieldmap(True, "\n".join(str(path) for path in outputs.values())))

        threading.Thread(target=worker, daemon=True).start()

    def _approve_fieldmap_qc(self) -> None:
        if not self.fieldmap_qc_confirmed.get():
            messagebox.showerror("Manual QC required", "Inspect both B0 fieldmaps and their PNGs, then select the confirmation checkbox.")
            return
        self._refresh_current_project()
        if not self.current_project_path or not milestone_complete(self.current_project, "fieldmap_visualization"):
            messagebox.showerror("QC PNGs incomplete", "Create both B0 QC PNGs before approving the fieldmap.")
            return
        set_project_milestone(self.current_project_path, "fieldmap_qc")
        self._refresh_current_project()
        self.fieldmap_status.configure(text="Fieldmap QC approved. Inputs are ready for BGFR and DI.", foreground="#176b2c")

    def _format_command(self, command: list[str]) -> str:
        return subprocess.list2cmdline(command) if platform.system() == "Windows" else shlex.join(command)

    def _append_command_log(self, text: str) -> None:
        if not self.developer_enabled:
            return
        self.command_log.configure(state="normal")
        self.command_log.insert("end", text + "\n")
        self.command_log.see("end")
        self.command_log.configure(state="disabled")

    def _gui_runner(self, command: list[str], interactive: bool = False) -> None:
        if self.developer_enabled:
            self.after(0, lambda command=command: self._append_command_log("$ " + self._format_command(command)))
        try:
            run_command(command, interactive)
        except Exception as exc:
            if self.developer_enabled:
                self.after(0, lambda exc=exc: self._append_command_log("ERROR: " + str(exc)))
            raise

    def _toggle_developer_mode(self) -> None:
        self.developer_enabled = self.developer_mode.get()
        if self.developer_enabled:
            self.command_log_frame.pack(fill="x", pady=(10, 0))
        else:
            self.command_log_frame.pack_forget()
        self._save_settings()

    def _load_project_dialog(self) -> None:
        selected = filedialog.askopenfilename(
            title="Load QSM project",
            filetypes=[("QSM project JSON", "*_qsm_project.json"), ("JSON files", "*.json")],
        )
        if not selected:
            return
        self.current_project_path = Path(selected)
        try:
            self._load_current_project_into_gui()
        except ProjectError as exc:
            messagebox.showerror("Could not load project", str(exc))

    def _refresh_current_project(self) -> None:
        if self.current_project_path is None:
            participant = self.values.get("participant_id")
            output = self.values.get("output_directory")
            if not participant or not output or not participant.get().strip() or not output.get().strip():
                return
            candidate = Path(output.get().strip()) / participant.get().strip() / f"{participant.get().strip()}_qsm_project.json"
            if not candidate.is_file():
                return
            self.current_project_path = candidate
        try:
            self._load_current_project_into_gui(show_message=False)
        except ProjectError as exc:
            self.resume_status.configure(text=f"Project needs attention: {exc}", foreground="#9b1c1c")

    @staticmethod
    def _numbers_text(values) -> str:
        return ", ".join(f"{float(value):g}" for value in values)

    def _load_current_project_into_gui(self, show_message: bool = True) -> None:
        if self.current_project_path is None:
            return
        project, header = load_project(self.current_project_path)
        self.current_project = project
        participant_directory = Path(project["output_directory"])
        assignments = {
            "participant_id": project["participant_id"],
            "magnitude_path": project["magnitude_path"],
            "phase_path": project["phase_path"],
            "echo_times": self._numbers_text(float(value) * 1000 for value in header["TE"]),
            "b0": f"{float(header['B0']):g}",
            "b0_direction": self._numbers_text(header["B0_dir"]),
            "central_frequency": f"{float(header['CF']):g}",
            "matrix_size": self._numbers_text(header["matrix_size"]),
            "voxel_size": self._numbers_text(header["voxel_size"]),
            "output_directory": str(participant_directory.parent),
            "mask_magnitude_path": project["magnitude_path"],
            "mask_t1_path": project.get("masking", {}).get("t1_path", ""),
            "field_magnitude_path": project["magnitude_path"],
            "field_phase_path": project["phase_path"],
            "field_mask_path": project.get("masking", {}).get("sc_mask", ""),
            "field_echo_times": self._numbers_text(float(value) * 1000 for value in header["TE"]),
        }
        for name, value in assignments.items():
            self.values[name].set(str(value))
        self.mask_qc_confirmed.set(milestone_complete(project, "mask_qc"))
        self.registration_qc_confirmed.set(milestone_complete(project, "registration_qc"))
        self.fieldmap_qc_confirmed.set(milestone_complete(project, "fieldmap_qc"))
        self.field_phase_correction.set(project.get("fieldmap", {}).get("phase_offset_correction", "bipolar"))
        qc_settings = project.get("fieldmap_qc_settings", {})
        self.values["field_cmin"].set(str(qc_settings.get("cmin", -3)))
        self.values["field_cmax"].set(str(qc_settings.get("cmax", 3)))
        self.values["field_cbar"].set(str(qc_settings.get("cbar", "bwr")))
        summary = milestone_summary(project)
        self.resume_status.configure(text=f"Loaded {project['participant_id']}\n{summary}", foreground="#176b2c")
        headers_valid = milestone_complete(project, "inputs_header")
        self.input_status.configure(
            text=("Loaded and validated the saved SEPIA JSON and MATLAB headers." if headers_valid else "Project loaded, but the saved SEPIA headers need to be recreated or repaired."),
            foreground="#176b2c" if headers_valid else "#9b1c1c",
        )
        self.mask_status.configure(text=summary, foreground="#176b2c")
        self.fieldmap_status.configure(text=summary, foreground="#176b2c")
        self._update_progress_bar()
        if show_message:
            messagebox.showinfo("Project loaded", summary)

    def _update_progress_bar(self) -> None:
        project = self.current_project
        groups = [
            ("inputs_header", milestone_complete(project, "inputs_header")),
            ("masks", all(milestone_complete(project, name) for name in ("megre_masks", "t1_sc_mask", "disc_labels", "vertebral_levels", "mask_qc", "registration", "registration_qc"))),
            ("field_map", milestone_complete(project, "fieldmap_qc")),
            ("noise_weights", milestone_complete(project, "noise_weights")),
            ("bgfr", milestone_complete(project, "bgfr")),
            ("dipole_inversion", milestone_complete(project, "dipole_inversion")),
        ]
        first_incomplete = next((key for key, complete in groups if not complete), None)
        for key, complete in groups:
            if complete:
                background, foreground, suffix = "#2e8b57", "white", " ✓"
            elif key == first_incomplete:
                background, foreground, suffix = "#3578b8", "white", " →"
            else:
                background, foreground, suffix = "#d9dde3", "#30343b", ""
            base = self.progress_labels[key].cget("text").replace(" ✓", "").replace(" →", "")
            self.progress_labels[key].configure(text=base + suffix, background=background, foreground=foreground)

    def _environment_summary(self) -> str:
        try:
            import matlab.engine  # type: ignore[import-not-found]  # noqa: F401
        except (ImportError, OSError) as exc:
            matlab_state = f"unavailable ({exc})"
        else:
            matlab_state = "importable"
        return (
            f"Operating system: {platform.system()}\n"
            f"Python: {sys.executable}\n"
            f"Conda environment: {os.environ.get('CONDA_DEFAULT_ENV', '(not detected)')}\n"
            f"MATLAB Engine: {matlab_state}"
        )

    def _check_environment(self) -> None:
        self.environment_status.configure(text=self._environment_summary())

    def _setup_commands(self) -> str:
        if platform.system() == "Darwin":
            engine_path = "/Applications/MATLAB_R2024b.app/extern/engines/python"
            activate = "source $(conda info --base)/etc/profile.d/conda.sh"
        else:
            engine_path = r"C:\Program Files\MATLAB\R2024b\extern\engines\python"
            activate = ""
        lines = [
            "conda env create -f qsm_pp_gui/environment.yml",
            activate,
            "conda activate qsm_gui",
            f'python -m pip install "{engine_path}"',
            'python -c "import matlab.engine; eng=matlab.engine.start_matlab(); print(eng.version()); eng.quit()"',
        ]
        return "\n".join(line for line in lines if line)


def main() -> None:
    QsmApp().mainloop()
