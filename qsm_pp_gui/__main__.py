"""Launch the GUI as either a package module or an executable directory."""

if __package__ in {None, ""}:
    # `python qsm_pp_gui` executes this file without package context. Add the
    # repository root so absolute imports work consistently on Windows/macOS.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from qsm_pp_gui.app import main
else:
    from .app import main


if __name__ == "__main__":
    main()
