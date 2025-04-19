#So, we start the engine and then define SEPIA with all of its toolboxes:
import PyNomad as nomad
import matlab.engine
import os
import nibabel as nib
import numpy as np
import sys
import json

# Here I join the two functions to create a single function that can be used for both LBV, PDF, VHSARP and the other methods available