import tkinter as tk
from tkinter import filedialog
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class MRIPixelSelector:
    def __init__(self, root):
        self.root = root
        self.root.title("MRI Pixel Selector")

        # Buttons
        self.load_button = tk.Button(root, text="Load NIfTI File", command=self.load_nifti)
        self.load_button.pack()

        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack()

        self.image_data = None
        self.fig, self.ax = plt.subplots()

    def load_nifti(self):
        file_path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii *.nii.gz")])
        if not file_path:
            return

        # Load NIfTI file
        nifti_img = nib.load(file_path)
        self.image_data = nifti_img.get_fdata()

        # Display the middle axial slice
        slice_index = self.image_data.shape[2] // 2
        self.display_slice(slice_index)

    def display_slice(self, slice_index):
        self.ax.clear()

        if self.image_data is not None:
            self.ax.imshow(self.image_data[:, :, slice_index], cmap="gray")
            self.ax.set_title(f"Axial Slice {slice_index}")
            self.ax.set_xlabel("Click on a pixel to select")

            # Connect click event
            self.fig.canvas.mpl_connect("button_press_event", self.on_click)

            # Draw the figure on the Tkinter canvas
            canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack()

    def on_click(self, event):
        if event.inaxes:
            x, y = int(event.xdata), int(event.ydata)
            value = self.image_data[x, y, self.image_data.shape[2] // 2]
            print(f"Selected Pixel: ({x}, {y}) - Value: {value}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MRIPixelSelector(root)
    root.mainloop()