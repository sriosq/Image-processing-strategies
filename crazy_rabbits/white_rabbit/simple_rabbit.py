import tkinter as tk
import time

class SimpleClock:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple Clock")

        self.running = False
        self.start_time = None
        self.elapsed_time = 0

        # Label to show time
        self.label = tk.Label(root, text="00:00:00", font=("Helvetica", 40))
        self.label.pack(pady=20)

        # Start button
        self.start_button = tk.Button(root, text="Start", command=self.start)
        self.start_button.pack(side="left", padx=10)

        # Stop button
        self.stop_button = tk.Button(root, text="Stop", command=self.stop)
        self.stop_button.pack(side="left", padx=10)

        # Reset button
        self.reset_button = tk.Button(root, text="Reset", command=self.reset)
        self.reset_button.pack(side="left", padx=10)

        self.update_clock()

    def update_clock(self):
        if self.running:
            now = time.time()
            self.elapsed_time = now - self.start_time
            formatted_time = time.strftime("%H:%M:%S", time.gmtime(self.elapsed_time))
            self.label.config(text=formatted_time)
        self.root.after(100, self.update_clock)

    def start(self):
        if not self.running:
            self.running = True
            self.start_time = time.time() - self.elapsed_time

    def stop(self):
        self.running = False

    def reset(self):
        self.running = False
        self.elapsed_time = 0
        self.label.config(text="00:00:00")

if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleClock(root)
    root.mainloop()