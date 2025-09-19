import tkinter as tk
import time


class TaskTimer:

    def __init__(self, parent, name="Task", goal_minutes=60, color="#1E90FF"):

        self.parent = parent
        self.name = name
        self.goal_time = goal_minutes * 60  # convert to seconds
        self.running = False
        self.start_time = None
        self.elapsed_time = 0
        self.color = color # Color for the water

        # Frame for each task
        self.frame = tk.Frame(parent)
        self.frame.pack(pady=5, fill="x")

        # Task name
        self.label_name = tk.Label(self.frame, text=name, font=("Constantia", 14), width=15, anchor="w")
        self.label_name.grid(row=0, column=0, padx=5)

        # Time label
        self.label_time = tk.Label(self.frame, text="00:00:00", font=("Constantia", 14))
        self.label_time.grid(row=0, column=1, padx=5)

        # Barrel canvas
        self.canvas = tk.Canvas(self.frame, width=40, height=100, bg="#E9F5B5", highlightthickness=1, relief="solid")
        self.canvas.grid(row=0, column=2, padx=10)
        self.barrel_fill = self.canvas.create_rectangle(0, 100, 40, 100, fill=self.color)  # initially empty

        # Buttons
        self.start_button = tk.Button(self.frame, text="Start", command=self.start)
        self.start_button.grid(row=0, column=3, padx=2)

        self.stop_button = tk.Button(self.frame, text="Stop", command=self.stop)
        self.stop_button.grid(row=0, column=4, padx=2)

        self.reset_button = tk.Button(self.frame, text="Reset", command=self.reset)
        self.reset_button.grid(row=0, column=5, padx=2)

        # Show goal time info
        self.goal_label = tk.Label(self.frame, text=f"Goal: {goal_minutes} min", font=("Helvetica", 10))
        self.goal_label.grid(row=0, column=6, padx=10)

        self.update_clock()

    def update_clock(self):
        if self.running:
            now = time.time()
            self.elapsed_time = now - self.start_time
            formatted_time = time.strftime("%H:%M:%S", time.gmtime(self.elapsed_time))
            self.label_time.config(text=formatted_time)

            # Update barrel fill
            progress = min(self.elapsed_time / self.goal_time, 1.0)  # 0 to 1
            fill_height = int(100 * (1 - progress))  # invert so fill grows upward
            self.canvas.coords(self.barrel_fill, 0, fill_height, 40, 100)
            self.canvas.itemconfig(self.barrel_fill, fill=self.color)  # ensure color is applied

        self.parent.after(100, self.update_clock)

    def start(self):
        if not self.running:
            self.running = True
            self.start_time = time.time() - self.elapsed_time

    def stop(self):
        self.running = False

    def reset(self):
        self.running = False
        self.elapsed_time = 0
        self.label_time.config(text="00:00:00")
        self.canvas.coords(self.barrel_fill, 0, 100, 40, 100)  # empty again


class MultiClockApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Task Time Tracker with Barrels")

        self.task_frame = tk.Frame(root)
        self.task_frame.pack(pady=10)

        # Controls to add timers
        self.entry_name = tk.Entry(root, width=20)
        self.entry_name.pack(side="left", padx=5)
        self.entry_name.insert(0, "New Task")

        self.entry_goal = tk.Entry(root, width=10)
        self.entry_goal.pack(side="left", padx=5)
        self.entry_goal.insert(0, "1")  # default: 1 min

        self.entry_color = tk.Entry(root, width=10)
        self.entry_color.pack(side="left", padx=5)
        self.entry_color.insert(0, "#1E90FF")  # default blue

        self.add_button = tk.Button(root, text="Add Timer", command=self.add_timer)
        self.add_button.pack(side="left", padx=5)

        self.unit_label = tk.Label(root, text="minutes")
        self.unit_label.pack(side="left")

    def add_timer(self):
        task_name = self.entry_name.get()
        try:
            goal_minutes = int(self.entry_goal.get())
        except ValueError:
            goal_minutes = 60
        color = self.entry_color.get()
        # Basic validation for hex code
        if not color.startswith("#") or len(color) != 7:
            color = "#1E90FF"
        TaskTimer(self.task_frame, task_name, goal_minutes, color)


if __name__ == "__main__":
    root = tk.Tk()
    app = MultiClockApp(root)
    root.mainloop()

