
import time
import os
import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QProgressBar, QColorDialog, QSpinBox
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QColor, QIcon


class TaskWidget(QWidget):
    def __init__(self, name, goal_minutes, color):
        super().__init__()

        self.name = name
        self.goal_seconds = goal_minutes * 60
        self.elapsed = 0
        self.color = color
        self.running = False

        # Layout
        layout = QHBoxLayout()
        self.label = QLabel(f"{name}: 0:00 / {goal_minutes} min")
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setStyleSheet(
            f"QProgressBar::chunk {{ background-color: {self.color.name()}; }}"
        )

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.toggle_timer)

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timer)

        self.kill_button = QPushButton("Kill")
        self.kill_button.clicked.connect(self.kill_timer)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_timer)

        layout.addWidget(self.label)
        layout.addWidget(self.progress)
        layout.addWidget(self.start_button)
        layout.addWidget(self.kill_button)
        layout.addWidget(self.reset_button)
        self.setLayout(layout)

    def toggle_timer(self):
        if not self.running:
            self.running = True
            self.timer.start(1000)
            self.start_button.setText("Stop")
        else:
            self.running = False
            self.timer.stop()
            self.start_button.setText("Start")

    def update_timer(self):
        self.elapsed += 1
        fill_ratio = min(self.elapsed / self.goal_seconds, 1.0)
        self.progress.setValue(int(fill_ratio * 100))

        minutes = self.elapsed // 60
        seconds = self.elapsed % 60
        self.label.setText(f"{self.name}: {minutes}:{seconds:02d} / {self.goal_seconds // 60} min")

        if self.elapsed >= self.goal_seconds:
            self.timer.stop()
            self.start_button.setText("Done")
            self.start_button.setEnabled(False)

    def kill_timer(self):
        # This closes the app, I need to fix that, maybe adding one widget on top so that there is always 1 widget active?
        self.timer.stop()
        parent = self.parent()
        if parent and isinstance(parent.layout(), QVBoxLayout):
            parent.layout().removeWidget(self)
        self.setParent(None)
        self.deleteLater()

    def reset_timer(self):
        self.timer.stop()
        self.running = False
        self.elapsed = 0
        self.progress.setValue(0)
        self.start_button.setText("Start")
        self.start_button.setEnabled(True)

        self.label.setText(f"{self.name}: 0:00 / {self.goal_seconds // 60} min")


class MultiTaskRabbit(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowIcon(QIcon("nuclear_timer.ico"))

        self.setWindowTitle("Timer Rabbit 🐇")
        self.layout = QVBoxLayout(self)

        # Input for new tasks
        input_layout = QHBoxLayout()
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Task name")

        self.time_input = QSpinBox()
        self.time_input.setRange(1, 720)
        self.time_input.setValue(60)
        self.time_input.setSuffix(" min")

        self.color_button = QPushButton("Pick Color")
        self.color_button.clicked.connect(self.pick_color)
        self.selected_color = QColor("#3498db")

        self.add_button = QPushButton("Add Task")
        self.add_button.clicked.connect(self.add_task)

        input_layout.addWidget(self.name_input)
        input_layout.addWidget(self.time_input)
        input_layout.addWidget(self.color_button)
        input_layout.addWidget(self.add_button)

        # Creating a task container to add and remove tasks safely
        self.tasks_container = QWidget()
        self.tasks_layout = QVBoxLayout()
        self.tasks_container.setLayout(self.tasks_layout)


        # Add both layouts to main layout
        self.layout.addLayout(input_layout)
        self.layout.addWidget(self.tasks_container)

    def pick_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.selected_color = color
            self.color_button.setStyleSheet(f"background-color: {color.name()};")

    def add_task(self):
        name = self.name_input.text().strip() or "Unnamed Task"
        goal_minutes = self.time_input.value()
        color = self.selected_color

        task = TaskWidget(name, goal_minutes, color)
        self.tasks_layout.addWidget(task)

        # reset inputs
        self.name_input.clear()
        self.time_input.setValue(30)

def main():
    app = QApplication(sys.argv)
    window = MultiTaskRabbit()
    window.resize(600, 400)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
