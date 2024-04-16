import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, ttk
import pandas as pd

import torch
import torch.nn as nn

class ClassificationModel(nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()
        self.layer1 = nn.Linear(X_train.shape[1], 128)
        self.layer2 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(0.2)
        self.layer3 = nn.Linear(256, 64)
        self.output = nn.Linear(64, 4)  # Output layer for 5 classes

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.dropout(x)
        x = torch.relu(self.layer3(x))
        x = self.output(x)
        return x

model = ClassificationModel()

# Loads model state dictionary
model_state = torch.load('model_weights.model')

# Applies the loaded state dictionary to the model
model.load_state_dict(model_state)

app = ctk.CTk()
app.title("DonorAI")

# Set window size
window_width = 800
window_height = 600
app.geometry(f"{window_width}x{window_height}")

# Set the appearance
ctk.set_appearance_mode("Dark")  # Set to 'Dark' mode

# Create a custom font for titles and labels
title_font = ctk.CTkFont(family="Roboto Medium", size=24)
label_font = ctk.CTkFont(family="Roboto", size=16)

# Function to open the file dialog and display file
def open_file_dialog(event=None):
    file_path = filedialog.askopenfilename(
        title="Select a file",
        filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")],  # Specific order for Windows compatibility
        defaultextension=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")]  # Adding defaultextension as well
    )
    if file_path:
        display_file(file_path)

# Function to display the CSV or XLSX file contents
def display_file(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        return

    for widget in content_frame.winfo_children():
        widget.destroy()

    tree = ttk.Treeview(content_frame)
    tree.pack(expand=True, fill='both')
    tree['columns'] = list(df.columns)
    tree['show'] = 'headings'
    for column in tree['columns']:
        tree.heading(column, text=column)
    for index, row in df.iterrows():
        tree.insert("", tk.END, values=list(row))

# Welcome label
welcome_label = ctk.CTkLabel(app, text="Welcome to DonorAI")
welcome_label.configure(font=title_font)
welcome_label.pack(pady=20)

# Frame for file browsing and display
content_frame = ctk.CTkFrame(app, corner_radius=10)
content_frame.pack(padx=20, pady=20, fill='both', expand=True)

outer_frame = ctk.CTkFrame(content_frame, corner_radius=10, width=700, height=250)
outer_frame.pack(pady=20, expand=True, fill='both')

inner_frame = ctk.CTkFrame(outer_frame, corner_radius=10, fg_color="#334257", border_color="#2C3D50", border_width=4, width=680, height=230)
inner_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
inner_frame.bind("<Button-1>", open_file_dialog)

drag_drop_label = ctk.CTkLabel(inner_frame, text="Drag and drop or browse")
drag_drop_label.configure(font=label_font)
drag_drop_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

app.mainloop()
