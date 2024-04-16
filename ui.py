import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from customtkinter import CTkFont

app = ctk.CTk()
app.title("DonorAI")

# Set window size
window_width = 800
window_height = 600
app.geometry(f"{window_width}x{window_height}")

# Get the screen dimension
screen_width = app.winfo_screenwidth()
screen_height = app.winfo_screenheight()

# Find the center point
center_x = int(screen_width / 2 - window_width / 2)
center_y = int(screen_height / 2 - window_height / 2)

# Set the position of the window to the center of the screen
app.geometry(f'+{center_x}+{center_y}')

# Function to open the file dialog
def open_file_dialog(event=None):
    file_path = filedialog.askopenfilename(
        title="Select a file",
        filetypes=(("Excel files", "*.xlsx"), ("CSV files", "*.csv"))
    )
    if file_path:
        print("File selected:", file_path)
    else:
        print("No file selected")

# Function to change the drop area color on enter
def on_enter(event):
    inner_frame.configure(fg_color="#2A3D45")  # Darken the drop area on hover

# Function to revert the drop area color on leave
def on_leave(event):
    inner_frame.configure(fg_color="#334257")  # Original drop area color

# Set the appearance
ctk.set_appearance_mode("Dark")  # Set to 'Dark' mode

# Create a custom font
custom_font = CTkFont(family="Roboto Medium", size=24)

# Welcome label
welcome_label = ctk.CTkLabel(app, text="Welcome to DonorAI")
welcome_label.configure(font=custom_font)  # Set the custom font
welcome_label.pack(pady=40)

# Create the staggered border effect using two frames
outer_frame = ctk.CTkFrame(app, corner_radius=10, width=700, height=250)
outer_frame.place(x=50, y=180)

inner_frame = ctk.CTkFrame(outer_frame, corner_radius=10, fg_color="#334257", border_color="#2C3D50", border_width=4, width=680, height=230)
inner_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Bind events to the inner frame for file dialog and hover effect
inner_frame.bind("<Button-1>", open_file_dialog)
inner_frame.bind("<Enter>", on_enter)
inner_frame.bind("<Leave>", on_leave)

# Drag and drop label
drag_drop_label = ctk.CTkLabel(inner_frame, text="Drag and drop or browse")
drag_drop_label.configure(font=CTkFont(family="Roboto", size=16))  # Set the custom font for drag_drop_label
drag_drop_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

app.mainloop()
