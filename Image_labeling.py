import os
from tkinter import *
from PIL import Image, ImageTk

# Path to the directory containing the images
image_directory = "C:\\Users\\Atharva\\Downloads\\VS\\Weed Detection\\train_images\\train_images"

# List to store the image filenames and their corresponding labels
image_labels = []

# Function to handle button click event for labeling
def label_image(label):
    global current_image_index
    image_labels.append((image_filenames[current_image_index], label))
    current_image_index += 1
    if current_image_index < len(image_filenames):
        display_image(image_filenames[current_image_index])
    else:
        root.quit()

# Function to display the next image
def display_image(image_filename):
    image_path = os.path.join(image_directory, image_filename)
    image = Image.open(image_path)
    image = image.resize((400, 400))  # Adjust the size of the displayed image as needed
    image_tk = ImageTk.PhotoImage(image)
    image_label.configure(image=image_tk)
    image_label.image = image_tk

# Get the list of image filenames in the directory
image_filenames = os.listdir(image_directory)

# Initialize variables
current_image_index = 0

# Create a GUI window
root = Tk()
root.title("Image Labeling")

# Create an image label widget
image_label = Label(root)
image_label.pack()

# Create buttons for labeling
weed_button = Button(root, text="Weed", command=lambda: label_image(1))
weed_button.pack(side=LEFT)
non_weed_button = Button(root, text="Non-Weed", command=lambda: label_image(0))
non_weed_button.pack(side=RIGHT)

# Display the first image
display_image(image_filenames[current_image_index])

# Start the GUI event loop
root.mainloop()

# Save the image labels to a CSV file
import pandas as pd
data = pd.DataFrame(image_labels, columns=['image_filename', 'label'])
data.to_csv("labels.csv", index=False)

print("Labels saved successfully!")

