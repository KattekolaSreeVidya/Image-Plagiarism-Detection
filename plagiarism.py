import cv2
import numpy as np
import os
from tkinter import Tk, Label, Button, filedialog, messagebox
from PIL import Image, ImageTk

def get_file_path():
    file_path = filedialog.askopenfilename(title="Select the image to check for plagiarism")
    return file_path

def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (256, 256))  # Resize to a standard size
    return resized_image

def calculate_histogram(image):
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    return cv2.normalize(histogram, histogram).flatten()  # Normalize histogram

def compare_histograms(hist1, hist2):
    # Using Bhattacharyya distance
    distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    similarity_percentage = (1 - distance) * 100
    return similarity_percentage

def compare_with_dataset(uploaded_image_path, dataset_folder, threshold=75.0):
    uploaded_image = cv2.imread(uploaded_image_path)
    if uploaded_image is None:
        return None, 0.0

    uploaded_image = preprocess_image(uploaded_image)
    uploaded_histogram = calculate_histogram(uploaded_image)
    highest_similarity = 0.0

    for root, _, files in os.walk(dataset_folder):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                dataset_image_path = os.path.join(root, filename)
                reference_image = cv2.imread(dataset_image_path)
                if reference_image is None:
                    continue
                
                reference_image = preprocess_image(reference_image)
                reference_histogram = calculate_histogram(reference_image)
                similarity_percentage = compare_histograms(uploaded_histogram, reference_histogram)

                if similarity_percentage > highest_similarity:
                    highest_similarity = similarity_percentage

    if highest_similarity >= threshold:
        return True, highest_similarity
    else:
        return False, highest_similarity

def select_image():
    uploaded_image_path = get_file_path()
    if not uploaded_image_path:
        return

    dataset_folder = r"C:\Users\VIDYA\Desktop\Image plagiarism detection\seaAnimalsdataset"
    is_similar, highest_similarity = compare_with_dataset(uploaded_image_path, dataset_folder)

    if is_similar:
        if highest_similarity >= 90:
            messagebox.showinfo("Result", f"The image is plagiarized! Similarity Percentage: {highest_similarity:.2f}%")
        else:
            messagebox.showinfo("Result", f"The image is similar to at least one image in the dataset.\nSimilarity Percentage: {highest_similarity:.2f}%")
    else:
        messagebox.showinfo("Result", "The image is not plagiarized!")

def main():
    root = Tk()
    root.title("Image Plagiarism Detection")

    Label(root, text="Image Plagiarism Detection System", font=("Arial", 16)).pack(pady=20)
    
    Button(root, text="Select Image to Check", command=select_image, font=("Arial", 14)).pack(pady=20)
    
    root.mainloop()

if __name__ == "__main__":
    main()
