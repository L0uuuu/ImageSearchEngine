import cv2
import os
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import time
from pathlib import Path
import pickle

# === Global VGG16 Model (Loaded Once) ===
vgg16_model = VGG16(weights='imagenet', include_top=False)

# === Descriptor Functions ===

def indexer_images_color_histogram(chemin_repertoire):
    index = {}
    for nom_fichier in os.listdir(chemin_repertoire):
        chemin_fichier = os.path.join(chemin_repertoire, nom_fichier)
        if chemin_fichier.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = cv2.imread(chemin_fichier)
            if image is None:
                continue
            histogramme = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            index[nom_fichier] = histogramme.flatten()
    return index

def indexer_images_grayscale_histogram(chemin_repertoire):
    index = {}
    for nom_fichier in os.listdir(chemin_repertoire):
        chemin_fichier = os.path.join(chemin_repertoire, nom_fichier)
        if chemin_fichier.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = cv2.imread(chemin_fichier, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            histogramme = cv2.calcHist([image], [0], None, [256], [0, 256])
            index[nom_fichier] = histogramme.flatten()
    return index

def calculer_correlogramme(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    glcm = graycomatrix(img, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    return correlation

def indexer_images_correlogramme(dossier_images):
    index_images = {}
    for nom_fichier in os.listdir(dossier_images):
        if nom_fichier.lower().endswith(('.png', '.jpg', '.jpeg')):
            chemin_image = os.path.join(dossier_images, nom_fichier)
            correlation = calculer_correlogramme(chemin_image)
            if correlation is not None:
                index_images[nom_fichier] = correlation
    return index_images

def extract_vgg16_features(image_path, model=vgg16_model):
    try:
        img = load_img(image_path, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        features = model.predict(img)
        return features.flatten()
    except:
        return None

def indexer_images_vgg16(chemin_repertoire, cache_file="vgg16_features.pkl"):
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    index = {}
    for nom_fichier in os.listdir(chemin_repertoire):
        chemin_fichier = os.path.join(chemin_repertoire, nom_fichier)
        if chemin_fichier.lower().endswith(('.png', '.jpg', '.jpeg')):
            features = extract_vgg16_features(chemin_fichier)
            if features is not None:
                index[nom_fichier] = features
    with open(cache_file, 'wb') as f:
        pickle.dump(index, f)
    return index

def rechercher_images_similaires_histogram(index, histogramme_requete, nombre_resultats=5):
    distances = []
    for nom_fichier, histogramme_indexe in index.items():
        distance = np.linalg.norm(histogramme_requete - histogramme_indexe)
        distances.append((nom_fichier, distance))
    distances.sort(key=lambda x: x[1])
    return distances[:nombre_resultats]

def rechercher_images_similaires_correlogramme(image_requete_path, index_images, nombre_resultats=5):
    correlation_requete = calculer_correlogramme(image_requete_path)
    if correlation_requete is None:
        return []
    distances = []
    for nom_fichier, correlation_indexe in index_images.items():
        distance = abs(correlation_requete - correlation_indexe)
        distances.append((nom_fichier, distance))
    distances.sort(key=lambda x: x[1])
    return distances[:nombre_resultats]

def rechercher_images_similaires_vgg16(index, features_requete, nombre_resultats=5):
    distances = []
    for nom_fichier, features_indexe in index.items():
        distance = np.linalg.norm(features_requete - features_indexe)
        distances.append((nom_fichier, distance))
    distances.sort(key=lambda x: x[1])
    return distances[:nombre_resultats]

# === Unified Search Engine ===

def search_engine(descriptor_type, query_image_path, dataset_path, nombre_resultats=5):
    start_time = time.time()
    if descriptor_type == "color_histogram":
        index = indexer_images_color_histogram(dataset_path)
        image_requete = cv2.imread(query_image_path)
        if image_requete is None:
            raise ValueError("Failed to load query image.")
        histogramme_requete = cv2.calcHist([image_requete], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
        results = rechercher_images_similaires_histogram(index, histogramme_requete, nombre_resultats)
    elif descriptor_type == "grayscale_histogram":
        index = indexer_images_grayscale_histogram(dataset_path)
        image_requete = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)
        if image_requete is None:
            raise ValueError("Failed to load query image.")
        histogramme_requete = cv2.calcHist([image_requete], [0], None, [256], [0, 256]).flatten()
        results = rechercher_images_similaires_histogram(index, histogramme_requete, nombre_resultats)
    elif descriptor_type == "correlogram":
        index = indexer_images_correlogramme(dataset_path)
        results = rechercher_images_similaires_correlogramme(query_image_path, index, nombre_resultats)
    elif descriptor_type == "vgg16":
        index = indexer_images_vgg16(dataset_path)
        features_requete = extract_vgg16_features(query_image_path)
        if features_requete is None:
            raise ValueError("Failed to process query image with VGG16.")
        results = rechercher_images_similaires_vgg16(index, features_requete, nombre_resultats)
    else:
        raise ValueError("Invalid descriptor type. Choose: color_histogram, grayscale_histogram, correlogram, vgg16")
    search_time = time.time() - start_time
    return results, search_time

# === Evaluation Module ===

def evaluate_results(results, ground_truth):
    relevant = set(ground_truth)
    retrieved = set([nom_fichier for nom_fichier, _ in results])
    true_positives = len(relevant.intersection(retrieved))
    precision = true_positives / len(results) if results else 0
    recall = true_positives / len(ground_truth) if ground_truth else 0
    return precision, recall

# === Enhanced GUI with Tkinter ===

class ImageSearchGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Search Engine")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f2f5")  # Light gray background

        # Main frame
        self.main_frame = tk.Frame(root, bg="#f0f2f5")
        self.main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Dataset path
        self.dataset_path = "dataset"
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)

        # === Descriptor Selection ===
        self.descriptor_frame = tk.LabelFrame(self.main_frame, text="Select Descriptor", font=("Arial", 12, "bold"), bg="#ffffff", fg="#333333", padx=10, pady=10)
        self.descriptor_frame.pack(fill="x", pady=10)

        self.descriptor_var = tk.StringVar(value="color_histogram")
        descriptors = [
            ("Color Histogram", "color_histogram"),
            ("Grayscale Histogram", "grayscale_histogram"),
            ("Correlogram", "correlogram"),
            ("VGG16 (Deep Learning)", "vgg16")
        ]
        for i, (text, value) in enumerate(descriptors):
            tk.Radiobutton(
                self.descriptor_frame, text=text, variable=self.descriptor_var, value=value,
                font=("Arial", 10), bg="#ffffff", fg="#333333", selectcolor="#e6f3ff"
            ).grid(row=i // 2, column=i % 2, sticky="w", padx=10, pady=5)

        # === Query Image Section ===
        self.query_frame = tk.Frame(self.main_frame, bg="#ffffff", bd=2, relief="groove")
        self.query_frame.pack(fill="x", pady=10)

        tk.Label(self.query_frame, text="Query Image", font=("Arial", 12, "bold"), bg="#ffffff", fg="#333333").pack(pady=5)
        self.query_label = tk.Label(self.query_frame, text="No image selected", font=("Arial", 10), bg="#ffffff", fg="#666666")
        self.query_label.pack(pady=5)
        ttk.Button(self.query_frame, text="Upload Image", command=self.upload_query_image, style="TButton").pack(pady=5)
        self.query_image_path = None

        # === Search Button ===
        self.search_button = ttk.Button(self.main_frame, text="Search", command=self.search, style="TButton")
        self.search_button.pack(pady=10)

        # === Status Label ===
        self.status_label = tk.Label(self.main_frame, text="", font=("Arial", 10, "italic"), bg="#f0f2f5", fg="#666666")
        self.status_label.pack(pady=5)

        # === Results Display ===
        self.result_frame = tk.LabelFrame(self.main_frame, text="Search Results", font=("Arial", 12, "bold"), bg="#ffffff", fg="#333333", padx=10, pady=10)
        self.result_frame.pack(fill="both", expand=True, pady=10)

        self.canvas = tk.Canvas(self.result_frame, bg="#ffffff")
        self.scrollbar = ttk.Scrollbar(self.result_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="#ffffff")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # === Performance and Evaluation ===
        self.metrics_frame = tk.Frame(self.main_frame, bg="#f0f2f5")
        self.metrics_frame.pack(fill="x", pady=10)

        self.performance_label = tk.Label(self.metrics_frame, text="", font=("Arial", 10), bg="#f0f2f5", fg="#333333")
        self.performance_label.pack(side="left", padx=20)
        self.evaluation_label = tk.Label(self.metrics_frame, text="", font=("Arial", 10), bg="#f0f2f5", fg="#333333")
        self.evaluation_label.pack(side="right", padx=20)

        # Configure ttk style
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 10), padding=10, background="#4a90e2", foreground="#ffffff")
        style.map("TButton", background=[("active", "#357abd")])

    def upload_query_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if file_path:
            self.query_image_path = file_path
            self.query_label.config(text=f"{os.path.basename(file_path)}")
            img = Image.open(file_path)
            img = img.resize((120, 120), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.query_label.config(image=photo, compound="top", bg="#ffffff")
            self.query_label.image = photo
            self.status_label.config(text="Query image loaded.")

    def search(self):
        if not self.query_image_path:
            messagebox.showerror("Error", "Please upload a query image.")
            return
        if not os.path.exists(self.dataset_path) or not os.listdir(self.dataset_path):
            messagebox.showerror("Error", "Dataset folder is empty or does not exist.")
            return

        self.status_label.config(text="Processing...")
        self.root.config(cursor="wait")
        self.root.update()

        try:
            results, search_time = search_engine(self.descriptor_var.get(), self.query_image_path, self.dataset_path)

            # Clear previous results
            for widget in self.scrollable_frame.winfo_children():
                widget.destroy()

            # Display results
            for i, (nom_fichier, distance) in enumerate(results):
                frame = tk.Frame(self.scrollable_frame, bg="#ffffff", bd=2, relief="groove")
                frame.grid(row=0, column=i, padx=10, pady=10)

                img_path = os.path.join(self.dataset_path, nom_fichier)
                img = Image.open(img_path)
                img = img.resize((120, 120), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                label = tk.Label(frame, image=photo, bg="#ffffff")
                label.image = photo
                label.pack()

                tk.Label(frame, text=f"{nom_fichier}\nDist: {distance:.4f}", font=("Arial", 9), bg="#ffffff", fg="#333333").pack()

            # Performance
            self.performance_label.config(text=f"Search Time: {search_time:.2f} seconds")

            # Evaluation
            base_name = os.path.basename(self.query_image_path).split('_')[0]
            ground_truth = [f for f in os.listdir(self.dataset_path) if f.startswith(base_name)]
            precision, recall = evaluate_results(results, ground_truth)
            self.evaluation_label.config(text=f"Precision: {precision:.2f}, Recall: {recall:.2f}")

            self.status_label.config(text="Search completed.")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_label.config(text="Error during search.")

        self.root.config(cursor="")

# === Main Execution ===

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSearchGUI(root)
    root.mainloop()