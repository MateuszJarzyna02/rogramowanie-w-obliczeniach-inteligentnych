import os
import numpy as np
import pandas as pd
from skimage import io, color, feature, img_as_ubyte
from skimage.util import view_as_windows
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
#Parametry
input_folders = ['obraz1', 'obraz2', 'obraz3']
output_folder = 'texture_samples'
sample_size = (128, 128)
bit_depth = 5
distances = [1, 3, 5]
angles = [0, 45, 90, 135]

# Wycinanie próbek tekstury
def extract_texture(input_folders, output_folder, sample_size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for folder in input_folders:
        category = os.path.basename(folder)
        category_output = os.path.join(output_folder, category)

        if not os.path.exists(category_output):
            os.makedirs(category_output)

        for filename in os.listdir(folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder, filename)
                img = io.imread(img_path)

                # Sprawdź czy obraz jest kolorowy
                if len(img.shape) == 3:
                    img = color.rgb2gray(img)

                windows = view_as_windows(img, sample_size, step=sample_size)

                for i in range(windows.shape[0]):
                    for j in range(windows.shape[1]):
                        sample = windows[i, j]
                        sample_path = os.path.join(
                            category_output,
                            f"{os.path.splitext(filename)[0]}_{i}_{j}.png"
                        )
                        io.imsave(sample_path, (sample * 255).astype(np.uint8))

# Obliczanie cech tekstury
def calculate_texture(image_path, distances, angles, bit_depth):
    img = io.imread(image_path)
    if len(img.shape) == 3:
        img = color.rgb2gray(img)

    img = img_as_ubyte(img)
    max_val = 2 ** bit_depth
    img = (img // (256 // max_val)).astype(np.uint8)

    features = []
    for d in distances:
        for angle in angles:
            glcm = feature.graycomatrix(img, [d], [np.deg2rad(angle)],
                                        levels=max_val, symmetric=True)

            dissimilarity = feature.graycoprops(glcm, 'dissimilarity')[0, 0]
            correlation = feature.graycoprops(glcm, 'correlation')[0, 0]
            contrast = feature.graycoprops(glcm, 'contrast')[0, 0]
            energy = feature.graycoprops(glcm, 'energy')[0, 0]
            homogeneity = feature.graycoprops(glcm, 'homogeneity')[0, 0]
            asm = feature.graycoprops(glcm, 'ASM')[0, 0]

            features.extend([dissimilarity, correlation, contrast, energy, homogeneity, asm])

    return features

def process_textures(input_folders, output_folder, sample_size, distances, angles, bit_depth):
    print("Ekstrakcja próbek")
    extract_texture(input_folders, output_folder, sample_size)
    print("Obliczanie cech")
    data = []
    columns = []

    for d in distances:
        for angle in angles:
            for feature_name in ['dissimilarity', 'correlation', 'contrast',
                                 'energy', 'homogeneity', 'ASM']:
                columns.append(f"{feature_name}_d{d}_a{angle}")
    columns.append('category')

    for category in os.listdir(output_folder):
        category_path = os.path.join(output_folder, category)

        for sample_file in os.listdir(category_path):
            sample_path = os.path.join(category_path, sample_file)

            features = calculate_texture(sample_path, distances, angles, bit_depth)
            features.append(category)  # Dodaj kategorię
            data.append(features)

    # Zapisz do pliku
    print("Zapisywanie danych")
    df = pd.DataFrame(data, columns=columns)
    df.to_csv('texture_features.csv', index=False)

    return df

#Klasyfikacja
def classify_textures(df):
    X = df.drop('category', axis=1).values
    y = df['category'].values

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("Trenowanie klasyfikatora")
    clf = SVC(kernel='rbf', gamma='scale')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Dokładność klasyfikacji: {accuracy:.2f}")

    return clf, le

if __name__ == "__main__":
    df = process_textures(input_folders, output_folder, sample_size, distances, angles, bit_depth)
    clf, le = classify_textures(df)