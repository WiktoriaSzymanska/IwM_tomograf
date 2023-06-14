import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy.fftpack import fft, ifft, fftfreq


# Wyznaczenie linii od punktu (x0, y0) do (x1, y1)
def bresenham_line(x0, y0, x1, y1):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    steep = dy > dx
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        dx, dy = dy, dx
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    y_step = 1 if y0 < y1 else -1
    error = dx / 2
    y = y0
    points = []
    for x in range(x0, x1 + 1):
        coord = (y, x) if steep else (x, y)
        points.append(coord)
        error -= dy
        if error < 0:
            y += y_step
            error += dx
    return points


# Wyznaczenie pozycji emiterów
def calculate_emitter_positions(radius, alpha, alpha_range, num_detectors, center):
    cx, cy = center
    fi = np.radians(alpha_range) / (num_detectors - 1)
    angles = np.radians(alpha) + np.arange(num_detectors) * fi - np.radians(alpha_range / 2)
    xdi = radius * np.cos(angles) + cx
    ydi = radius * np.sin(angles) + cy
    return np.vstack((xdi, ydi)).T.astype(int)


# Wyznaczenie pozycji detektorów
def calculate_detector_positions(radius, alpha, alpha_range, num_detectors, center):
    cx, cy = center
    fi = np.radians(alpha_range) / (num_detectors - 1)
    angles = np.radians(alpha) + np.arange(num_detectors) * fi - np.radians(alpha_range / 2) + np.pi
    xdi = radius * np.cos(angles) + cx
    ydi = radius * np.sin(angles) + cy
    return np.vstack((xdi, ydi)).T.astype(int)


# Wyznaczenie linii skanów więdzy emiterem a detektorami
def scan_lines(emitters, detectors):
    lines = []
    for emitter, detector in zip(emitters, detectors[::-1]):
        x0, y0 = emitter
        x1, y1 = detector
        lines.append(np.array(bresenham_line(x0, y0, x1, y1)))
    return lines


# Zaznaczenie linii skanów na obrazie
def draw_scan_lines(image, emitters, detectors, lines):
    marked_image = image.copy()
    draw = ImageDraw.Draw(marked_image)
    for line in lines:
        for point in line:
            x, y = point
            draw.point((x, y), fill=(0, 0, 255))
    for emitter in emitters:
        draw.rectangle([emitter[0] - 2, emitter[1] - 2, emitter[0] + 2, emitter[1] + 2], fill=(255, 0, 0))
    for detector in detectors:
        draw.rectangle([detector[0] - 2, detector[1] - 2, detector[0] + 2, detector[1] + 2], fill=(0, 255, 0))
    return marked_image


# Normalizacja obrazu w skali szarości
def normalize_img(image):
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    return image.astype(np.uint8)


# Transformata Radona dla jednej iteracji - wyznaczenie pojedynczej linii sinogramu
def radon_transform(gray_image, lines):
    width, height = gray_image.size
    sums = []
    for line in lines:
        sum = 0
        for point in line:
            x, y = point
            if x < width and y < height:
                sum += gray_image.getpixel((x, y))
        sums.append(sum)
    return sums


# Funkcja generująca obrazy dla kolejnych iteracji skanowania
# Caching dla szybszego ładowania obrazów po zmianie iteracji do wyświetlenia
@st.cache_data
def perform_scan_iterations(radius, alpha_range, num_detectors, center, _image, num_iterations, show_iterations):
    gray_image = image.convert("L")
    sinogram = np.zeros((num_detectors, num_iterations), dtype=np.float32)
    marked_images = []  # Obrazy z zaznaczonymi liniami skanu dla kolejnych iteracji
    sinograms = []  # Obrazy z sinogramami dla kolejnych iteracji

    # Obliczenie kroku kątowego na podstawie liczby iteracji
    alpha_step = 360 / num_iterations
    # Wykonanie skanowania dla każdej iteracji
    for i in range(num_iterations):
        alpha = i * alpha_step
        emitters = calculate_emitter_positions(radius, alpha, alpha_range, num_detectors, center)
        detectors = calculate_detector_positions(radius, alpha, alpha_range, num_detectors, center)
        lines = scan_lines(emitters, detectors)  # Wyznaczenie linii skanów
        # Zaznaczenie pozycji i skanowanych linii na obrazie
        marked_image = draw_scan_lines(image, emitters, detectors, lines)
        marked_images.append(marked_image)

        # Generowanie sinogramu
        sinogram_line = radon_transform(gray_image, lines)  # Transformata radona
        sinogram[:, i] = sinogram_line  # Zaznaczenie na sinogramie
        if show_iterations or i == num_iterations-1:
            sinograms.append(sinogram.copy())
    return marked_images, sinograms


def clip(array, min, max):
    array[array < min] = min
    array[array > max] = max
    return array


def filter_sinogram(sinogram):
    n = sinogram.shape[0]
    filter = 2 * np.abs(fftfreq(n).reshape(-1, 1))
    result = ifft(fft(sinogram, axis=0) * filter, axis=0)
    result = clip(np.real(result), 0, 1)
    return result


# Odtwarzanie obrazu na podstawie sinogramu - odwrotna transformata Radona
# Caching dla szybszego ładowania obrazów po zmianie iteracji do wyświetlenia
@st.cache_data
def backprojection(sinogram, radius, alpha_range, num_detectors, center, width, height, num_iterations, show_iterations, filter):

    if filter:
        sinogram = filter_sinogram(sinogram)

    output_image = np.zeros((height, width), dtype=np.float32)
    output_images = []
    alpha_step = 360 / num_iterations
    # Wykonanie odtwarzania dla każdej iteracji
    for i in range(num_iterations):
        alpha = i * alpha_step
        emitters = calculate_emitter_positions(radius, alpha, alpha_range, num_detectors, center)
        detectors = calculate_detector_positions(radius, alpha, alpha_range, num_detectors, center)
        lines = scan_lines(emitters, detectors)
        # Odwrotna transformata Radona
        for j, line in enumerate(lines):
            for point in line:
                x, y = point
                if 0 <= x < width and 0 <= y < height:
                    output_image[y, x] += sinogram[j, i]
        if show_iterations or i == num_iterations-1:
            output_images.append(output_image.copy())
    return output_images


st.set_page_config(layout="wide")
st.markdown('# Symulator tomografu')
container = st.container()
col1, col2, col3 = st.columns(3)
col1.markdown('Obraz wejściowy')
col2.markdown('Sinogram')
col3.markdown('Obraz wyjściowy')

# Wybór pliku wejściowego
file_path = container.file_uploader("Wybierz plik wejściowy", type=["jpg", "jpeg", "png"])
if file_path is None:
    file_path = "tomograf-zdjecia/Shepp_logan.jpg"

image = Image.open(file_path)

# Zmiana rozmiaru obrazu
max_width, max_height = 400, 400
width, height = image.size
if width > max_width or height > max_height:
    image = image.resize((max_width, max_height))

# Wymiary obrazu
width, height = image.size
# Współrzędne środka obrazu
center = (width // 2, height // 2)
# Promień
radius = min(width, height) // 2

# Wyświetlanie parametrów
st.sidebar.title("Parametry")
alpha_range = st.sidebar.slider("Kąt rozwarcia detektorów", 1, 180, 140)
num_detectors = st.sidebar.slider("Liczba detektorów", 2, 270, 140)
num_iterations = st.sidebar.slider("Liczba iteracji skanowania", 2, 180, 50)
show_iterations = st.sidebar.checkbox("Pokaż iteracje skanowania", value=False)
filter = st.sidebar.checkbox("Użyj filtra", value=False)

# Wywołanie funkcji z spinnerem
with st.spinner('Trwa wykonywanie skanowania...'):
    # Wykonanie skanowania
    marked_images, sinograms = perform_scan_iterations(radius, alpha_range, num_detectors, center, image, num_iterations, show_iterations)

if show_iterations:
    # Wyświetlenie suwaka
    selected_iteration = container.slider("Wybierz iterację", 1, num_iterations, num_iterations)
    final_sinogram = sinograms[-1]
    # Wyświetlenie obrazu na podstawie wybranej iteracji
    col1.image(marked_images[selected_iteration - 1], use_column_width=True)
else:
    selected_iteration = 1
    final_sinogram = sinograms[0]
    col1.image(image, use_column_width=True)

final_sinogram = normalize_img(final_sinogram)

# Wyświetlenie sinogramu na podstawie wybranej iteracji
sinogram = sinograms[selected_iteration - 1]
fig, ax = plt.subplots()
ax.imshow(sinogram, cmap='gray')
ax.axis('off')
col2.pyplot(fig)

# Odtworzenie obrazu
reconstructed_images = backprojection(final_sinogram, radius, alpha_range, num_detectors, center, width, height, num_iterations, show_iterations, filter)

# Wyświetlenie sinogramu na podstawie wybranej iteracji
reconstructed_image = reconstructed_images[selected_iteration - 1]
fig, ax = plt.subplots()
ax.imshow(reconstructed_image, cmap='gray')
ax.axis('off')
col3.pyplot(fig)
