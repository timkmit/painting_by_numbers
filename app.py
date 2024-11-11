import cv2
import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchvision import models
from PIL import Image
import os

# === Настройки конфигурации ===
NUM_COLORS_MAIN = 15  # Количество цветов для главных объектов (корректируем до 20)
NUM_COLORS_BACKGROUND = 10  # Количество цветов для фона
MIN_AREA_CONTOUR = 30  # Минимальная площадь полигона (в пикселях)
MAX_DIMENSION = 1000  # Максимальный размер изображения для обработки (в пикселях)
MEAN_SHIFT_RADIUS = 21  # Радиус фильтра Mean Shift для сглаживания фона
MEAN_SHIFT_COLOR = 41  # Радиус цветового пространства для фильтрации фона
FONT_SIZE = 0.3  # Размер шрифта для номеров цветов на контурном изображении
FONT_COLOR = (255, 0, 0)  # Цвет шрифта для номеров на контурном изображении (красный)
CONTOUR_COLOR = (0, 0, 0)  # Цвет контуров на контурном изображении (черный)
CONTOUR_THICKNESS = 1  # Толщина контуров
FONT_THICKNESS = 1  # Толщина шрифта для номеров

def segment_image(image):
    model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
    preprocess = T.Compose([
        T.ToPILImage(),
        T.Resize((MAX_DIMENSION, MAX_DIMENSION)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(input_batch)["out"][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()
    mask = np.isin(output_predictions, [15, 7]).astype(np.uint8)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask

def create_paint_by_numbers(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка при загрузке изображения: {image_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width = image.shape[:2]
    aspect_ratio = width / height
    if width > height:
        new_width = 1500
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = 1500
        new_width = int(new_height * aspect_ratio)
    image = cv2.resize(image, (new_width, new_height))

    mask = segment_image(image)
    main_objects = cv2.bitwise_and(image, image, mask=mask)
    background = cv2.bitwise_and(image, image, mask=1 - mask)

    main_objects[np.all(main_objects == [255, 255, 255], axis=-1)] = 0
    background[np.all(background == [255, 255, 255], axis=-1)] = 0

    background_smooth = cv2.pyrMeanShiftFiltering(background, MEAN_SHIFT_RADIUS, MEAN_SHIFT_COLOR)

    main_pixel_values = main_objects.reshape((-1, 3))
    main_pixel_values = np.float32(main_pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, main_labels, main_centers = cv2.kmeans(main_pixel_values, NUM_COLORS_MAIN, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    main_centers = np.uint8(main_centers)
    main_labeled_image = main_centers[main_labels.flatten()].reshape(main_objects.shape)

    background_pixel_values = background_smooth.reshape((-1, 3))
    background_pixel_values = np.float32(background_pixel_values)
    _, background_labels, background_centers = cv2.kmeans(background_pixel_values, NUM_COLORS_BACKGROUND, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    background_centers = np.uint8(background_centers)

    colors = ""
    colors += "Main Object Colors (RGB):\n"
    for i, color in enumerate(main_centers):
        colors += f"Color {i + 1}: {color}\n"
    colors += "\nBackground Colors (RGB):\n"
    for i, color in enumerate(background_centers):
        colors += f"Color {i + 1 + len(main_centers)}: {color}\n"

    background_labeled_image = background_centers[background_labels.flatten()].reshape(background.shape)
    combined_image = cv2.add(background_labeled_image, main_labeled_image)
    combined_image = np.clip(combined_image, 0, 255)

    blank_image = np.ones_like(combined_image, dtype=np.uint8) * 255
    transparent_image = ((combined_image * 0.8) + (blank_image * 0.2)).astype(np.uint8)  # Четвертое изображение, светлее
    label_matrix_main = main_labels.reshape((new_height, new_width))
    label_matrix_background = background_labels.reshape((new_height, new_width))
    all_centers = np.concatenate((main_centers, background_centers), axis=0)

    for i, color in enumerate(all_centers):
        if i < len(main_centers):
            mask = np.uint8(label_matrix_main == i) * 255
        else:
            mask = np.uint8(label_matrix_background == (i - len(main_centers))) * 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area >= MIN_AREA_CONTOUR:
                cv2.drawContours(blank_image, [contour], -1, CONTOUR_COLOR, CONTOUR_THICKNESS)
                cv2.drawContours(transparent_image, [contour], -1, CONTOUR_COLOR, CONTOUR_THICKNESS)
                if contour_area > MIN_AREA_CONTOUR * 2:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cv2.putText(blank_image, str(i + 1), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, FONT_COLOR, FONT_THICKNESS)
                        cv2.putText(transparent_image, str(i + 1), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, FONT_COLOR, FONT_THICKNESS)

    fig, ax = plt.subplots(1, 4, figsize=(24, 6))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(combined_image)
    ax[1].set_title("Paint by Numbers")
    ax[1].axis("off")

    ax[2].imshow(blank_image)
    ax[2].set_title("Outline with Numbers")
    ax[2].axis("off")

    ax[3].imshow(transparent_image)
    ax[3].set_title("Transparent with Numbers")
    ax[3].axis("off")

    os.makedirs("./saved", exist_ok=True)
    combined_image = Image.fromarray(combined_image.astype(np.uint8))
    combined_image.save("./saved/combined_image.jpg")
    blank_image = Image.fromarray(blank_image.astype(np.uint8))
    blank_image.save("./saved/blank_image.jpg")
    transparent_image = Image.fromarray(transparent_image.astype(np.uint8))
    transparent_image.save("./saved/transparent_image.jpg")
    with open("./saved/colors.txt", "w", encoding="utf-8") as file:
        file.write(colors)

    plt.tight_layout()
    plt.show()

create_paint_by_numbers("test.jpg")
