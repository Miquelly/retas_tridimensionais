import cv2
import os

path = "p002g01c02_plot"
pre_imgs = os.listdir(path)
imgs = []

# name = f"p002g01c02_plot/plot{i}.png"

for i in range(len(pre_imgs)):
    array = cv2.imread(f"video/plot{i}.png")

    if array is None:
        print(f"Erro ao carregar a imagem plot{i}.png")
    else:
        imgs.append(array)

if len(imgs) == 0:
    print("Nenhuma imagem foi carregada.")
else:
    cv2_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    size = (imgs[0].shape[1], imgs[0].shape[0])
    video = cv2.VideoWriter("plots2.mp4", cv2_fourcc, 8, size)
    for img in imgs:
        video.write(img)
    video.release()
