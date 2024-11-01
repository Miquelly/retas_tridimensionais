import colorsys
import json
import sys
import warnings
from typing import Dict, Tuple


import cv2
import imutils
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.protobuf.json_format import Parse
from is_msgs.image_pb2 import HumanKeypoints as HKP
from is_msgs.image_pb2 import ObjectAnnotations
from Criar_df import Criar_DataFrame

links = [
    (HKP.Value("LEFT_SHOULDER"), HKP.Value("RIGHT_SHOULDER")),
    (HKP.Value("LEFT_SHOULDER"), HKP.Value("LEFT_HIP")),
    (HKP.Value("RIGHT_SHOULDER"), HKP.Value("RIGHT_HIP")),
    (HKP.Value("LEFT_HIP"), HKP.Value("RIGHT_HIP")),
    (HKP.Value("LEFT_SHOULDER"), HKP.Value("LEFT_EAR")),
    (HKP.Value("RIGHT_SHOULDER"), HKP.Value("RIGHT_EAR")),
    (HKP.Value("LEFT_SHOULDER"), HKP.Value("LEFT_ELBOW")),
    (HKP.Value("LEFT_ELBOW"), HKP.Value("LEFT_WRIST")),
    (HKP.Value("LEFT_HIP"), HKP.Value("LEFT_KNEE")),
    (HKP.Value("LEFT_KNEE"), HKP.Value("LEFT_ANKLE")),
    (HKP.Value("RIGHT_SHOULDER"), HKP.Value("RIGHT_ELBOW")),
    (HKP.Value("RIGHT_ELBOW"), HKP.Value("RIGHT_WRIST")),
    (HKP.Value("RIGHT_HIP"), HKP.Value("RIGHT_KNEE")),
    (HKP.Value("RIGHT_KNEE"), HKP.Value("RIGHT_ANKLE")),
    (HKP.Value("NOSE"), HKP.Value("LEFT_EYE")),
    (HKP.Value("LEFT_EYE"), HKP.Value("LEFT_EAR")),
    (HKP.Value("NOSE"), HKP.Value("RIGHT_EYE")),
    (HKP.Value("RIGHT_EYE"), HKP.Value("RIGHT_EAR")),
]


def editar_DataFrame(id, x, df, parts, gesture):
    df = df._append(
        {
            "Frame": x,
            "ID": id,
            "Gesture": gesture,
            "x_LEFT_SHOULDER": parts[HKP.Value("LEFT_SHOULDER")][0],
            "y_LEFT_SHOULDER": parts[HKP.Value("LEFT_SHOULDER")][1],
            "z_LEFT_SHOULDER": parts[HKP.Value("LEFT_SHOULDER")][2],
            "x_RIGHT_SHOULDER": parts[HKP.Value("RIGHT_SHOULDER")][0],
            "y_RIGHT_SHOULDER": parts[HKP.Value("RIGHT_SHOULDER")][1],
            "z_RIGHT_SHOULDER": parts[HKP.Value("RIGHT_SHOULDER")][2],
            "x_LEFT_HIP": parts[HKP.Value("LEFT_HIP")][0],
            "y_LEFT_HIP": parts[HKP.Value("LEFT_HIP")][1],
            "z_LEFT_HIP": parts[HKP.Value("LEFT_HIP")][2],
            "x_RIGHT_HIP": parts[HKP.Value("RIGHT_HIP")][0],
            "y_RIGHT_HIP": parts[HKP.Value("RIGHT_HIP")][1],
            "z_RIGHT_HIP": parts[HKP.Value("RIGHT_HIP")][2],
            "x_LEFT_EAR": parts[HKP.Value("LEFT_EAR")][0],
            "y_LEFT_EAR": parts[HKP.Value("LEFT_EAR")][1],
            "z_LEFT_EAR": parts[HKP.Value("LEFT_EAR")][2],
            "x_RIGHT_EAR": parts[HKP.Value("RIGHT_EAR")][0],
            "y_RIGHT_EAR": parts[HKP.Value("RIGHT_EAR")][1],
            "z_RIGHT_EAR": parts[HKP.Value("RIGHT_EAR")][2],
            "x_LEFT_ELBOW": parts[HKP.Value("LEFT_ELBOW")][0],
            "y_LEFT_ELBOW": parts[HKP.Value("LEFT_ELBOW")][1],
            "z_LEFT_ELBOW": parts[HKP.Value("LEFT_ELBOW")][2],
            "x_RIGHT_ELBOW": parts[HKP.Value("RIGHT_ELBOW")][0],
            "y_RIGHT_ELBOW": parts[HKP.Value("RIGHT_ELBOW")][1],
            "z_RIGHT_ELBOW": parts[HKP.Value("RIGHT_ELBOW")][2],
            "x_LEFT_WRIST": parts[HKP.Value("LEFT_WRIST")][0],
            "y_LEFT_WRIST": parts[HKP.Value("LEFT_WRIST")][1],
            "z_LEFT_WRIST": parts[HKP.Value("LEFT_WRIST")][2],
            "x_RIGHT_WRIST": parts[HKP.Value("RIGHT_WRIST")][0],
            "y_RIGHT_WRIST": parts[HKP.Value("RIGHT_WRIST")][1],
            "z_RIGHT_WRIST": parts[HKP.Value("RIGHT_WRIST")][2],
            "x_LEFT_KNEE": parts[HKP.Value("LEFT_KNEE")][0],
            "y_LEFT_KNEE": parts[HKP.Value("LEFT_KNEE")][1],
            "z_LEFT_KNEE": parts[HKP.Value("LEFT_KNEE")][2],
            "x_RIGHT_KNEE": parts[HKP.Value("RIGHT_KNEE")][0],
            "y_RIGHT_KNEE": parts[HKP.Value("RIGHT_KNEE")][1],
            "z_RIGHT_KNEE": parts[HKP.Value("RIGHT_KNEE")][2],
            "x_LEFT_ANKLE": parts[HKP.Value("LEFT_ANKLE")][0],
            "y_LEFT_ANKLE": parts[HKP.Value("LEFT_ANKLE")][1],
            "z_LEFT_ANKLE": parts[HKP.Value("LEFT_ANKLE")][2],
            "x_RIGHT_ANKLE": parts[HKP.Value("RIGHT_ANKLE")][0],
            "y_RIGHT_ANKLE": parts[HKP.Value("RIGHT_ANKLE")][1],
            "z_RIGHT_ANKLE": parts[HKP.Value("RIGHT_ANKLE")][2],
            "x_NOSE": parts[HKP.Value("NOSE")][0],
            "y_NOSE": parts[HKP.Value("NOSE")][1],
            "z_NOSE": parts[HKP.Value("NOSE")][1],
            "x_LEFT_EYE": parts[HKP.Value("LEFT_EYE")][0],
            "y_LEFT_EYE": parts[HKP.Value("LEFT_EYE")][1],
            "z_LEFT_EYE": parts[HKP.Value("LEFT_EYE")][2],
            "x_RIGHT_EYE": parts[HKP.Value("RIGHT_EYE")][0],
            "y_RIGHT_EYE": parts[HKP.Value("RIGHT_EYE")][1],
            "z_RIGHT_EYE": parts[HKP.Value("RIGHT_EYE")][2],
        },
        ignore_index=True,
    )

    return df


def _id_to_rgb_color(id) -> Tuple[float, float, float]:

    hue = (id % 20) / 20
    saturation = 0.8
    luminance = 0.6
    r, g, b = [x for x in colorsys.hls_to_rgb(hue, luminance, saturation)]

    return r, g, b


def render_skeletons_3d(skeletons, ax, identifier=0):

    parts: Dict = {}

    for skeleton in skeletons.objects:

        # if skeleton.id != identifier:
        #     continue
        # for part in skeleton.keypoints:
        #     parts[part.id] = (part.position.x, part.position.y, part.position.z)
        for link_parts in links:
            begin, end = link_parts
            if begin in parts and end in parts:
                x_pair = [parts[begin][0], parts[end][0]]
                y_pair = [parts[begin][1], parts[end][1]]
                z_pair = [parts[begin][2], parts[end][2]]
                color = _id_to_rgb_color(id=skeleton.id)
                ax.plot(
                    x_pair,
                    y_pair,
                    zs=z_pair,
                    linewidth=3,
                    color=color,
                )

    return identifier, parts


def load_json(filename):
    with open(file=filename, mode="r", encoding="utf-8") as file:
        options = Parse(file.read(), ObjectAnnotations())
        return options


def main() -> None:

    """Plot - Matplot"""

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    plt.ioff()

    """ DataFrame - {name_df}.csv """

    name_df = "p002g01c02"
    if not os.path.isfile(f"{name_df}.csv"):
        Criar_DataFrame(name_df)

    p002g01c02 = pd.read_csv(
        f"{name_df}.csv",
        dtype={
            "Frame": int,
            "ID": int,
            "Gesture": "category",
        },
    )

    ultimo_frame = p002g01c02["Frame"].max()

    if np.isnan(ultimo_frame):
        ultimo_frame = 0
    else:
        ultimo_frame += 1

    frames = 563 + 1

    for x in range(ultimo_frame, frames):

        ax.clear()
        ax.view_init(azim=0, elev=20)
        ax.set_xlim(-4, 4)
        ax.set_xticks(np.arange(-4, 4, 1))
        ax.set_ylim(-4, 4)
        ax.set_yticks(np.arange(-4, 4, 1))
        ax.set_zlim(-0.25, 2)
        ax.set_zticks(np.arange(0, 2, 0.5))
        ax.set_xlabel("X", labelpad=20)
        ax.set_ylabel("Y", labelpad=10)
        ax.set_zlabel("Z", labelpad=5)

        """ load_json """

        caminho = f"p002g01_3d/{x}.json"
        arq = load_json(caminho)
        id, parts = render_skeletons_3d(arq, ax)

        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)

        """ view frame and 3d """

        height = 540

        view_3d = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # cv2.imshow("name", view_3d)
        view_3d_resized = imutils.resize(view_3d, height=height)
        # print(view_3d_resized.shape)

        imagem = cv2.imread(f"p002g01c02_frames/{x}.png")
        # cv2.imshow("name1", imagem)
        imagem_resized = imutils.resize(imagem, height=height)
        # print(imagem_resized.shape)

        cv2.imshow("Frame x View_3d", np.hstack([imagem_resized, view_3d_resized]))

        # gesture = gesture._append({'ID': i, 'Gesture': i*2}, ignore_index=True)

        # temp_gesture = pd.DataFrame({'ID': [i], 'Gesture': [i * 2]})
        # # Concatenar o DataFrame temporário ao DataFrame principal
        # gesture = pd.concat([gesture, temp_gesture], ignore_index=True)

        """ Comandos classificacao """

        key = cv2.waitKey(-1)  # Ele espera até que alguma

        if key == ord("a"):
            # a: Esquerda
            p002g01c02 = editar_DataFrame(id, x, p002g01c02, parts, 1)

        elif key == ord("d"):
            # d: Direita
            p002g01c02 = editar_DataFrame(id, x, p002g01c02, parts, 2)

        elif key == ord("w"):
            # w: Duas apontando
            p002g01c02 = editar_DataFrame(id, x, p002g01c02, parts, 3)

        elif key == ord("s"):
            # s: Nenhuma
            p002g01c02 = editar_DataFrame(id, x, p002g01c02, parts, 0)

        p002g01c02.to_csv(f"{name_df}.csv", index=False)

        if key == ord("q"):
            break

            # p002g01c02.to_csv(f"{name_df}.csv", index=False)
            # print(p002g01c02)
            # return

    """ Salve DataFrame """

    p002g01c02.to_csv(f"{name_df}.csv", index=False)
    print(p002g01c02)


if __name__ == "__main__":
    main()

# source .venv/bin/activate
# https://hub.asimov.academy/tutorial/rgb-e-mescla-de-imagens-entendendo-e-aplicando-em-python-com-opencv/

# https://br.matplotlib.net/stable/tutorials/introductory/images.html

# https://github.com/PyImageSearch/imutils
# https://snyk.io/advisor/python/imutils/functions/imutils.resize

# https://www.youtube.com/watch?v=hpX_L5dYh-g

# https://hub.asimov.academy/tutorial/como-verificar-se-um-arquivo-existe-em-python-sem-excecoes/

# https://medium.com/@guilhermegobbo04/homotetia-e-vis%C3%A3o-computacional-qual-a-rela%C3%A7%C3%A3o-f23ca39b8db6

# https://www.tecgraf.puc-rio.br/ftp_pub/lfm/CIV2802-GeometriaComputacional.pdf
