import colorsys
from typing import Dict, Tuple

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.protobuf.json_format import Parse
from is_msgs.image_pb2 import HumanKeypoints as HKP
from is_msgs.image_pb2 import ObjectAnnotations
from Classificador import Gesture

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


def configure_plot(ax):
    """Configurações do gráfico 3D."""
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


def _id_to_rgb_color(id) -> Tuple[float, float, float]:

    hue = (id % 20) / 20
    saturation = 0.8
    luminance = 0.6
    r, g, b = [x for x in colorsys.hls_to_rgb(hue, luminance, saturation)]

    return r, g, b


def render_skeletons_3d(skeletons, ax, identifier=0):

    parts: Dict = {}

    for skeleton in skeletons.objects:

        if skeleton.id != identifier:
            continue
        for part in skeleton.keypoints:
            parts[part.id] = (part.position.x, part.position.y, part.position.z)
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


def text(imagem, list_parametros, variavel, x):
    cv2.putText(
        imagem,
        f"Frame: {x}",
        (300, 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    text00 = f"Gesto_Real: {variavel}"
    cv2.putText(
        imagem, text00, (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )
    text = f"Gesto_Classificado: {list_parametros[4]}"
    cv2.putText(
        imagem, text, (300, 225), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )
    text0 = f"left_distancia: {list_parametros[0]:.2}"
    cv2.putText(
        imagem, text0, (300, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )
    text1 = f"left_teta: {list_parametros[1]:.4}"
    cv2.putText(
        imagem, text1, (300, 275), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )
    text2 = f"right_distancia: {list_parametros[2]:.2}"
    cv2.putText(
        imagem, text2, (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )
    text3 = f"right_teta: {list_parametros[3]:.4}"
    cv2.putText(
        imagem, text3, (300, 325), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )


def view(fig, list_parametros, variavel, x):
    """view frame, skeletons_3d and parameters"""
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)

    height = 540

    view_3d = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # cv2.imshow("name", view_3d)
    view_3d_resized = imutils.resize(view_3d, height=height)
    # print(view_3d_resized.shape)

    imagem = cv2.imread(f"p002g01c02_frames/{x}.png")
    # cv2.imshow("name1", imagem)

    text(imagem, list_parametros, variavel, x)

    imagem_resized = imutils.resize(imagem, height=height)
    # print(imagem_resized.shape)

    cv2.imshow("Frame x View_3d", np.hstack([imagem_resized, view_3d_resized]))


def main():

    """Plot - Matplot"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    plt.ioff()

    """ DataFrame - {name_df}.csv """
    name_df = "p002g01c02"

    p002g01c02 = pd.read_csv(
        f"{name_df}.csv",
        dtype={
            "Frame": int,
            "ID": int,
            "Gesture": "category",
        },
    )

    X = p002g01c02.copy()

    X.drop(
        columns=[
            "ID",
            "Gesture",
            "x_LEFT_HIP",
            "y_LEFT_HIP",
            "z_LEFT_HIP",
            "x_RIGHT_HIP",
            "y_RIGHT_HIP",
            "z_RIGHT_HIP",
            "x_LEFT_EAR",
            "y_LEFT_EAR",
            "z_LEFT_EAR",
            "x_RIGHT_EAR",
            "y_RIGHT_EAR",
            "z_RIGHT_EAR",
            "x_LEFT_KNEE",
            "y_LEFT_KNEE",
            "z_LEFT_KNEE",
            "x_RIGHT_KNEE",
            "y_RIGHT_KNEE",
            "z_RIGHT_KNEE",
            "x_LEFT_ANKLE",
            "y_LEFT_ANKLE",
            "z_LEFT_ANKLE",
            "x_RIGHT_ANKLE",
            "y_RIGHT_ANKLE",
            "z_RIGHT_ANKLE",
            "x_LEFT_EYE",
            "y_LEFT_EYE",
            "z_LEFT_EYE",
            "x_RIGHT_EYE",
            "y_RIGHT_EYE",
            "z_RIGHT_EYE",
        ],
        inplace=True,
    )

    X.set_index("Frame", inplace=True)

    y = p002g01c02["Gesture"]
    y = y.set_axis(p002g01c02["Frame"])

    """ Classificador """
    obj = Gesture()

    for x in X.index:

        "view"
        variavel = y.loc[x]
        configure_plot(ax)

        """ load_json """
        caminho = f"p002g01_3d/{x}.json"
        arq = load_json(caminho)
        render_skeletons_3d(arq, ax)

        """ Parametros classificador """
        npX = X.loc[x].to_numpy()
        list_parametros = obj.for_plot(x, npX)

        """ Reta """
        p0_L = npX[0:3]  # Ombro
        p2_L = npX[12:15]  # pulso
        p1_L = npX[6:9]  # cotovelo

        p0_R = npX[3:6]
        p2_R = npX[15:18]
        p1_R = npX[9:12]

        Xreg = np.array(
            [
                [p0_R[0], p0_R[1]],
                [p1_R[0], p1_R[1]],
                [p2_R[0], p2_R[1]],
            ]
        )
        yreg = np.array(
            [
                p0_R[2],
                p1_R[2],
                p2_R[2],
            ]
        )

        (
            rightx,
            righty,
            rightz,
            leftx,
            lefty,
            leftz,
            reg1,
            reg2,
            reg3,
            reg4,
        ) = obj.reta_regressao_projecao(npX)

        diretor_right = np.array([1 / reg1, 1 / reg2, 1])
        diretor_left = np.array([1 / reg3, 1 / reg4, 1])

        normal = np.array([0, 0, 1])

        alpha = obj.angulo_plano(diretor_left, normal)

        # ax.plot(
        #     [0, diretor_left[0]],
        #     [0, diretor_left[1]],
        #     [0, diretor_left[2]],
        #     color="g",
        # )

        ax.plot(
            rightx,
            righty,
            rightz,
            color="g",
        )

        # ax.plot(
        #     leftx,
        #     lefty,
        #     leftz,
        #     color="c",
        # )

        ax.scatter(
            rightx,
            righty,
            rightz,
            color="b",
        )
        ax.scatter(
            leftx,
            lefty,
            leftz,
            color="b",
        )

        # ax.plot(
        #     xxx,
        #     yyy,
        #     reg.predict(aaa),
        #     color="c",
        # )

        rightx, righty, rightz, leftx, lefty, leftz = obj.reta(npX)

        if list_parametros[4] == 3:
            ax.plot(
                rightx,
                righty,
                rightz,
                color="m",
            )

            ax.plot(
                leftx,
                lefty,
                leftz,
                color="b",
            )

        elif list_parametros[4] == 2:
            ax.plot(
                rightx,
                righty,
                rightz,
                color="m",
            )

        elif list_parametros[4] == 1:
            ax.plot(
                leftx,
                lefty,
                leftz,
                color="b",
            )

        """view frame, skeletons_3d and parameters"""
        view(fig, list_parametros, variavel, x)

        key = cv2.waitKey(-1)  # Ele espera até que alguma
        if key == ord("q"):
            break

    result = obj.predict(X, y)

    string_list = y.to_list()
    float_list = [float(num) for num in string_list]

    r = [a == b for a, b in zip(float_list, result.to_list())]

    contador_false = r.count(False)

    print(f"Falses: {contador_false}")


if __name__ == "__main__":
    main()
