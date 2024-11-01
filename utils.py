import cv2
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import imutils
import numpy as np

links = [
    "LEFT_SHOULDER",
    "LEFT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_SHOULDER",
    "RIGHT_ELBOW",
    "RIGHT_WRIST",
]

a = [
    "x_",
    "y_",
    "z_",
]


def a():
    # Generating matplotlib figure and save it as a png image
    plt.plot([0, 5], [0, 5])
    plt.savefig("plot.png")

    # Reading figure saved as png image
    img_plot = cv2.imread("plot.png")

    # Displaying image using opencv imshow
    cv2.imshow("Image", img_plot)

    # Waiting for any key to terminate image window
    cv2.waitKey(0)


##################################
# https://pt.stackoverflow.com/questions/399817/como-fazer-unificar-as-janelas-de-exibi%C3%A7%C3%A3o


def b():
    # Suponha que data é obtido corretamente
    view_3d = ""

    # Carregar a imagem
    imagem = cv2.imread("4.png")

    # Redimensionar a imagem para que tenha a mesma altura que view_3d
    altura_view_3d = view_3d.shape[0]
    imagem_redimensionada = imutils.resize(imagem, height=altura_view_3d)

    # Opcional: Redimensionar view_3d para a mesma largura que imagem_redimensionada
    # view_3d = cv2.resize(view_3d, (imagem_redimensionada.shape[1], altura_view_3d))

    # Exibir a imagem redimensionada
    cv2.imshow("Original", imagem_redimensionada)

    # Exibir as duas imagens lado a lado
    print(imagem.shape, imagem_redimensionada.shape, view_3d.shape)

    # Concatenar as imagens
    cv2.imshow("ESQUERDA/DIREITA", np.hstack([imagem_redimensionada, view_3d]))

    # Aguarda uma tecla para fechar as janelas
    cv2.waitKey(0)
    cv2.destroyAllWindows()


######################################
def c():

    COLUNAS = [
        "ID",
        "Gesture",
    ]

    df = pd.DataFrame(columns=COLUNAS)

    for i in range(5):
        df = df._append({"ID": i, "Gesture": i * 2}, ignore_index=True)

        # temp_df = pd.DataFrame({'ID': [i], 'Gesture': [i * 2]})
        # # Concatenar o DataFrame temporário ao DataFrame principal
        # df = pd.concat([df, temp_df], ignore_index=True)

    print(df)
    df.to_csv("your_file_name.csv", index=False)
    df.info()


def d():

    chunksize = 10

    name_df = "p002g01c02"

    reader = pd.read_csv(
        f"{name_df}.csv",
        chunksize=chunksize,
        dtype={
            "Frame": int,
            "ID": int,
            "Gesture": "category",
        },
    )

    links = [
        ("LEFT_SHOULDER", "LEFT_ELBOW"),
        ("LEFT_ELBOW", "LEFT_WRIST"),
        ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
        ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ]

    for chunk in reader:

        print(chunk.head(10))

        #     parts[part.id] = (part.position.x, part.position.y, part.position.z)
        for parts in links:
            begin, end = parts

            x_pair = [chunk[f"x_{begin}"].iloc[0], chunk[f"x_{end}"].iloc[0]]
            y_pair = [chunk[f"y_{begin}"].iloc[0], chunk[f"y_{end}"].iloc[0]]
            z_pair = [chunk[f"z_{begin}"].iloc[0], chunk[f"z_{end}"].iloc[0]]

            # color = _id_to_rgb_color(id=skeleton.id)
            # ax.plot(
            #     x_pair,
            #     y_pair,
            #     zs=z_pair,
            #     linewidth=3,
            #     color=color,
            # )

        break


def e():

    # Exemplo de dicionário
    dados = {
        "Nome": ["Ana", "Carlos", "João", "Maria"],
        "Idade": [23, 34, 45, 29],
        "Cidade": ["São Paulo", "Rio de Janeiro", "Belo Horizonte", "Curitiba"],
    }

    oi = {}

    if len(oi.keys()) == 0:
        print("tese")
    print(dados["Nome"])

    # Criando o DataFrame a partir do dicionário
    df = pd.DataFrame(dados)

    # Exibindo o DataFrame
    print(df)
    print(df.index.max())


def f():

    # Criando uma série vazia
    serie_vazia = pd.Series(dtype="float64")

    # Adicionando elementos usando indexação direta
    serie_vazia[0] = 1.5
    serie_vazia[1] = 2.5
    serie_vazia[2] = 3.5

    # Exibindo a série após a adição
    print(serie_vazia[0])

    a = serie_vazia.to_numpy()
    print(a)


def g():
    p = np.zeros((6, 3))
    print(p)


def main():
    g()


if __name__ == "__main__":
    main()
