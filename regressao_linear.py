import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imutils
import cv2


class Regressao:
    def __init__(self):
        pass

    def media(self, lista: list):

        somatorio = np.sum(lista)

        M_ = 1 / len(lista) * somatorio

        return M_

    def desvio_padrao(self, lista: list):

        M_ = self.media(lista)

        soma = 0

        for elemento in lista:

            soma += (elemento - M_) ** 2

        S = np.sqrt((1 / len(lista)) * soma)

        return S

    def correlacao(self, listax: list, listay):

        Mx = self.media(listax)
        My = self.media(listay)

        Sx = self.desvio_padrao(listax)
        Sy = self.desvio_padrao(listay)

        if len(listax) == len(listay):
            n = len(listax)
        else:
            n = 0
            return

        soma = 0

        for elemento in range(n):
            x = (listax[elemento] - Mx) / Sx
            y = (listay[elemento] - My) / Sy

            soma += x * y

        r = (1 / n) * soma

        return r

    def regressao_simples(self, listx: list, listy: list):

        r = self.correlacao(listx, listy)

        Sy = self.desvio_padrao(listx)
        Sx = self.desvio_padrao(listy)

        inclinacao = (r * Sy) / Sx

        y_ = self.media(listy)
        x_ = self.media(listx)

        intercepto = y_ - (inclinacao * x_)

        print(f"Reta: {inclinacao:.2}x + {intercepto:.2}")

        return inclinacao, intercepto

    def regressao_multipla(self, listx: list, listy: list, listz: list):

        x_ = self.media(listx)
        y_ = self.media(listy)
        z_ = self.media(listz)

        if len(listx) == len(listy) and len(listy) == len(listz):
            n = len(listx)
        else:
            n = 0

        x_y = 0
        x = 0
        y = 0
        z = 0

        for elemento in range(n):

            x_y += (listx[elemento] - x_) * (listy[elemento] - y_)
            x2 = 0
            x += listx[elemento] - x_
            y += listy[elemento] - y_
            z += listz[elemento] - z_

    def plot(self, listx: list, listy: list, listz: list, px, py, pz):

        inclinacao, intercepto = self.regressao_simples(listx, listy)

        # print(f"Reta: {inclinacao:.2}x + {intercepto:.2}")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        plt.ioff()

        # fig, ax = plt.subplots()

        fig.suptitle("Regressão", fontsize=10)

        ax.clear()
        ax.set_title("Gráfico de Pontos")
        ax.view_init(azim=0, elev=20)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_zlim(-0.25, 2)
        ax.set_xticks(np.arange(-4, 4, 1))
        ax.set_yticks(np.arange(-4, 4, 1))
        ax.set_zticks(np.arange(0, 2, 0.5))
        ax.set_xlabel("X", labelpad=20)
        ax.set_ylabel("Y", labelpad=10)
        ax.set_zlabel("Z", labelpad=5)

        ax.scatter(listx, listy, listz, color="blue")

        ax.plot(px, py, pz, label=f"Reta: {inclinacao:.2}x + {intercepto:.2}")
        ax.legend()

        plt.show()


def main():
    obj = Regressao()

    listx = [1, 2, 3]
    listy = [0, 1, 2]
    listz = [0, 0, 0]

    px = [1, 2]
    py = [0, 1]
    pz = [0, 0]

    dados = {
        "Mes": [1, 2, 3],
        "Venda": [1, 2, 3],
        "Arrecadacao": [1, 1, 2],
    }
    df = pd.DataFrame(dados)

    print(df)
    print(df.index.max())

    # obj.regressao_simples(df["Venda"], df["Arrecadacao"])

    obj.plot(df["Venda"], df["Arrecadacao"], listz, px, py, pz)


if __name__ == "__main__":
    main()
