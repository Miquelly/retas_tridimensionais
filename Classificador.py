from typing import Dict

import numpy as np
import pandas as pd
from is_msgs.image_pb2 import HumanKeypoints as HKP
from sklearn.linear_model import LinearRegression


class Gesture:
    def __init__(self):
        pass

    def extract_points(self, parts: Dict):
        p = np.zeros(3)

        if parts == None:
            p0_L = p
            p2_L = p
            p1_L = p
            p0_R = p
            p2_R = p
            p1_R = p

        else:
            p0_L = np.array(parts[1][HKP.Value("LEFT_SHOULDER")])  # Ombro
            p2_L = np.array(parts[1][HKP.Value("LEFT_WRIST")])  # pulso
            p1_L = np.array(parts[1][HKP.Value("LEFT_ELBOW")])  # cotovelo

            p0_R = np.array(parts[1][HKP.Value("RIGHT_SHOULDER")])
            p2_R = np.array(parts[1][HKP.Value("RIGHT_WRIST")])
            p1_R = np.array(parts[1][HKP.Value("RIGHT_ELBOW")])

        return p0_L, p1_L, p2_L, p0_R, p1_R, p2_R

    def distancia_euclidiana_2d(self, p1, p2):
        """
        Calcula a distância euclidiana entre dois pontos bidimensionais.

        Parâmetros:
        p1 -- um array ou lista contendo as coordenadas (x1, y1) do ponto 1
        p2 -- um array ou lista contendo as coordenadas (x2, y2) do ponto 2

        Retorna:
        A distância euclidiana entre os pontos.
        """
        p1 = np.array(p1)
        p2 = np.array(p2)

        distancia = np.linalg.norm(
            p2 - p1
        )  # Calcula a norma (distância) entre os pontos

        return distancia

    def colineares(self, p0, p1, p2):

        vetor1 = np.subtract(p1, p0)
        vetor2 = np.subtract(p2, p0)

        norm_vetor1 = np.linalg.norm(vetor1)
        norm_vetor2 = np.linalg.norm(vetor2)

        prod_interno = np.dot(vetor1, vetor2)

        sim = prod_interno / (norm_vetor1 * norm_vetor2)

        if sim > 1:
            sim = 1
        if sim < 0:
            sim = 0

        teta = (180 * np.arccos(sim)) / np.pi

        return teta

    def angulo_plano(self, diretor, normal):

        prod_interno = np.dot(diretor, normal)

        norma_d = np.linalg.norm(diretor)
        norma_n = np.linalg.norm(normal)

        alpha_ = np.arccos(
            prod_interno / (norma_d * norma_n)
        )  # (np.pi / 2) - np.arccos(prod_interno / (norma_d * norma_n))
        alpha = (180 * alpha_) / np.pi

        return alpha

    def classificador_for_plot(self, parts: Dict):

        p0_L, p1_L, p2_L, p0_R, p1_R, p2_R = self.extract_points(parts)

        limiar_distancia = 0.25
        limiar_colinear: int = 30

        pt1_L = p0_L[:2]  # Ombro
        pt2_L = p2_L[:2]  # pulso

        pt1_R = p0_R[:2]
        pt2_R = p2_R[:2]

        left_distancia = self.distancia_euclidiana_2d(pt1_L, pt2_L)
        right_distancia = self.distancia_euclidiana_2d(pt1_R, pt2_R)

        left_teta = self.colineares(p0_L, p1_L, p2_L)
        right_teta = self.colineares(p0_R, p1_R, p2_R)

        """ Arvore de Decisao """
        if left_distancia > limiar_distancia and right_distancia > limiar_distancia:
            if left_teta < limiar_colinear and right_teta < limiar_colinear:
                y = 3
            elif left_teta < limiar_colinear:
                y = 1
            elif right_teta < limiar_colinear:
                y = 2
            else:
                y = 0
        elif left_distancia > limiar_distancia:
            if left_teta < limiar_colinear:
                y = 1
            else:
                y = 0
        elif right_distancia > limiar_distancia:
            if right_teta < limiar_colinear:
                y = 2
            else:
                y = 0
        else:
            y = 0

        return (
            left_distancia,
            left_teta,
            right_distancia,
            right_teta,
            y,
        )

    def reta_geometria(self, parts: Dict):

        p0_L, p1_L, p2_L, p0_R, p1_R, p2_R = self.extract_points(parts)

        left_vector = p2_L - p0_L
        right_vector = p2_R - p0_R

        xL = [p2_L[0], left_vector[0]]
        yL = [p2_L[1], left_vector[1]]
        zL = [p2_L[2], left_vector[2]]

        xR = [p2_R[0], right_vector[0]]
        yR = [p2_R[1], right_vector[1]]
        zR = [p2_R[2], right_vector[2]]

        return xL, yL, zL, xR, yR, zR

    def predict(self, dfX: pd.DataFrame, dfy: pd.DataFrame):

        y: pd.Series = pd.Series()

        limiar_distancia = 0.25
        limiar_colinear: int = 50

        if dfX.index.max() == dfy.index.max():
            max = dfX.index.max()
        else:
            max = 0

        for x in dfX.index:

            npX = dfX.loc[x].to_numpy()

            # pt1_L = [x_LEFT_SHOULDER, y_LEFT_SHOULDER, z_LEFT_SHOULDER] -- [0:3]
            # pt2_L = [x_LEFT_WRIST, y_LEFT_WRIST, z_LEFT_WRIST] -- [12:15]

            # pt1_R = [x_RIGHT_SHOULDER, y_RIGHT_SHOULDER, z_RIGHT_SHOULDER] -- [3:6]
            # pt2_R = [x_RIGHT_WRIST, y_RIGHT_WRIST, z_RIGHT_WRIST] -- [15:18]

            pt1_L = npX[0:2]  # Ombro
            pt2_L = npX[12:14]  # pulso

            pt1_R = npX[3:5]
            pt2_R = npX[15:17]

            left_distancia = self.distancia_euclidiana_2d(pt1_L, pt2_L)
            right_distancia = self.distancia_euclidiana_2d(pt1_R, pt2_R)

            p0_L = npX[0:3]  # Ombro
            p2_L = npX[12:15]  # pulso
            p1_L = npX[6:9]  # cotovelo

            p0_R = npX[3:6]
            p2_R = npX[15:18]
            p1_R = npX[9:12]

            left_teta = self.colineares(p0_L, p1_L, p2_L)
            right_teta = self.colineares(p0_R, p1_R, p2_R)

            """ Árvore de decisão """
            if left_distancia > limiar_distancia and right_distancia > limiar_distancia:
                if left_teta < limiar_colinear and right_teta < limiar_colinear:
                    y.loc[x] = 3
                elif left_teta < limiar_colinear:
                    y.loc[x] = 1
                elif right_teta < limiar_colinear:
                    y.loc[x] = 2
                else:
                    y.loc[x] = 0
            elif left_distancia > limiar_distancia:
                if left_teta < limiar_colinear:
                    y.loc[x] = 1
                else:
                    y.loc[x] = 0
            elif right_distancia > limiar_distancia:
                if right_teta < limiar_colinear:
                    y.loc[x] = 2
                else:
                    y.loc[x] = 0
            else:
                y.loc[x] = 0

        return y

    def reta_regressao_projecao(self, parts: Dict):

        p0_L, p1_L, p2_L, p0_R, p1_R, p2_R = self.extract_points(parts)

        # Right
        xreg = np.array(
            [
                [p0_R[0]],
                [p1_R[0]],
                [p2_R[0]],
            ]
        )

        yreg = np.array(
            [
                [p0_R[1]],
                [p1_R[1]],
                [p2_R[1]],
            ]
        )

        zreg = np.array(
            [
                p0_R[2],
                p1_R[2],
                p2_R[2],
            ]
        )

        reg1 = LinearRegression().fit(xreg, zreg)
        reg2 = LinearRegression().fit(yreg, zreg)

        # Left
        xreg = np.array(
            [
                [p0_L[0]],
                [p1_L[0]],
                [p2_L[0]],
            ]
        )

        yreg = np.array(
            [
                [p0_L[1]],
                [p1_L[1]],
                [p2_L[1]],
            ]
        )

        zreg = np.array(
            [
                p0_L[2],
                p1_L[2],
                p2_L[2],
            ]
        )

        reg3 = LinearRegression().fit(xreg, zreg)
        reg4 = LinearRegression().fit(yreg, zreg)

        xL = [p2_L[0], (1 / reg3.coef_[0])]
        yL = [p2_L[1], (1 / reg4.coef_[0])]
        zL = [p2_L[2], 1]

        xR = [p2_R[0], (1 / reg1.coef_[0])]
        yR = [p2_R[1], (1 / reg2.coef_[0])]
        zR = [p2_R[2], 1]

        return xL, yL, zL, xR, yR, zR
