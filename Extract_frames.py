import os
import cv2
import numpy as np

def frames_de_video(entrada, caminho_saida):
    entrada_capturada = cv2.VideoCapture(entrada)
    cont = 0
    while entrada_capturada.isOpened():
        retorno, frame = entrada_capturada.read()

        if retorno:
            cv2.imwrite(os.path.join(caminho_saida, '%d.png') %cont, frame)
            cont +=1
        else:
            break
    cv2.destroyAllWindows()
    entrada_capturada.release()

def main():
    
    name_video = "p002g01c02.mp4"
    caminho_saida = ''

    frames_de_video(name_video, caminho_saida)

if __name__ == "__main__":
    main()

