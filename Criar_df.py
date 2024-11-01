import pandas as pd


def Criar_DataFrame(name: str):

    COLUNAS = [
        "Frame",
        "ID",
        "Gesture",
        "x_LEFT_SHOULDER",
        "y_LEFT_SHOULDER",
        "z_LEFT_SHOULDER",
        "x_RIGHT_SHOULDER",
        "y_RIGHT_SHOULDER",
        "z_RIGHT_SHOULDER",
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
        "x_LEFT_ELBOW",
        "y_LEFT_ELBOW",
        "z_LEFT_ELBOW",
        "x_RIGHT_ELBOW",
        "y_RIGHT_ELBOW",
        "z_RIGHT_ELBOW",
        "x_LEFT_WRIST",
        "y_LEFT_WRIST",
        "z_LEFT_WRIST",
        "x_RIGHT_WRIST",
        "y_RIGHT_WRIST",
        "z_RIGHT_WRIST",
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
        "x_NOSE",
        "y_NOSE",
        "z_NOSE",
        "x_LEFT_EYE",
        "y_LEFT_EYE",
        "z_LEFT_EYE",
        "x_RIGHT_EYE",
        "y_RIGHT_EYE",
        "z_RIGHT_EYE",
    ]

    df = pd.DataFrame(columns=COLUNAS)

    df.to_csv(f"{name}.csv", index=False)

    return df
