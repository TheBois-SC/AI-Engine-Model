import torch
from Config.initialization_model import (
    Init_Main_Model_PT,
    Init_Model_Segmentation_TF,
    Init_Wear_Model_PT
)
from Config.Inference import DetectingFashion

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def Main(path_image: str):
    mainModel = Init_Main_Model_PT(device=device)
    wearModel = Init_Wear_Model_PT(device=device)
    tfModel = Init_Model_Segmentation_TF()

    fashion_result = DetectingFashion(
        path=path_image,
        model_main=mainModel,
        model_wear=wearModel,
        device=device,
        tf_model=tfModel)

    return fashion_result


if __name__ == "__main__":
    print(Main("./Gambar.jpg"))
