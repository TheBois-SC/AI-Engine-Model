import torch, time
from Config.inference import DetectingFashion
from Config.initialization_model import (
    Init_Main_Model_PT,
    Init_Model_Segmentation_TF,
    Init_Wear_Model_PT
)

device = torch.device('cpu')
start_time = time.time()


def Main(base64_image: str):
    mainModel = Init_Main_Model_PT(device=device)
    wearModel = Init_Wear_Model_PT(device=device)
    tfModel = Init_Model_Segmentation_TF()

    fashion_result = DetectingFashion(
        base64_image=base64_image,
        model_main=mainModel,
        model_wear=wearModel,
        device=device,
        tf_model=tfModel)

    return fashion_result  # Return ID Kategori


if __name__ == "__main__":  # Example Use
    image64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAAcABwBAREA/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/APf6RmCqWYgKBkk9qAQwBBBB5BHelorivG2ts1o+nWkqgNxPJn/x0f1/KvLXvtQsHe0stQu7aAMCUhnZQWA5OAR3P6ewr034a66bzwvIL+7d54Lp4y88hZmGFYcn/ex+FWNW0fxa9xLJa6hZ3kDEkQyboSB/dGNwPHc4rl28I+MtTdlaCx09PWW43/kEX+ZFY+v/AAs8W2zwf2Nd2+oGQM07uqwCNsjAGSS2cnn2rtPh34R1TR/D88GviAXkl00gEW1xs2qBk465BrvqKKK//9k="
    print(f"Hasil Pengenalan Gambar: {Main(base64_image=image64)}")
    print("--- %s seconds ---" % (time.time() - start_time))
