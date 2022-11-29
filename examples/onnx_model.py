from carla.Inference import OnnxModel
import cv2

if __name__ == "__main__":
    # initialization of the model should be done once
    model_path = "../carla/yolox_s.onnx"
    model = OnnxModel(model_path)

    # load the image into memory
    img_path = "../test_imgs/test2.png"
    img = cv2.imread(img_path)

    # preprocess and then predict
    preprocessed_img = model.preprocess(img)
    prediction = model.detect_sign(preprocessed_img)

    print(f"prediction is :{prediction[0]}")
    print(f"confidence is :{prediction[1]}")

