from ultralytics import YOLO
import os

if __name__ == "__main__":
    # Config path
    CONFIG_PATH = "./config/v8-nc1-custombottleneck2.yaml"
    #CONFIG_PATH = "./config/v8-nc1-addconvc2fdetect.yaml"
    #CONFIG_PATH = "./config/v8-nc1-addconvc2fdetectwithcbn.yaml"
    # Output path
    OUTPUT_PATH = "/hy-tmp/yolo-results-300e/"
    # Dataset path
    DATASET_PATH = "../datasets/Shuttlecock/dataset.yaml"
    # Load the model
    model = YOLO("./config/v8-nc1-custombottleneck2.yaml")
    #model = YOLO("./config/v8-nc1-addconvc2fdetect.yaml")
    #model = YOLO("./config/v8-nc1-addconvc2fdetectwithcbn.yaml")
    # Train the model
    model.train(data=DATASET_PATH, epochs=300, project=OUTPUT_PATH, name=os.path.splitext(os.path.basename(CONFIG_PATH))[0], batch=16)  # custombottleneck2
    #model.train(data=DATASET_PATH, epochs=300, project=OUTPUT_PATH, name=os.path.splitext(os.path.basename(CONFIG_PATH))[0], batch=128)  # addconvc2fdetect
    #model.train(data=DATASET_PATH, epochs=300, project=OUTPUT_PATH, name=os.path.splitext(os.path.basename(CONFIG_PATH))[0], batch=48)  # addconvc2fdetectwithcbn
