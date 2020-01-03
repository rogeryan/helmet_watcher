# training parameters
EPOCHS = 50
BATCH_SIZE = 8
NUM_CLASSES = 20
IMAGE_HEIGHT = 300
IMAGE_WIDTH = 300
CHANNELS = 3

# When the iou value of the anchor and the real box is less than the IoU_threshold,
# the anchor is divided into negative classes, otherwise positive.
IoU_threshold = 0.6

# generate anchor
SIZES = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
RATIOS = [[1, 2, 0.5]] * 5

background_sample_num = 128

# focal loss
alpha = 0.25
gamma = 2

cls_loss_weight = 0.5
reg_loss_weight = 0.5

# dataset
PASCAL_VOC_DIR = "/home/aistudio/data/helmet/VOC2028"
OBJECT_CLASSES = {"hat": 1, "dog": 2, "person": 3}
# (0, 1)
train_ratio = 0.8


# directory of saving model
save_model_dir = "saved_model"
