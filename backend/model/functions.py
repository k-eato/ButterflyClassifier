import numpy as np
import tensorflow as tf

# Define variables
IMG_SHAPE = [224, 224, 3]
CLASS_NAMES = [
    "ADONIS",
    "AFRICAN GIANT SWALLOWTAIL",
    "AMERICAN SNOOT",
    "AN 88",
    "APPOLLO",
    "ATALA",
    "BANDED ORANGE HELICONIAN",
    "BANDED PEACOCK",
    "BECKERS WHITE",
    "BLACK HAIRSTREAK",
    "BLUE MORPHO",
    "BLUE SPOTTED CROW",
    "BROWN SIPROETA",
    "CABBAGE WHITE",
    "CAIRNS BIRDWING",
    "CHECQUERED SKIPPER",
    "CHESTNUT",
    "CLEOPATRA",
    "CLODIUS PARNASSIAN",
    "CLOUDED SULPHUR",
    "COMMON BANDED AWL",
    "COMMON WOOD-NYMPH",
    "COPPER TAIL",
    "CRECENT",
    "CRIMSON PATCH",
    "DANAID EGGFLY",
    "EASTERN COMA",
    "EASTERN DAPPLE WHITE",
    "EASTERN PINE ELFIN",
    "ELBOWED PIERROT",
    "GOLD BANDED",
    "GREAT EGGFLY",
    "GREAT JAY",
    "GREEN CELLED CATTLEHEART",
    "GREY HAIRSTREAK",
    "INDRA SWALLOW",
    "IPHICLUS SISTER",
    "JULIA",
    "LARGE MARBLE",
    "MALACHITE",
    "MANGROVE SKIPPER",
    "MESTRA",
    "METALMARK",
    "MILBERTS TORTOISESHELL",
    "MONARCH",
    "MOURNING CLOAK",
    "ORANGE OAKLEAF",
    "ORANGE TIP",
    "ORCHARD SWALLOW",
    "PAINTED LADY",
    "PAPER KITE",
    "PEACOCK",
    "PINE WHITE",
    "PIPEVINE SWALLOW",
    "POPINJAY",
    "PURPLE HAIRSTREAK",
    "PURPLISH COPPER",
    "QUESTION MARK",
    "RED ADMIRAL",
    "RED CRACKER",
    "RED POSTMAN",
    "RED SPOTTED PURPLE",
    "SCARCE SWALLOW",
    "SILVER SPOT SKIPPER",
    "SLEEPY ORANGE",
    "SOOTYWING",
    "SOUTHERN DOGFACE",
    "STRAITED QUEEN",
    "TROPICAL LEAFWING",
    "TWO BARRED FLASHER",
    "ULYSES",
    "VICEROY",
    "WOOD SATYR",
    "YELLOW SWALLOW TAIL",
    "ZEBRA LONG WING",
]


# Build model from pretrained MobileNet
base_model = tf.keras.applications.MobileNetV3Large(
    input_shape=IMG_SHAPE, include_top=False, weights="imagenet"
)
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
dense_layer = tf.keras.layers.Dense(150)
prediction_layer = tf.keras.layers.Dense(75)
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.25)(x)
x = dense_layer(x)
x = tf.keras.layers.Dropout(0.25)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

# Loads the weights
checkpoint_path = "model/training_1/cp.ckpt"
model.load_weights(checkpoint_path)


# Evaluate image and retrieve predicted class
def inference(image):
    image = tf.image.resize(image, [224, 224], antialias=True)
    image = tf.reshape(image, [1, 224, 224, 3])
    prediction = model.predict(image)
    max_index = np.argmax(prediction)
    butterfly_name = CLASS_NAMES[max_index]
    return butterfly_name
