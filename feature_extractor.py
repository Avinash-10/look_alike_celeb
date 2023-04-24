from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras.utils.layer_utils import get_source_inputs
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from tqdm import tqdm

filename = pickle.load(open('filename.pkl','rb'))

model = VGGFace(model='resnet50', include_top=False, input_shape=(224,224,3), pooling='avg ')

def feature_extract(image_path, model):
    img = image.load_img(image_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_image = np.expand_dims(img_array,axis=0)
    preprocessed_img = preprocess_input(expanded_image)

    result = model.predict(preprocessed_img).flatten()

    return result

feature = []

for file in tqdm(filename):
    feature.append(feature_extract(file,model))

pickle.dump(feature, open('embedding.pkl','wb'))