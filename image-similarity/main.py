# couldnt run because of ssl error HTTPSConnectionPool(host='github.com', port=443):
# Max retries exceeded with url: /serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5 

import os

#local model path
path = "model/vgg_face_weights.h5"


from deepface import DeepFace

im1 = 'images/kiernan_shipka.png'
im2 = 'images/mckenna_grace.png'

os.environ["DEEPFACE_HOME"] = "model/vgg_face_weights.h5"

result = DeepFace.verify(im1, im2, model_name="VGG-Face")

# print(im1)
# result = DeepFace.verify(im1, im2)
# print(result)