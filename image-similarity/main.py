# wont work on work-laptop due to persistent ssl error. 

import os

# to suppress flops rounfing error warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'

#local model path
path = "model/vgg_face_weights.h5"


from deepface import DeepFace

im1 = 'images/shipka.png'
im2 = 'images/grace.png'

result = DeepFace.verify(im1, im2, model_name="VGG-Face")

print(result)

# print(im1)
# result = DeepFace.verify(im1, im2)
# print(result)