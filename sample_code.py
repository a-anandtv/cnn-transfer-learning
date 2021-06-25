from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from numpy import expand_dims
from matplotlib import pyplot

# Set the model
model = VGG16()

# View the model
model.summary()

# Visualize the filters
filters, biases = model.layers[1].get_weights()

print (filters.shape)

# Normalize filters
filters_min, filters_max = filters.min(), filters.max()
filters = (filters - filters_min) / (filters_max - filters_min)

# View the filters
n_filters = 10
ith = 1
for i in range(n_filters):
    f = filters[:, :, :, i]
    for j in range(3):
        aplot = pyplot.subplot(n_filters, 3, ith)
        aplot.set_xticks([])
        aplot.set_yticks([])
        pyplot.imshow(f[:, :, j], cmap="gray")
        ith += 1

pyplot.show()

# Modify the model to have the first CNN layer
model = Model(inputs=model.inputs, outputs=model.layers[1].output)

model.summary()

# Load images
bird = load_img("../visualize_filters/test_data/bird.jpeg", target_size=(224, 224))
elephant = load_img("../visualize_filters/test_data/elephant.jpeg", target_size=(224, 224))
hiccup = load_img("../visualize_filters/test_data/hiccup.jpg", target_size=(224, 224))
two_subjects = load_img("../visualize_filters/test_data/2subjects.jpg", target_size=(224, 224))

# Preprocess images and View feature filters
# Bird

# View image
pyplot.imshow(bird)
pyplot.show()

bird = img_to_array(bird)
bird = expand_dims(bird, axis=0)
bird = preprocess_input(bird)

features = model.predict(bird)

filt_dim = 8
ith = 1

for i in range(filt_dim):
  for j in range(filt_dim):
    aplt = pyplot.subplot(filt_dim, filt_dim, ith)
    aplt.set_xticks([])
    aplt.set_yticks([])
    pyplot.imshow(features[0,:,:,ith-1], cmap="gray")
    ith += 1

pyplot.show()

# Elephant

# View image
pyplot.imshow(elephant)
pyplot.show()

elephant = img_to_array(elephant)
elephant = expand_dims(elephant, axis=0)
elephant = preprocess_input(elephant)

features = model.predict(elephant)

filt_dim = 8
ith = 1

for i in range(filt_dim):
  for j in range(filt_dim):
    aplt = pyplot.subplot(filt_dim, filt_dim, ith)
    aplt.set_xticks([])
    aplt.set_yticks([])
    pyplot.imshow(features[0,:,:,ith-1], cmap="gray")
    ith += 1

pyplot.show()

# Hiccup

# View image
pyplot.imshow(hiccup)
pyplot.show()

hiccup = img_to_array(hiccup)
hiccup = expand_dims(hiccup, axis=0)
hiccup = preprocess_input(hiccup)

features = model.predict(hiccup)

filt_dim = 8
ith = 1

for i in range(filt_dim):
  for j in range(filt_dim):
    aplt = pyplot.subplot(filt_dim, filt_dim, ith)
    aplt.set_xticks([])
    aplt.set_yticks([])
    pyplot.imshow(features[0,:,:,ith-1], cmap="gray")
    ith += 1

pyplot.show()

# 2 subjects

# View image
pyplot.imshow(two_subjects)
pyplot.show()

two_subjects = img_to_array(two_subjects)
two_subjects = expand_dims(two_subjects, axis=0)
two_subjects = preprocess_input(two_subjects)

features = model.predict(two_subjects)

filt_dim = 8
ith = 1

for i in range(filt_dim):
  for j in range(filt_dim):
    aplt = pyplot.subplot(filt_dim, filt_dim, ith)
    aplt.set_xticks([])
    aplt.set_yticks([])
    pyplot.imshow(features[0,:,:,ith-1], cmap="gray")
    ith += 1

pyplot.show()