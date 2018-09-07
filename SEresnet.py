## Implementation for SE-Resnet
import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
from keras import backend as K
K.clear_session()
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Permute
from keras import backend as K

## Function for squeeze and excitation block
def squeeze_excite_block(inputX):
    ratio = 6
    filters = int(inputX.shape[-1])
    
    inputX_SE = GlobalAveragePooling2D()(inputX)
    inputX_SE = Reshape((1, 1, filters))(inputX_SE)

    inputX_SE = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal',use_bias=False)( inputX_SE)
    inputX_SE = Dense(filters, activation='sigmoid',kernel_initializer='he_normal', use_bias=False)( inputX_SE)

    outputX_SE = multiply([inputX, inputX_SE])
   
    return outputX_SE


## Fucntion for identity block
def identityBlock(inputX, filterSize, filters):
    Filters1, Filters2, Filters3 = filters 
    inputXcopy = inputX
    
    inputX = Conv2D(filters = Filters1, kernel_size = (1, 1), strides = (1,1), padding = 'valid',kernel_initializer = glorot_uniform(seed=0))(inputX)
    inputX =BatchNormalization(axis = 3)(inputX)
    inputX = Activation('relu')(inputX)
   
    inputX = Conv2D(filters = Filters2, kernel_size = (filterSize,filterSize), strides = (1,1), padding = 'same',kernel_initializer = glorot_uniform(seed=0))(inputX)
    inputX = BatchNormalization(axis = 3)(inputX)
    inputX = Activation('relu')(inputX)
   
    inputX = Conv2D(filters = Filters3, kernel_size = (1, 1), strides = (1,1), padding = 'valid',kernel_initializer = glorot_uniform(seed=0))(inputX)
    inputX = BatchNormalization(axis = 3)(inputX)
    
    inputX = squeeze_excite_block(inputX)
    
    inputX = layers.Add()([inputX, inputXcopy])
    inputX = Activation('relu')(inputX)
    return inputX


## Function for convolution block
def convolutionBlock(inputX, filterSize, filters, s=2):
    ############# Parameters ######################
    # inputX == Input tensor of shape (nSamples, nHeight, nWidth, nChannels)
    # filterSize
    # filters == Number of filters per layer
   
    Filters1, Filters2, Filters3 = filters
    
    inputXcopy = inputX
   
    inputX = Conv2D(filters = Filters1, kernel_size = (1, 1), strides = (s,s),kernel_initializer = glorot_uniform(seed=0))(inputX)
    inputX =BatchNormalization(axis = 3)(inputX)
    inputX = Activation('relu')(inputX)
    
    inputX = Conv2D(filters = Filters2, kernel_size = (filterSize,filterSize), padding = 'same',strides = (1,1),kernel_initializer = glorot_uniform(seed=0))(inputX)
    inputX = BatchNormalization(axis = 3)(inputX)
    inputX = Activation('relu')(inputX)
    
    inputX = Conv2D(filters = Filters3, kernel_size = (1, 1), strides = (1,1),kernel_initializer = glorot_uniform(seed=0))(inputX)
    inputX = BatchNormalization(axis = 3)(inputX)
    
    inputXcopy = Conv2D(filters = Filters3, kernel_size = (1, 1), strides = (s,s), kernel_initializer = glorot_uniform(seed=0))(inputXcopy)
    inputXcopy = BatchNormalization(axis = 3)(inputXcopy)
    
    print(inputX.shape, inputXcopy.shape)
    
    inputX = squeeze_excite_block(inputX)
   
    inputX = layers.Add()([inputX, inputXcopy])
    inputX = Activation('relu')(inputX)
    return inputX



## Function for Resnet50
def ResNet50(input_shape = (64, 64, 3), classes = 6):
        
        # Zero Padding(size = 3) --> Conv2D --> BatchNormalization
        
        X = Input(input_shape)
        
        inputX = ZeroPadding2D((3,3))(X)
        
        print(inputX.shape)
        
        ###################Changes  for MNIST Data ###############################3
        # Block 1 kernel size from 7 to 3
        # Block 3 s from 2 to 1
        ############################################################333
        
        # Block1
        inputX = Conv2D(64,  (3, 3), strides = (2, 2), kernel_initializer = glorot_uniform(seed=0))(inputX)
        # print("block1",inputX.shape)
        inputX = BatchNormalization(axis = 3)(inputX)
        inputX = Activation('relu')(inputX)
        #print(inputX.shape)
        inputX = MaxPooling2D((3, 3), strides=(2, 2))(inputX)
        #print(inputX.shape)
        
        #Block2
        inputX = convolutionBlock(inputX, filterSize = 3, filters = [64, 64, 256], s=1)
        #print("block2",inputX.shape)
        inputX = identityBlock(inputX, filterSize = 3, filters = [64, 64, 256])
        inputX = identityBlock(inputX, filterSize = 3, filters = [64, 64, 256])
        #print(inputX.shape)
        
        #Block3
        inputX = convolutionBlock(inputX, filterSize = 3, filters = [128, 128, 512], s = 1) ### Chnaged striding to 1)
        inputX = identityBlock(inputX, filterSize = 3, filters = [128, 128, 512])
        inputX = identityBlock(inputX, filterSize = 3, filters = [128, 128, 512])
        inputX = identityBlock(inputX, filterSize = 3, filters = [128, 128, 512])
        
        #Block4
        inputX = convolutionBlock(inputX, filterSize = 3, filters = [256, 256,  1024], s = 2)
        inputX = identityBlock(inputX, filterSize = 3, filters = [256, 256,  1024])
        inputX = identityBlock(inputX, filterSize = 3, filters = [256, 256,  1024])
        inputX = identityBlock(inputX, filterSize = 3, filters = [256, 256,  1024])
        inputX = identityBlock(inputX, filterSize = 3, filters = [256, 256,  1024])
        inputX = identityBlock(inputX, filterSize = 3, filters = [256, 256, 1024])
        
        #Blo512
        inputX = convolutionBlock(inputX, filterSize = 3, filters = [512, 512, 2048], s = 2)
        inputX = identityBlock(inputX, filterSize = 3, filters = [512, 512,  2048])
        inputX = identityBlock(inputX, filterSize = 3, filters = [512, 512,  2048])
   
        inputX = AveragePooling2D((2,2))(inputX)
        print(inputX.shape)
        # output layer
        inputX = Flatten()(inputX)
        inputX = Dense(classes, activation='softmax', kernel_initializer = glorot_uniform(seed=0))(inputX)
    
        # Create model
        model = Model(inputs = X, outputs = inputX)
        return model

## MNIST Dataset
from keras.datasets import mnist
from keras.utils import np_utils

nb_classes = 10

img_rows, img_cols = 28, 28

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

model = ResNet50(input_shape = (28, 28, 1), classes = 10)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs = 10, batch_size = 128)

preds = model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))



