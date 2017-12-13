"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this. 

Usage:
       python CapsNet.py
       python CapsNet.py --epochs 50
       python CapsNet.py --epochs 50 --num_routing 3
       ... ...
       
Result:
    Validation accuracy > 99.5% after 20 epochs. Converge to 99.66% after 50 epochs.
    About 110 seconds per epoch on a single GTX1070 GPU card
    
Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""
from keras.models import Sequential
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask

from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
import cv2
import math
from scipy import ndimage
from sklearn.datasets.base import _pkl_filepath

K.set_image_data_format('channels_last')
src_path = "./dataset/"

def CapsNet(input_shape, n_class, num_routing, batch_size):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param num_routing: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, batch_size=batch_size, num_routing=num_routing,
                             name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])
    return train_model, eval_model


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def train(model, data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=args.debug)
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (0.9 ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})

    """"""
    # Training without data augmentation:
    model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])
    

    # Begin: Training with data augmentation ---------------------------------------------------------------------#
    def train_generator(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
                        steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                        epochs=args.epochs,
                        validation_data=[[x_test, y_test], [y_test, x_test]],
                        callbacks=[log, tb, checkpoint, lr_decay])
    # End: Training with data augmentation -----------------------------------------------------------------------#

    model.save_weights(args.save_dir) #'/trained_model.h5'
    print('Trained model saved to \'%s\'' % (args.save_dir))
    
#     # save cnn
#     json_string = model.to_json()
#      
#     with open(args.save_dir + '/trained_model.json','w') as json_file :
#         json_file.write(json_string)

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model


def test(model, data):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=100)
    print('-'*50)
    print('Test acc:', np.max(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])

    import matplotlib.pyplot as plt
    from utils import combine_images
    from PIL import Image

    img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save("real_and_recon.png")
    print()
    print('Reconstructed images are saved to ./real_and_recon.png')
    print('-'*50)
    plt.imshow(plt.imread("real_and_recon.png", ))
    plt.show()


def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)


def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty


def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

def estimation(filename, rSize, isFixSizeRatio, pLocal):
    # read the image
    gray = cv2.imread(src_path+filename) 
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    if isFixSizeRatio == True:
        rows,cols = gray.shape
        if rows > cols:
            factor = rSize/rows
            rows = int(rSize)
            cols = int(round(cols*factor))
            gray = cv2.resize((255-gray), (cols,rows))
        else:
            factor = rSize/cols
            rows = int(round(rows*factor))
            cols = int(rSize)
            gray = cv2.resize((255-gray), (cols, rows))
    else:
        rows,cols = rSize, rSize
        gray = cv2.resize((255-gray), (cols, rows))

    if pLocal == 0:
        colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
        rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))

        gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')

        shiftx,shifty = getBestShift(gray)
        shifted = shift(gray,shiftx,shifty)
        gray = shifted

    elif pLocal == 1:
        colsPadding = (0,int(math.floor((28-cols))))
        rowsPadding = (0,int(math.floor((28-rows))))
        gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')
    
    elif pLocal == 2:
        colsPadding = (int(math.floor((28-cols))),0)
        rowsPadding = (0,int(math.floor((28-rows))))
        gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')

    elif pLocal == 3:
        colsPadding = (0,int(math.floor((28-cols))))
        rowsPadding = (int(math.floor((28-rows))),0)
        gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')

    elif pLocal == 4:
        colsPadding = (int(math.floor((28-cols))),0)
        rowsPadding = (int(math.floor((28-rows))),0)
        gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')

    (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    flatten = (gray.flatten()/255.0) #()-255 #255.0
    return (np.asfarray(flatten).reshape((1,28,28,1)))

def CompetitionTest(filepath, mSize, isFixSizeRatio, pLocal, isEachSizeAcc):
    file_list = os.listdir(filepath)
    file_list.sort()
    
    chk_answer = {}
    if isEachSizeAcc == False:
        chk_answer[0] = {"s" : 0, "f" : 0, "name" : []}
    for size in mSize:
        chk_answer[size] = {"s" : 0, "f" : 0, "name" : []}
    
    for filename in file_list:
        sumResult = []
        answer = filename.split('_')[1].split('.')[0]
        for size in mSize:
            y_pred, x_recon = eval_model.predict(estimation(filename, size, isFixSizeRatio, pLocal) , batch_size=1, verbose=1)
            chk_answer[size]["s" if str(np.argmax(y_pred)) == str(answer) else "f"] += 1
            if(str(np.argmax(y_pred)) != str(answer)):
                chk_answer[size]["name"].append(filename)
            
            for row in y_pred:
                sumResult.append(row)
                
            print(u"FileName: %s 예측: %s 정답: %s" % (filename,str(np.argmax(y_pred)), answer))
            print(y_pred)

        if isEachSizeAcc == False:
            sum_y_pred = np.sum(sumResult, axis=0)
            chk_answer[0]["s" if str(np.argmax(sum_y_pred)) == str(answer) else "f"] += 1
            if(str(np.argmax(sum_y_pred)) != str(answer)):
                chk_answer[0]["name"].append(filename)
            print(u"FileName: %s 예측: %s 정답: %s" % (filename,str(np.argmax(sum_y_pred)), answer))
            print(sum_y_pred)
    
    last_answer = 0
    if isEachSizeAcc == False:
        mSize = [0]
    for chk_Val in mSize:
        defualtName = chk_answer[chk_Val]
        percent = defualtName["s"] /len(file_list) * 100
        last_answer += defualtName["s"]
        print(u"All: %s, True: %s, False: %s, TrueRatio: %s, FailFiles: %s" % (str(len(file_list)),defualtName["s"],defualtName["f"],str(round(percent, 2)) + "%",defualtName["name"]))
    
    percent = last_answer / (len(file_list)*len(mSize)) * 100
    print(u"최종 %s per" % str(round(percent, 2)))
  

if __name__ == "__main__":
    import numpy as np
    import os
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks
    from keras.utils.vis_utils import plot_model

    # setting the hyper parameters
    # C:\Users\jinhak>jupyter notebook
    # python capsulenet_test.py --batch_size=1000 --epochs=1 --lam_recon=0.392 --num_routing=2 --shift_fraction=0.2 --debug=1 --save_dir='./result/trained_model_test.h5' --is_training=1 --weights='/trained_model.h5' --lr=0.001
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int) #1
    parser.add_argument('--epochs', default=1, type=int) #50
    parser.add_argument('--lam_recon', default=0.392, type=float)  # 784 * 0.0005, paper uses sum of SE, here uses MSE 0.392
    parser.add_argument('--num_routing', default=2, type=int)  # num_routing should > 0  3
    parser.add_argument('--shift_fraction', default=0.1, type=float) #0.1
    parser.add_argument('--debug', default=0, type=int)  # debug>0 will save weights by TensorBoard
    parser.add_argument('--save_dir', default='./result/trained_model_test.h5')
    parser.add_argument('--is_training', default=0, type=int) #1
    parser.add_argument('--weights', default='./result/trained_20171127.h5') #None /trained_model.h5  /trained_20171127.h5
    parser.add_argument('--lr', default=0.001, type=float)
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # define model
    model, eval_model = CapsNet(input_shape=x_train.shape[1:],
                                n_class=len(np.unique(np.argmax(y_train, 1))),
                                num_routing=args.num_routing,
                                batch_size=args.batch_size)
    model.summary()
    plot_model(model, to_file=args.save_dir+'/model.png', show_shapes=True)
    
    # train or test
    if args.is_training == 0:  # init the model weights with provided one
        model.load_weights(args.weights)

    if args.is_training:
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        #test(model=eval_model, data=(x_test, y_test))
       
        #eval_model.load_weights(args.save_dir + '/trained_model.h5')
        while True:
            filename = input("Image 파일명을 입력하세요 (형식 xxxx.jpg): ")
            filepath = src_path
            file_list = os.listdir(filepath)
            file_list.sort()
             
            mSize = np.array([22]) #22 , 17,18,19,20,21,22,23,24,25
            isFixSizeRatio = True #Img Size 비율 유지:True
            pLocal = 0 #0:Center, 1:LeftTop, 2:RightTop, 3:LeftBottom, 4:RightBottom
            isEachSizeAcc = False #Size 통합  Acc : False
            if filename == "test":
                CompetitionTest(src_path, mSize, isFixSizeRatio, pLocal, isEachSizeAcc)
                
            elif filename in file_list :                
                fig = plt.figure()        
                i = 0
                sumResult = []
                for size in np.sort(mSize):
                    y_pred, x_recon = eval_model.predict(estimation(filename, size, isFixSizeRatio, pLocal) , batch_size=1, verbose=1) #estimation(filename, size) verbose=0
                    print("Size별 예측  y 라벨은?", np.argmax(y_pred,1), y_pred, size)

                    subplot = fig.add_subplot(len(mSize),3,1+i)
                    subplot.set_xticks(list(range(10)))
                    subplot.set_ylim(0,1)
                    subplot.bar(list(range(10)), y_pred[0], align='center')
             
                    subplot1 = fig.add_subplot(len(mSize),3,2+i)
                    subplot1.imshow(estimation(filename, size, isFixSizeRatio, pLocal).reshape(28,28), cmap='Greys') #,interpolation = 'nearest')
                    
                    subplot2 = fig.add_subplot(len(mSize),3,3+i)
                    subplot2.imshow(x_recon.reshape(28,28), cmap='Greys') #,interpolation = 'nearest')
                    
                    for row in y_pred:
                        sumResult.append(row)
                        
                    i = i + 3
                    #plt.title("손글씨 결과 ")
                    print(sumResult)
                print("최종 예측은?", np.argmax(np.sum(sumResult, axis=0)), np.sum(sumResult, axis=0))
                plt.show()
            else : 
                print("다시 입력")
        
            if input("Continue? (y/n) : ") in ("n","N"):
                break

    

