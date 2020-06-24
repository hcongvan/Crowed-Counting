import tensorflow as tf

class FCRN(tf.keras.Model):
    def __init__(self, input, name='FCRN'):
        super(FCRN,self).__init__()
        self.x = tf.keras.layers.Input(input)
        self.block1 = self.convBlock(64,1)
        self.down1 = tf.keras.layers.MaxPool2D(name='max_pooling_1')
        self.block2 = self.convBlock(128,2)
        self.down2 = tf.keras.layers.MaxPool2D(name='max_pooling_2')
        self.block3 = self.convBlock(256,3)
        self.down3 = tf.keras.layers.MaxPool2D(name='max_pooling_3')
        self.block4 = self.convBlock(512,4)
        self.up1 = tf.keras.layers.Conv2DTranspose(
            filters=256,
            kernel_size=3,
            strides=(2,2),
            padding='same',
            data_format="channels_last",
            activation=None,
            name='conv2d_transpose_1'
        )
        self.block5 = self.convBlock(256,5)
        self.up2 = tf.keras.layers.Conv2DTranspose(
            filters=128,
            kernel_size=3,
            strides=(2,2),
            padding='same',
            data_format="channels_last",
            activation=None,
            name='conv2d_transpose_2'
        )
        self.block6 = self.convBlock(128,6)
        self.up3 = tf.keras.layers.Conv2DTranspose(
            filters=64,
            kernel_size=3,
            strides=(2,2),
            padding='same',
            data_format="channels_last",
            activation=None,
            name='conv2d_transpose_3'
        )
        self.block7 = self.convBlock(64,7)
        self.conv_last = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=1,
            strides=1,
            padding='same',
            data_format="channels_last",
            name='conv2d_last'
        )
        
    def convBlock(self,filter,index):
        block = tf.keras.Sequential([tf.keras.layers.Conv2D(
            filters=filter,
            kernel_size=3,
            strides=1,
            padding='same',
            data_format="channels_last",
            activation='relu',
            kernel_initializer='VarianceScaling',
            bias_initializer='TruncatedNormal',
            name='conv2d_{}_{}'.format(index,i)
        ) for i in range(3)])
        
        return block

    def call(self,x,train=True):
        block1 = self.block1(x)
        pool1 = self.down1(block1)
        block2 = self.block2(pool1)
        pool2 = self.down2(block2)
        block3 = self.block3(pool2)
        pool3 = self.down3(block3)
        block4 = self.block4(pool3)
        up1 = self.up1(block4)
        tmp1 = tf.keras.layers.concatenate([block3,up1],axis=3)
        block5 = self.block5(tmp1)
        up2 = self.up2(block5)
        tmp2 = tf.keras.layers.concatenate([block2,up2],axis=3)
        block6 = self.block6(tmp2)
        up3 = self.up3(block6)
        tmp3 = tf.keras.layers.concatenate([block1,up3],axis=3)
        block7 = self.block7(tmp3)
        out = self.conv_last(block7)
        return out
