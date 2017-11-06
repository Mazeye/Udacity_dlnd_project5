
# coding: utf-8

# # Face Generation
# In this project, you'll use generative adversarial networks to generate new images of faces.
# ### Get the Data
# You'll be using two datasets in this project:
# - MNIST
# - CelebA
# 
# Since the celebA dataset is complex and you're doing GANs in a project for the first time, we want you to test your neural network on MNIST before CelebA.  Running the GANs on MNIST will allow you to see how well your model trains sooner.
# 
# If you're using [FloydHub](https://www.floydhub.com/), set `data_dir` to "/input" and use the [FloydHub data ID](http://docs.floydhub.com/home/using_datasets/) "R5KrjnANiKVhLWAkpXhNBe".

# In[1]:


data_dir = './data'

# FloydHub - Use with data ID "R5KrjnANiKVhLWAkpXhNBe"
#data_dir = '/input'


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper

helper.download_extract('mnist', data_dir)
helper.download_extract('celeba', data_dir)


# ## Explore the Data
# ### MNIST
# As you're aware, the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset contains images of handwritten digits. You can view the first number of examples by changing `show_n_images`. 

# In[2]:


show_n_images = 25

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
get_ipython().magic('matplotlib inline')
import os
from glob import glob
from matplotlib import pyplot

mnist_images = helper.get_batch(glob(os.path.join(data_dir, 'mnist/*.jpg'))[:show_n_images], 28, 28, 'L')
pyplot.imshow(helper.images_square_grid(mnist_images, 'L'), cmap='gray')


# ### CelebA
# The [CelebFaces Attributes Dataset (CelebA)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset contains over 200,000 celebrity images with annotations.  Since you're going to be generating faces, you won't need the annotations.  You can view the first number of examples by changing `show_n_images`.

# In[3]:


show_n_images = 9

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
mnist_images = helper.get_batch(glob(os.path.join(data_dir, 'img_align_celeba/*.jpg'))[:show_n_images], 28, 28, 'RGB')
pyplot.imshow(helper.images_square_grid(mnist_images, 'RGB'))


# ## Preprocess the Data
# Since the project's main focus is on building the GANs, we'll preprocess the data for you.  The values of the MNIST and CelebA dataset will be in the range of -0.5 to 0.5 of 28x28 dimensional images.  The CelebA images will be cropped to remove parts of the image that don't include a face, then resized down to 28x28.
# 
# The MNIST images are black and white images with a single [color channel](https://en.wikipedia.org/wiki/Channel_(digital_image%29) while the CelebA images have [3 color channels (RGB color channel)](https://en.wikipedia.org/wiki/Channel_(digital_image%29#RGB_Images).
# ## Build the Neural Network
# You'll build the components necessary to build a GANs by implementing the following functions below:
# - `model_inputs`
# - `discriminator`
# - `generator`
# - `model_loss`
# - `model_opt`
# - `train`
# 
# ### Check the Version of TensorFlow and Access to GPU
# This will check to make sure you have the correct version of TensorFlow and access to a GPU

# In[4]:


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


# ### Input
# Implement the `model_inputs` function to create TF Placeholders for the Neural Network. It should create the following placeholders:
# - Real input images placeholder with rank 4 using `image_width`, `image_height`, and `image_channels`.
# - Z input placeholder with rank 2 using `z_dim`.
# - Learning rate placeholder with rank 0.
# 
# Return the placeholders in the following the tuple (tensor of real input images, tensor of z data)

# In[5]:


import problem_unittests as tests

def model_inputs(image_width, image_height, image_channels, z_dim):
    """
    Create the model inputs
    :param image_width: The input image width
    :param image_height: The input image height
    :param image_channels: The number of image channels
    :param z_dim: The dimension of Z
    :return: Tuple of (tensor of real input images, tensor of z data, learning rate)
    """
    # TODO: Implement Function
    real_input_images = tf.placeholder(tf.float32,(None,image_width,image_height,image_channels),name="real_input_images")
    z_data = tf.placeholder(tf.float32,(None,z_dim),name="z_data")
    learning_rate = tf.placeholder(tf.float32,name="lr",shape=())
    return (real_input_images,z_data,learning_rate)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_model_inputs(model_inputs)


# ### Discriminator
# Implement `discriminator` to create a discriminator neural network that discriminates on `images`.  This function should be able to reuse the variables in the neural network.  Use [`tf.variable_scope`](https://www.tensorflow.org/api_docs/python/tf/variable_scope) with a scope name of "discriminator" to allow the variables to be reused.  The function should return a tuple of (tensor output of the discriminator, tensor logits of the discriminator).

# In[29]:


def discriminator(images, reuse=False,alpha=0.2,dropout=0.5):
    """
    Create the discriminator network
    :param images: Tensor of input image(s)
    :param reuse: Boolean if the weights should be reused
    :return: Tuple of (tensor output of the discriminator, tensor logits of the discriminator)
    """
    # TODO: Implement Function
    with tf.variable_scope('discriminator', reuse=reuse):
        cov1 = tf.layers.conv2d(images, 128,5, strides=2, padding='same',kernel_initializer=tf.contrib.layers.xavier_initializer())
        cov1 = tf.maximum(alpha*cov1,cov1)

        cov2 = tf.layers.conv2d(cov1,256,4,strides=2, padding='same',kernel_initializer=tf.contrib.layers.xavier_initializer())
        cov2 = tf.layers.batch_normalization(cov2, training=True)
        cov2 = tf.maximum(alpha*cov2,cov2)

        cov3 = tf.layers.conv2d(cov2,512,3,strides=2, padding='same',kernel_initializer=tf.contrib.layers.xavier_initializer())
        cov3 = tf.layers.batch_normalization(cov3, training=True)
        cov3 = tf.maximum(alpha*cov3,cov3)
        
        cov4 = tf.layers.conv2d(cov3,1024,2,strides=2, padding='same',kernel_initializer=tf.contrib.layers.xavier_initializer())
        cov4 = tf.layers.batch_normalization(cov4, training=True)
        cov4 = tf.maximum(alpha*cov4,cov4)

        flat = tf.reshape(cov4, (-1, 2*2*1024))
        dropout_layer = tf.layers.dropout(inputs=flat, rate=dropout)
        logits = tf.layers.dense(dropout_layer, 1)
        outputs = tf.sigmoid(logits)    
    return (outputs, logits)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_discriminator(discriminator, tf)


# ### Generator
# Implement `generator` to generate an image using `z`. This function should be able to reuse the variables in the neural network.  Use [`tf.variable_scope`](https://www.tensorflow.org/api_docs/python/tf/variable_scope) with a scope name of "generator" to allow the variables to be reused. The function should return the generated 28 x 28 x `out_channel_dim` images.

# In[30]:


def generator(z, out_channel_dim, is_train=True, alpha=0.2):
    """
    Create the generator network
    :param z: Input z
    :param out_channel_dim: The number of channels in the output image
    :param is_train: Boolean if generator is being used for training
    :return: The tensor output of the generator
    """
    # TODO: Implement Function
    with tf.variable_scope('generator', reuse= not is_train):
        f_c = tf.layers.dense(z,7*7*256)
        f_c =tf.reshape(f_c,(-1,7,7,256))
        f_c = tf.layers.batch_normalization(f_c, training=is_train)
        f_c = tf.maximum(alpha * f_c, f_c)
        
        cov1 = tf.layers.conv2d_transpose(f_c,128,5,strides=2, padding='same')
        cov1 = tf.layers.batch_normalization(cov1, training=True)
        cov1 = tf.maximum(alpha*cov1,cov1)

        cov2 = tf.layers.conv2d_transpose(cov1,64,4,strides=2, padding='same')
        cov2 = tf.layers.batch_normalization(cov2, training=True)
        cov2 = tf.maximum(alpha*cov2,cov2)
        
        logits = tf.layers.conv2d_transpose(cov2,out_channel_dim,3,strides=1, padding='same')
        outputs = tf.tanh(logits)
        return outputs


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_generator(generator, tf)


# ### Loss
# Implement `model_loss` to build the GANs for training and calculate the loss.  The function should return a tuple of (discriminator loss, generator loss).  Use the following functions you implemented:
# - `discriminator(images, reuse=False)`
# - `generator(z, out_channel_dim, is_train=True)`

# In[31]:


def model_loss(input_real, input_z, out_channel_dim,alpha=0.2, dropout=0.5):
    """
    Get the loss for the discriminator and generator
    :param input_real: Images from the real dataset
    :param input_z: Z input
    :param out_channel_dim: The number of channels in the output image
    :return: A tuple of (discriminator loss, generator loss)
    """
    # TODO: Implement Function
    g_out = generator(input_z, out_channel_dim, is_train=True)
    d_out_real,d_logits_real = discriminator(input_real, reuse=False,alpha=alpha,dropout=dropout)
    d_out_fake,d_logits_fake = discriminator(g_out, reuse=True,alpha=alpha,dropout=dropout)
    
    dloss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,labels=tf.ones_like(d_out_real)* 0.9))
    dloss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_out_fake)))
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_out_fake)))
    d_loss = dloss_real + dloss_fake
    return (d_loss,g_loss)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_model_loss(model_loss)


# ### Optimization
# Implement `model_opt` to create the optimization operations for the GANs. Use [`tf.trainable_variables`](https://www.tensorflow.org/api_docs/python/tf/trainable_variables) to get all the trainable variables.  Filter the variables with names that are in the discriminator and generator scope names.  The function should return a tuple of (discriminator training operation, generator training operation).

# In[32]:


def model_opt(d_loss, g_loss, learning_rate, beta1):
    """
    Get optimization operations
    :param d_loss: Discriminator loss Tensor
    :param g_loss: Generator loss Tensor
    :param learning_rate: Learning Rate Placeholder
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :return: A tuple of (discriminator training operation, generator training operation)
    """
    # TODO: Implement Function
    trainable_vars = tf.trainable_variables()
    d_vars = [var for var in trainable_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in trainable_vars if var.name.startswith('generator')]
    
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)
    
    return (d_optimizer, g_optimizer)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_model_opt(model_opt, tf)


# ## Neural Network Training
# ### Show Output
# Use this function to show the current output of the generator during training. It will help you determine how well the GANs is training.

# In[33]:


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np

def show_generator_output(sess, n_images, input_z, out_channel_dim, image_mode):
    """
    Show example output for the generator
    :param sess: TensorFlow session
    :param n_images: Number of Images to display
    :param input_z: Input Z Tensor
    :param out_channel_dim: The number of channels in the output image
    :param image_mode: The mode to use for images ("RGB" or "L")
    """
    cmap = None if image_mode == 'RGB' else 'gray'
    z_dim = input_z.get_shape().as_list()[-1]
    example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])

    samples = sess.run(
        generator(input_z, out_channel_dim, False),
        feed_dict={input_z: example_z})

    images_grid = helper.images_square_grid(samples, image_mode)
    pyplot.imshow(images_grid, cmap=cmap)
    pyplot.show()


# ### Train
# Implement `train` to build and train the GANs.  Use the following functions you implemented:
# - `model_inputs(image_width, image_height, image_channels, z_dim)`
# - `model_loss(input_real, input_z, out_channel_dim)`
# - `model_opt(d_loss, g_loss, learning_rate, beta1)`
# 
# Use the `show_generator_output` to show `generator` output while you train. Running `show_generator_output` for every batch will drastically increase training time and increase the size of the notebook.  It's recommended to print the `generator` output every 100 batches.

# In[38]:


def train(epoch_count, batch_size, z_dim, learning_rate, beta1, get_batches, data_shape, data_image_mode):
    """
    Train the GAN
    :param epoch_count: Number of epochs
    :param batch_size: Batch Size
    :param z_dim: Z dimension
    :param learning_rate: Learning Rate
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :param get_batches: Function to get batches
    :param data_shape: Shape of the data
    :param data_image_mode: The image mode to use for images ("RGB" or "L")
    """
    # TODO: Build Model
    step = 0
    show_step = 100
    pstep = 20
    images_to_show = 9
    _,img_width,img_height,img_channels = data_shape
    
    real_input_images,z_data,lr = model_inputs(img_width,img_height,img_channels, z_dim)
    d_loss,g_loss = model_loss(real_input_images,z_data,img_channels,)
    d_optimizer, g_optimizer = model_opt(d_loss, g_loss, lr, beta1)
    print("Training starts")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(epoch_count):
            for batch_images in get_batches(batch_size):
                # TODO: Train Model
                step = step + 1
                batch_images=batch_images*2
                z_batch = np.random.uniform(-1, 1, size=(batch_size, z_dim))
                sess.run(d_optimizer, feed_dict={real_input_images: batch_images, z_data: z_batch, lr: learning_rate})
                #生成器运行两次以赶上辨识器速度
                sess.run(g_optimizer, feed_dict={z_data: z_batch, real_input_images: batch_images, lr: learning_rate})
                sess.run(g_optimizer, feed_dict={z_data: z_batch, real_input_images: batch_images, lr: learning_rate})
                
                if step % pstep == 0:
                    train_loss_d = d_loss.eval({z_data: z_batch, real_input_images: batch_images})
                    train_loss_g = g_loss.eval({z_data: z_batch})
                    print("Step: {}:".format(step),
                          "Epoch {}/{}...".format(epoch_i+1, epoch_count),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "Generator Loss: {:.4f}".format(train_loss_g))
                if step % show_step == 0:
                    show_generator_output(sess, images_to_show, z_data, img_channels, data_image_mode)
    print("Training Finished")


# ### MNIST
# Test your GANs architecture on MNIST.  After 2 epochs, the GANs should be able to generate images that look like handwritten digits.  Make sure the loss of the generator is lower than the loss of the discriminator or close to 0.

# In[28]:


batch_size = 64
z_dim = 128
learning_rate = 0.0003
beta1 = 0.5


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
epochs = 2

mnist_dataset = helper.Dataset('mnist', glob(os.path.join(data_dir, 'mnist/*.jpg')))
with tf.Graph().as_default():
    train(epochs, batch_size, z_dim, learning_rate, beta1, mnist_dataset.get_batches,
          mnist_dataset.shape, mnist_dataset.image_mode)


# ### CelebA
# Run your GANs on CelebA.  It will take around 20 minutes on the average GPU to run one epoch.  You can run the whole epoch or stop when it starts to generate realistic faces.

# In[39]:


batch_size = 32
z_dim = 84
learning_rate = 0.0001
beta1 = 0.3


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
epochs = 1

celeba_dataset = helper.Dataset('celeba', glob(os.path.join(data_dir, 'img_align_celeba/*.jpg')))
with tf.Graph().as_default():
    train(epochs, batch_size, z_dim, learning_rate, beta1, celeba_dataset.get_batches,
          celeba_dataset.shape, celeba_dataset.image_mode)


# ### Submitting This Project
# When submitting this project, make sure to run all the cells before saving the notebook. Save the notebook file as "dlnd_face_generation.ipynb" and save it as a HTML file under "File" -> "Download as". Include the "helper.py" and "problem_unittests.py" files in your submission.
