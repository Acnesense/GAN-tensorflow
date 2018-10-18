import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

if os.environ.get('DISPLAY', '')== '':
    plt.switch_backend('agg')

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot=True)

noise_dim = 100
gen_hidden_dim = 256
disc_hidden_dim = 256
img_dim = 28 * 28
learning_rate = 0.0002
epoch = 200
batch_size = 128

def plot(samples):
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        plt.imshow(sample.reshape(28, 28))
    return fig


with tf.variable_scope('generator'):

    G_W1 = tf.Variable(tf.random_normal(shape=[noise_dim, gen_hidden_dim], stddev=5e-2))
    G_b1 = tf.Variable(tf.zeros(shape=[gen_hidden_dim]))

    G_Wout = tf.Variable(tf.random_normal(shape=[gen_hidden_dim, img_dim], stddev=5e-2))
    G_bout = tf.Variable(tf.zeros(shape=[img_dim]))

with tf.variable_scope('discriminator'):
    D_W1 = tf.Variable(tf.random_normal(shape=[img_dim, disc_hidden_dim], stddev=5e-2))
    D_b1 = tf.Variable(tf.zeros(shape=[disc_hidden_dim]))

    D_Wout = tf.Variable(tf.random_normal(shape=[disc_hidden_dim, 1], stddev=5e-2))
    D_bout = tf.Variable(tf.zeros(shape=[1]))


def generator(x):

    G_h1 = tf.add(tf.matmul(x, G_W1), G_b1)
    G_h1 = tf.nn.relu(G_h1)

    output = tf.add(tf.matmul(G_h1, G_Wout), G_bout)
    output = tf.nn.sigmoid(output)
    
    return output


def discriminator(x):
    
    D_h1 = tf.add(tf.matmul(x, D_W1), D_b1)
    D_h1 = tf.nn.relu(D_h1)

    output = tf.add(tf.matmul(D_h1, D_Wout), D_bout)
    output = tf.nn.sigmoid(output)

    return output


def main():
    # placeholder
    gen_input = tf.placeholder(tf.float32, [None, noise_dim])
    disc_input = tf.placeholder(tf.float32, [None, img_dim])

    # output of generator network
    gen_output = generator(gen_input)

    # output of 2 discriminator networks
    fake_output = discriminator(gen_output)
    real_output = discriminator(disc_input)

    # loss function
    disc_loss = -tf.reduce_mean(tf.log(real_output) + tf.log(1. -fake_output))
    gen_loss = -tf.reduce_mean(tf.log(fake_output))

    tvar = tf.trainable_variables()
    gvar = [var for var in tvar if 'generator' in var.name]
    dvar = [var for var in tvar if 'discriminator' in var.name]

    disc_train_step = tf.train.AdamOptimizer(learning_rate).minimize(disc_loss, var_list=dvar)
    gen_train_step = tf.train.AdamOptimizer(learning_rate).minimize(gen_loss, var_list=gvar)
    
    num_img = 0
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for i in range(epoch):

            for j in range(int(mnist.train.num_examples / batch_size)):
                train_fake_batch = np.random.uniform(-1., 1., size=[batch_size, noise_dim])
                train_data_batch, _ = mnist.train.next_batch(batch_size)
                _, d_loss = sess.run([disc_train_step, disc_loss], feed_dict={gen_input : train_fake_batch, disc_input : train_data_batch})
                _, g_loss = sess.run([gen_train_step, gen_loss], feed_dict={gen_input : train_fake_batch})


            print("iteration :", i, "discriminator_loss : ", d_loss, ",generative_loss :", g_loss)

            if i % 5 == 0:
                samples = sess.run(gen_output, feed_dict={gen_input : np.random.uniform(-1., 1., [64, noise_dim])})
                fig = plot(samples)
                plt.savefig('generated_image/%s.png' % str(num_img).zfill(3), bbox_inches='tight')
                num_img += 1
                plt.close(fig)


if __name__ == "__main__":
    main()
