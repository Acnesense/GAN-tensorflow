import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot=True)

noise_dim = 100
gen_hidden_dim = 256
img_dim = 784
disc_hidden_dim = 256
learning_rate = 0.0002
epoch = 2000
batch_size = 200


def generator(x):

    G_W1 = tf.Variable(tf.truncated_normal(shape=[noise_dim, gen_hidden_dim], stddev=0.1))
    G_b1 = tf.Variable(tf.truncated_normal(shape=[gen_hidden_dim], stddev=0.1))
    G_h1 = tf.add(tf.matmul(x, G_W1), G_b1)
    G_h1 = tf.nn.relu(G_h1)

    G_Wout = tf.Variable(tf.truncated_normal(shape=[gen_hidden_dim, img_dim], stddev=0.1))
    G_bout = tf.Variable(tf.truncated_normal(shape=[img_dim], stddev=0.1))
    output = tf.add(tf.matmul(G_h1, G_Wout), G_bout)
    output = tf.sigmoid(output)
    
    return output


def discriminator(x):

    D_W1 = tf.Variable(tf.truncated_normal(shape=[img_dim, disc_hidden_dim], stddev=0.1))
    D_b1 = tf.Variable(tf.truncated_normal(shape=[disc_hidden_dim], stddev=0.1))
    D_h1 = tf.add(tf.matmul(x, D_W1), D_b1)
    D_h1 = tf.nn.relu(D_h1)

    D_Wout = tf.Variable(tf.truncated_normal(shape=[disc_hidden_dim, 1], stddev=0.1))
    D_bout = tf.Variable(tf.truncated_normal(shape=[1], stddev=0.1))
    output = tf.add(tf.matmul(D_h1, D_Wout), D_bout)
    output = tf.nn.sigmoid(output)

    return output


def main():
    # placeholder
    gen_input = tf.placeholder(tf.float32, [None, noise_dim])
    disc_input = tf.placeholder(tf.float32, [None, img_dim])

    # output
    gen_output = generator(gen_input)
    fake_output = discriminator(gen_output)
    real_output = discriminator(disc_input)

    # loss function
    disc_loss = -tf.reduce_mean(tf.log(real_output) + tf.log(1. -fake_output))
    gen_loss = tf.reduce_mean(tf.log(1-fake_output))

    disc_train_step = tf.train.AdamOptimizer(learning_rate).minimize(disc_loss)
    gen_train_step = tf.train.AdamOptimizer(learning_rate).minimize(gen_loss)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for i in range(epoch):
            for j in range(50):
           
                #train_fake_batch = np.random.randn(batch_size, noise_dim)
                train_fake_batch = np.random.uniform(-1., 1., size=[batch_size, noise_dim])
                train_data_batch, _ = mnist.train.next_batch(batch_size)
                sess.run([disc_train_step, gen_train_step], feed_dict={gen_input : train_fake_batch, disc_input : train_data_batch})

            if (i+1) % 100 == 0:
                        d_loss, g_loss = sess.run([disc_loss, gen_loss], feed_dict={gen_input : train_fake_batch, disc_input : train_data_batch})
                        print("discriminator_loss : ", d_loss, ",generative_loss :", g_loss)

if __name__ == "__main__":
    main()
