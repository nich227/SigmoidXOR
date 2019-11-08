# We want to use Tensorflow version 1.15.0 (newer versions breaks code :o)
# This is only necessary with Google Colab
# %tensorflow_version 1.15.0

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#   each column of S is a stimulus vector
S = [[1, 0, 1, 0],
     [1, 1, 0, 0]]
S = np.array(S)

#   each column of O is a target vector
O = [[1, 0, 0, 0],  # AND
     [1, 1, 1, 0],  # OR
     [0, 1, 1, 0]]  # XOR
O = np.array(O)


def SigmoidNet(S, O, h_len, numSteps=10000, gamma=.4):
    #   Define some size constants
    s_len = S.shape[0]  # number of input units
    o_len = O.shape[0]  # number of output units
    numS = S.shape[1]   # number of stimuli

    #   Clear the graph workspace
    tf.reset_default_graph()

    #   Initialize weights
    V = tf.get_variable(
        'V', shape=[o_len, h_len], initializer=tf.initializers.random_normal(0, .1))
    W = tf.get_variable(
        'W', shape=[h_len, s_len], initializer=tf.initializers.random_normal(0, .1))

    #   Funnels for inputting real data to the graph
    s_i = tf.placeholder(dtype=tf.float32, shape=[s_len, 1])
    o_i = tf.placeholder(dtype=tf.float32, shape=[o_len, 1])

    #   Define the Forward pass for stimulus pattern i
    #   in the tensorflow op graph (all of these variables are tensors)
    r_i = W @ s_i
    h_i = 1/(1+tf.exp(-r_i))
    f_i = V @ h_i
    p_i = 1/(1+tf.exp(-f_i))
    o_i_T = tf.transpose(o_i)
    c_i = -o_i_T@tf.log(p_i) - (1 - o_i_T)@tf.log(1 - p_i)

    #   Define a tensorflow optimizer (this is an op that finds the gradient
    #   and applies the weight update formula with respect to a particular objective)
    opt = tf.train.GradientDescentOptimizer(
        gamma/numS)  # Give the optimizer a learning rate
    train = opt.minimize(c_i)  # Tell the optimizer what value to minimize

    #   Display lists and values
    ln_list = []
    P = np.zeros((o_len, numS))

    with tf.Session() as sess:

        init = tf.global_variables_initializer()
        sess.run(init)

        #   Training Loop (Batch)
        for b in range(0, numSteps):

            #   Important variable values over stimuli
            #   for display purposes
            ln = 0

            for i in range(0, numS):

                #   Grab our stimulus and target pair
                s_i_val = S[:, i, np.newaxis]
                o_i_val = O[:, i, np.newaxis]

                '''
                #   Stochastic pick
                pick = np.random.choice(numS)

                #   Grab our stimulus and target pair
                s_i_val = S[:, pick, np.newaxis]
                o_i_val = O[:, pick, np.newaxis]
              '''

                #   Tell tensorflow how to feed the values into the placeholders
                feed_dict = {s_i: s_i_val, o_i: o_i_val}

                #   Run the ops that get the output of the network and train it
                #   and give it the values for the placeholders (i.e. the input)
                p_i_val, c_i_val, dummy = sess.run(
                    [p_i, c_i, train], feed_dict=feed_dict)

                #   Calculate and display network values
                ln = ln + 1/numS * c_i_val
                P[:, i] = np.squeeze(p_i_val)

            #   Append to display lists
            ln_list.append(np.squeeze(ln))

    plt.plot(ln_list)

    print("FinalLn : ", ln)

    print("\nDesired Output Columns: ")
    print(O)

    print("\nCurrent Output Columns: ")
    print(np.round(P, 2))


SigmoidNet(S, O, 10)  # 10 hidden units
