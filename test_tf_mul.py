import tensorflow as tf

init = tf.global_variables_initializer()
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    
    q = tf.Variable(tf.random_normal([100,10,50]))
    w = tf.Variable(tf.random_normal([100,1,50]))
    idx = tf.Variable(tf.random_normal([100,10,1]))

    test1 = tf.multiply(q,w)