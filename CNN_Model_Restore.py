import tensorflow as tf

# img shape [-1,28*28,]
def detect(img):

    sess = tf.Session()

    # load meta
    saver = tf.train.import_meta_graph('model/CNN_MNIST_Model.ckpt-2000.meta')
    saver.restore(sess,tf.train.latest_checkpoint('model/'))

    # restore placeholder variable
    graph = tf.get_default_graph()
    x_data = graph.get_tensor_by_name('x_data:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')

    # restore op
    op_to_predict = graph.get_tensor_by_name('op_to_predict:0')

    # feed_dict
    feed_dict = {x_data:img,keep_prob:1.0}

    prediction = sess.run(op_to_predict,feed_dict)

    print('Identified numbers:%d'%prediction[0])