import tensorflow as tf


# tf.reset_default_graph()

xs = tf.placeholder(tf.float32,[None,28 * 28])

W = tf.Variable(tf.zeros([784,10]),dtype=tf.float32,name='weights')
b = tf.Variable(tf.zeros([10]),dtype=tf.float32,name='biases')
keep_prob = tf.placeholder(tf.float32)

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

# x:输入 W：权重
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

# x:输入
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# img shape [-1,28,28,1]
def forward(img):
    img_data = tf.reshape(xs,[-1,28,28,1])
    ## conv1 layer##
    W_conv1 = weight_variable([5,5,1,32])   #5*5的采样窗口 32个卷积核从一个平面抽取特征 32个卷积核是自定义的
    b_conv1 = bias_variable([32])   #每一个卷积核一个偏置值
    h_conv1 = tf.nn.relu(conv2d(img_data,W_conv1)+b_conv1) # output size 28x28x32
    h_pool1 = max_pool_2x2(h_conv1)      # output size 14x14x32

    ## conv2 layer##
    W_conv2 = weight_variable([5,5,32,64])   #5*5的采样窗口 32个卷积核从一个平面抽取特征 32个卷积核是自定义的
    b_conv2 = bias_variable([64])   #每一个卷积核一个偏置值
    h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2) # output size 14x14x64
    h_pool2 = max_pool_2x2(h_conv2)      # output size 7x7x64

    ## fc1 layer ##
    W_fc1 = weight_variable([7*7*64,1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

    ## fc2 layer ##
    W_fc2 = weight_variable([1024,10])
    b_fc2 = bias_variable([10])
    y_forward = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

    prediction = tf.argmax(y_forward,1)

    init = tf.global_variables_initializer() # init
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        saver.restore(sess,'model/CNN_MNIST_Model.ckpt')
        print('Model restored!')

        predint = sess.run(prediction,feed_dict={img_data:img,keep_prob: 1.0})

        print('recognize result:')
        print(predint[0])






