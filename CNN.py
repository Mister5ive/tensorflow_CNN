import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../MNIST_data/',one_hot=True)

training_epochs = 2000
training_batch = 50
display_step = 50

sess = tf.InteractiveSession()

## Save to file
# remember to define the same dtype and shape when restore

# W = tf.Variable(tf.zeros([784,10]),dtype=tf.float32,name='weights')
# b = tf.Variable(tf.zeros([10]),dtype=tf.float32,name='biases')
xs = tf.placeholder(tf.float32,[None,28 * 28],name='x_data')
ys = tf.placeholder(tf.float32,[None,10],name='y_data')
keep_prob = tf.placeholder(tf.float32,name='keep_prob')



def weight_variable(shape,name):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial,name=name)

def bias_variable(shape,name):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial,name=name)

# x:输入 W：权重
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

# x:输入
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


x_data = tf.reshape(xs,[-1,28,28,1])

## conv1 layer##
W_conv1 = weight_variable([5,5,1,32],'weight_conv1')   #5*5的采样窗口 32个卷积核从一个平面抽取特征 32个卷积核是自定义的
b_conv1 = bias_variable([32],'biases_conv1')   #每一个卷积核一个偏置值
h_conv1 = tf.nn.relu(conv2d(x_data,W_conv1)+b_conv1) # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)      # output size 14x14x32

## conv2 layer##
W_conv2 = weight_variable([5,5,32,64],'weight_conv2')   #5*5的采样窗口 32个卷积核从一个平面抽取特征 32个卷积核是自定义的
b_conv2 = bias_variable([64],'biases_conv2')   #每一个卷积核一个偏置值
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2) # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)      # output size 7x7x64

## fc1 layer ##
W_fc1 = weight_variable([7*7*64,1024],'weight_fc1')
b_fc1 = bias_variable([1024],'biases_fc1')
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([1024,10],'weight_fc2')
b_fc2 = bias_variable([10],'biases_fc2')
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

# get prediction digit
prediction_digit = tf.argmax(prediction,1,name='op_to_predict')

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1])) # 每行计算交叉熵
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(ys,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


saver = tf.train.Saver()  # defaults to saving all variables

sess.run(tf.global_variables_initializer())

for i in range(training_epochs):
    batch_xs,batch_ys = mnist.train.next_batch(training_batch)
    if i % display_step == 0:
        print('step:%d, training accuracy:%g'%(i,accuracy.eval(feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:1.0})))
    train_step.run(feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
saver.save(sess,'model/CNN_MNIST_Model.ckpt', global_step = i + 1)   ##保存模型参数

print("test accuracy %g"%accuracy.eval(feed_dict={
       xs: mnist.test.images, ys:mnist.test.labels, keep_prob: 1.0}))



