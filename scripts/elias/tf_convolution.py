import numpy as np
import tensorflow as tf


#should actually go in a class, now tf recreates devices each time
def conv3d(a_in,a_filt):

	r_in = a_in.reshape(1,a_in.shape[0],a_in.shape[1],a_in.shape[2],1)
	x = tf.placeholder(tf.float32, shape=[1,a_in.shape[0],a_in.shape[1],a_in.shape[2],1], name='x-input')

	a_filt = a_filt.reshape(a_filt.shape[0],a_filt.shape[1],a_filt.shape[2],1,1)
	print a_filt.shape
	W = tf.Variable( tf.constant(a_filt, dtype=tf.float32) )

	conv = tf.nn.conv3d(input=x, filter=W, strides=[1, 1, 1, 1, 1], padding="SAME", name="conv")

	out = None
	#with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
	with tf.Session() as sess:
	    sess.run( tf.global_variables_initializer() )
	    print 'tf conv'
	    out = sess.run(fetches=conv, feed_dict={x:r_in})
	return out[0,:,:,:,0]


if __name__=="__main__":
	a_in = np.array([[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]]])
	a_filt = np.array([[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,2,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]]])
	result = conv3d(a_in,a_filt)
	print result
	print result.shape

