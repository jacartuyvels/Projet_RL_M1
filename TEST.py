import tensorflow as tf
import numpy as np   

class TEST:
    def __init__(self):
        self.x = tf.placeholder("float")
        self.y = tf.placeholder("float")
        self.w = tf.Variable([1.0, 2.0])#, name="w") #initial value of variable, and optional name
        self.y_model = tf.multiply(self.x, self.w[0]) + self.w[1]
        self.error = tf.square(self.y - self.y_model)
        self.train_op = tf.train.GradientDescentOptimizer(0.01).minimize(self.error)
        self.model = tf.global_variables_initializer()
    def score(self):      
        print("do score")
                     
    def learn(self):
        with tf.Session() as session:
            session.run(self.model)
            for i in range(1000):
                x_value = np.random.rand()
                y_value = x_value * 2 + 6
                session.run(self.train_op, feed_dict={self.x: x_value, self.y: y_value})
            self.w_value = session.run(self.w)


test = TEST()
test.score()
test.learn()

print("Predicted model: {a:.3f}x + {b:.3f}".format(a=test.w_value[0], b=test.w_value[1]))