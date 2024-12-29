import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.compat.v1.disable_eager_execution()


def constant_nodes():
    node1 = tf.constant(3.0, tf.float32)
    node2 = tf.constant(4.0)

    session = tf.compat.v1.Session()
    res = session.run([node1, node2])
    print(res)

    a = tf.constant(5)
    b = tf.constant(2)
    c = tf.constant(3)

    d = tf.multiply(a, b)
    e = tf.add(c, b)

    f = tf.subtract(d, e)

    session = tf.compat.v1.Session()
    res = session.run(f)
    print(res)


def placeholders():
    a = tf.compat.v1.placeholder(tf.float32)
    b = tf.compat.v1.placeholder(tf.float32)

    sum_node = a + b

    session = tf.compat.v1.Session()
    res = session.run(sum_node, {a: [1, 3], b: [2, 4]})

    print(res)


def variables():
    W = tf.compat.v1.Variable([0.3], tf.float32)
    b = tf.compat.v1.Variable([-0.3], tf.float32)
    x = tf.compat.v1.placeholder(tf.float32)

    linear_model = W * x + b
    init = tf.compat.v1.global_variables_initializer()

    session = tf.compat.v1.Session()
    session.run(init)
    res = session.run(linear_model, {x: [1, 2, 3, 4, 10**2]})
    print(res)


def computational_graph():
    W = tf.compat.v1.Variable([0.3], tf.float32)
    b = tf.compat.v1.Variable([-0.3], tf.float32)
    x = tf.compat.v1.placeholder(tf.float32)

    linear_model = W * x + b

    y = tf.compat.v1.placeholder(tf.float32)
    squared_deltas = linear_model - y
    loss = tf.compat.v1.reduce_sum(squared_deltas)

    init = tf.compat.v1.global_variables_initializer()

    session = tf.compat.v1.Session()
    session.run(init)

    res = session.run(loss, {x: [1, 2, 3, 4], y: [1, 2, 3, 4]})
    print(abs(res))

    opt = tf.compat.v1.train.GradientDescentOptimizer(0.01)
    train = opt.minimize(loss)

    session.run(init)
    for i in range(1000):
        session.run(train, {x: [1, 2, 3, 4], y: [1, 2, 3, 4]})

    res = session.run([W, b])
    print(res)


constant_nodes()
placeholders()
variables()
computational_graph()
