'''
NARRE
@author:
Chong Chen (cstchenc@163.com)

@ created:
27/8/2017
@references:
Chong Chen, Min Zhang, Yiqun Liu, and Shaoping Ma. 2018. Neural Attentional Rating Regression with Review-level Explanations. In WWW'18.
'''


import tensorflow as tf

class NARRE(object):
    def __init__(
            self, review_num_u, review_num_i, review_len_u, review_len_i, user_num, item_num, num_classes,
            user_vocab_size, item_vocab_size, n_latent, embedding_id, attention_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        self.input_u = tf.placeholder(tf.int32, [None, review_num_u, review_len_u], name="input_u")
        self.input_i = tf.placeholder(tf.int32, [None, review_num_i, review_len_i], name="input_i")
        self.input_reuid = tf.placeholder(tf.int32, [None, review_num_u], name='input_reuid')
        self.input_reiid = tf.placeholder(tf.int32, [None, review_num_i], name='input_reuid')
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.input_uid = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_iid = tf.placeholder(tf.int32, [None, 1], name="input_iid")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.drop0 = tf.placeholder(tf.float32, name="dropout0")
        iidW = tf.Variable(tf.random_uniform([item_num + 2, embedding_id], -0.1, 0.1), name="iidW")
        uidW = tf.Variable(tf.random_uniform([user_num + 2, embedding_id], -0.1, 0.1), name="uidW")

        l2_loss = tf.constant(0.0)
        with tf.name_scope("user_embedding"):
            self.W1 = tf.Variable(
                tf.random_uniform([user_vocab_size, embedding_size], -1.0, 1.0),
                name="W1")
            self.embedded_user = tf.nn.embedding_lookup(self.W1, self.input_u)
            self.embedded_users = tf.expand_dims(self.embedded_user, -1)


        with tf.name_scope("item_embedding"):
            self.W2 = tf.Variable(
                tf.random_uniform([item_vocab_size, embedding_size], -1.0, 1.0),
                name="W2")
            self.embedded_item = tf.nn.embedding_lookup(self.W2, self.input_i)
            self.embedded_items = tf.expand_dims(self.embedded_item, -1)


        pooled_outputs_u = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("user_conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                self.embedded_users = tf.reshape(self.embedded_users, [-1, review_len_u, embedding_size, 1])

                conv = tf.nn.conv2d(
                    self.embedded_users,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, review_len_u - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs_u.append(pooled)
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool_u = tf.concat(3,pooled_outputs_u)
        
        self.h_pool_flat_u = tf.reshape(self.h_pool_u, [-1, review_num_u, num_filters_total])

        pooled_outputs_i = []

        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("item_conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                self.embedded_items = tf.reshape(self.embedded_items, [-1, review_len_i, embedding_size, 1])
                conv = tf.nn.conv2d(
                    self.embedded_items,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, review_len_i - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs_i.append(pooled)
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool_i = tf.concat(3,pooled_outputs_i)
        self.h_pool_flat_i = tf.reshape(self.h_pool_i, [-1, review_num_i, num_filters_total])
        
        with tf.name_scope("dropout"):
            self.h_drop_u = tf.nn.dropout(self.h_pool_flat_u, 1.0)
            self.h_drop_i = tf.nn.dropout(self.h_pool_flat_i, 1.0)
        with tf.name_scope("attention"):
            Wau = tf.Variable(
                tf.random_uniform([num_filters_total, attention_size], -0.1, 0.1), name='Wau')
            Wru = tf.Variable(
                tf.random_uniform([embedding_id, attention_size], -0.1, 0.1), name='Wru')
            Wpu = tf.Variable(
                tf.random_uniform([attention_size, 1], -0.1, 0.1), name='Wpu')
            bau = tf.Variable(tf.constant(0.1, shape=[attention_size]), name="bau")
            bbu = tf.Variable(tf.constant(0.1, shape=[1]), name="bbu")
            self.iid_a = tf.nn.relu(tf.nn.embedding_lookup(iidW, self.input_reuid))
            self.u_j = tf.einsum('ajk,kl->ajl', tf.nn.relu(
                tf.einsum('ajk,kl->ajl', self.h_drop_u, Wau) + tf.einsum('ajk,kl->ajl', self.iid_a, Wru) + bau),
                                             Wpu)+bbu  # None*u_len*1

            self.u_a = tf.nn.softmax(self.u_j,1)  # none*u_len*1

            print self.u_a

            Wai = tf.Variable(
                tf.random_uniform([num_filters_total, attention_size], -0.1, 0.1), name='Wai')
            Wri = tf.Variable(
                tf.random_uniform([embedding_id, attention_size], -0.1, 0.1), name='Wri')
            Wpi = tf.Variable(
                tf.random_uniform([attention_size, 1], -0.1, 0.1), name='Wpi')
            bai = tf.Variable(tf.constant(0.1, shape=[attention_size]), name="bai")
            bbi = tf.Variable(tf.constant(0.1, shape=[1]), name="bbi")
            self.uid_a = tf.nn.relu(tf.nn.embedding_lookup(uidW, self.input_reiid))
            self.i_j =tf.einsum('ajk,kl->ajl', tf.nn.relu(
                tf.einsum('ajk,kl->ajl', self.h_drop_i, Wai) + tf.einsum('ajk,kl->ajl', self.uid_a, Wri) + bai),
                                             Wpi)+bbi

            self.i_a = tf.nn.softmax(self.i_j,1)  # none*len*1

            l2_loss += tf.nn.l2_loss(Wau)
            l2_loss += tf.nn.l2_loss(Wru)
            l2_loss += tf.nn.l2_loss(Wri)
            l2_loss += tf.nn.l2_loss(Wai)

        with tf.name_scope("add_reviews"):
            self.u_feas = tf.reduce_sum(tf.multiply(self.u_a, self.h_drop_u), 1)
            self.u_feas = tf.nn.dropout(self.u_feas, self.dropout_keep_prob)
            self.i_feas = tf.reduce_sum(tf.multiply(self.i_a, self.h_drop_i), 1)
            self.i_feas = tf.nn.dropout(self.i_feas, self.dropout_keep_prob)
        with tf.name_scope("get_fea"):

            iidmf = tf.Variable(tf.random_uniform([item_num + 2, embedding_id], -0.1, 0.1), name="iidmf")
            uidmf = tf.Variable(tf.random_uniform([user_num + 2, embedding_id], -0.1, 0.1), name="uidmf")

            self.uid = tf.nn.embedding_lookup(uidmf,self.input_uid)
            self.iid = tf.nn.embedding_lookup(iidmf,self.input_iid)
            self.uid = tf.reshape(self.uid,[-1,embedding_id])
            self.iid = tf.reshape(self.iid,[-1,embedding_id])
            Wu = tf.Variable(
                tf.random_uniform([num_filters_total, n_latent], -0.1, 0.1), name='Wu')
            bu = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bu")
            self.u_feas = tf.matmul(self.u_feas, Wu)+self.uid + bu

            Wi = tf.Variable(
                tf.random_uniform([num_filters_total, n_latent], -0.1, 0.1), name='Wi')
            bi = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bi")
            self.i_feas = tf.matmul(self.i_feas, Wi) +self.iid+ bi

       

        with tf.name_scope('ncf'):

            self.FM = tf.multiply(self.u_feas, self.i_feas)
            self.FM = tf.nn.relu(self.FM)

            self.FM=tf.nn.dropout(self.FM,self.dropout_keep_prob)

            Wmul=tf.Variable(
                tf.random_uniform([n_latent, 1], -0.1, 0.1), name='wmul')

            self.mul=tf.matmul(self.FM,Wmul)
            self.score=tf.reduce_sum(self.mul,1,keep_dims=True)

            self.uidW2 = tf.Variable(tf.constant(0.1, shape=[user_num + 2]), name="uidW2")
            self.iidW2 = tf.Variable(tf.constant(0.1, shape=[item_num + 2]), name="iidW2")
            self.u_bias = tf.gather(self.uidW2, self.input_uid)
            self.i_bias = tf.gather(self.iidW2, self.input_iid)
            self.Feature_bias = self.u_bias + self.i_bias

            self.bised = tf.Variable(tf.constant(0.1), name='bias')

            self.predictions = self.score + self.Feature_bias + self.bised

        with tf.name_scope("loss"):
            losses = tf.nn.l2_loss(tf.subtract(self.predictions, self.input_y))

            self.loss = losses + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            self.mae = tf.reduce_mean(tf.abs(tf.subtract(self.predictions, self.input_y)))
            self.accuracy =tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.predictions, self.input_y))))
