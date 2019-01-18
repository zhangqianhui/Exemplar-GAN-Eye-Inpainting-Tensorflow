import tensorflow as tf
from ops import conv2d, lrelu, de_conv, instance_norm, Residual, fully_connect
from utils import save_images
import numpy as np, os

class ExemplarGAN(object):

    # build model
    def __init__(self, batch_size, max_iters, model_path, data_ob, sample_path, log_dir, learning_rate, is_load, lam_recon,
                 lam_gp, use_sp, beta1, beta2, n_critic):

        self.batch_size = batch_size
        self.max_iters = max_iters
        self.model_path = model_path
        self.data_ob = data_ob
        self.sample_path = sample_path
        self.log_dir = log_dir
        self.learning_rate = learning_rate
        self.log_vars = []
        self.channel = data_ob.channel
        self.shape = data_ob.shape
        self.lam_recon = lam_recon
        self.lam_gp = lam_gp
        self.use_sp = use_sp
        self.is_load = is_load
        self.beta1 = beta1
        self.beta2 = beta2
        self.n_critic = n_critic
        self.output_size = data_ob.image_size
        self.input_img = tf.placeholder(tf.float32, [batch_size, self.output_size, self.output_size, self.channel])
        self.exemplar_images = tf.placeholder(tf.float32, [batch_size, self.output_size, self.output_size, self.channel])
        self.img_mask = tf.placeholder(tf.float32, [batch_size, self.output_size, self.output_size, self.channel])
        self.exemplar_mask =  tf.placeholder(tf.float32, [batch_size, self.output_size, self.output_size, self.channel])
        self.domain_label = tf.placeholder(tf.int32, [batch_size])
        self.lr_decay = tf.placeholder(tf.float32, None, name='lr_decay')

    def build_model_GAN(self):

        self.incomplete_img = self.input_img * (1 - self.img_mask)
        self.local_real_img = self.input_img * self.img_mask

        self.x_tilde = self.encode_decode(self.incomplete_img, self.exemplar_images, 1 - self.img_mask, self.exemplar_mask, reuse=False)
        self.local_fake_img = self.x_tilde * self.img_mask

        self.D_real_gan_logits = self.discriminate(self.input_img, self.exemplar_images, self.local_real_img, spectural_normed=self.use_sp, reuse=False)
        self.D_fake_gan_logits = self.discriminate(self.x_tilde, self.exemplar_images, self.local_fake_img, spectural_normed=self.use_sp, reuse=True)

        self.D_loss = self.loss_dis(self.D_real_gan_logits, self.D_fake_gan_logits)
        self.G_gan_loss = self.loss_gen(self.D_fake_gan_logits)

        self.recon_loss = tf.reduce_mean(
            tf.reduce_sum(tf.abs(self.x_tilde - self.input_img), axis=[1, 2, 3]) / (
            self.output_size * self.output_size * self.channel))

        self.G_loss = self.G_gan_loss + self.lam_recon * self.recon_loss

        self.log_vars.append(("D_loss", self.D_loss))
        self.log_vars.append(("G_loss", self.G_loss))

        self.t_vars = tf.trainable_variables()
        self.d_vars = [var for var in self.t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in self.t_vars if 'encode_decode' in var.name]

        print "d_vars", len(self.d_vars)
        print "e_vars", len(self.g_vars)

        self.saver = tf.train.Saver()
        for k, v in self.log_vars:
            tf.summary.scalar(k, v)

    def build_test_model_GAN(self):

        self.incomplete_img = self.input_img * (1 - self.img_mask)
        self.x_tilde = self.encode_decode(self.incomplete_img, self.exemplar_images, 1 - self.img_mask, self.exemplar_mask, reuse=False)
        self.t_vars = tf.trainable_variables()
        self.g_vars = [var for var in self.t_vars if 'encode_decode' in var.name]
        self.saver = tf.train.Saver()

    def loss_dis(self, d_real_logits, d_fake_logits):

        l1 = tf.reduce_mean(tf.nn.softplus(-d_real_logits))
        l2 = tf.reduce_mean(tf.nn.softplus(d_fake_logits))

        return l1 + l2

    def loss_gen(self, d_fake_logits):
        return tf.reduce_mean(tf.nn.softplus(-d_fake_logits))

    def test(self, test_step):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)
            load_step = test_step
            self.saver.restore(sess, os.path.join(self.model_path, 'model_{:06d}.ckpt'.format(load_step)))
            batch_num = len(self.data_ob.test_images_name) / self.batch_size

            for j in range(batch_num):

                test_data_list, batch_eye_pos, test_ex_list, test_eye_pos = self.data_ob.getTestNextBatch(batch_num=j, batch_size=self.batch_size,
                                                                               is_shuffle=False)
                batch_images_array = self.data_ob.getShapeForData(test_data_list, is_test=True)
                batch_exem_array = self.data_ob.getShapeForData(test_ex_list, is_test=True)
                batch_eye_pos = np.squeeze(batch_eye_pos)
                test_eye_pos = np.squeeze(test_eye_pos)
                x_tilde, incomplete_img = sess.run(
                    [self.x_tilde, self.incomplete_img],
                    feed_dict={self.input_img: batch_images_array, self.exemplar_images: batch_exem_array, self.img_mask: self.get_Mask(batch_eye_pos),
                               self.exemplar_mask: self.get_Mask(test_eye_pos)})
                output_concat = np.concatenate(
                    [batch_images_array, batch_exem_array, incomplete_img, x_tilde], axis=0)
                print output_concat.shape
                save_images(output_concat, [output_concat.shape[0] / 4, 4],
                            '{}/{:02d}_output.jpg'.format(self.sample_path, j))

    # do train
    def train(self):

        d_trainer = tf.train.AdamOptimizer(self.learning_rate * self.lr_decay, beta1=self.beta1, beta2=self.beta2)
        d_gradients = d_trainer.compute_gradients(self.D_loss, var_list=self.d_vars)
        opti_D = d_trainer.apply_gradients(d_gradients)

        m_trainer = tf.train.AdamOptimizer(self.learning_rate * self.lr_decay, beta1=self.beta1, beta2=self.beta2)
        m_gradients = m_trainer.compute_gradients(self.G_loss, var_list=self.g_vars)
        opti_M = m_trainer.apply_gradients(m_gradients)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
            step = 0
            step2 = 0
            lr_decay = 1

            if self.is_load:
                self.saver.restore(sess, os.path.join(self.model_path, 'model_{:06d}.ckpt'.format(step)))

            while step <= self.max_iters:

                if step > 20000 and lr_decay > 0.1:
                    lr_decay = (self.max_iters - step) / float(self.max_iters - 10000)

                for i in range(self.n_critic):

                    train_data_list, batch_eye_pos, batch_train_ex_list, batch_ex_eye_pos = self.data_ob.getNextBatch(step2, self.batch_size)
                    batch_images_array = self.data_ob.getShapeForData(train_data_list)
                    batch_exem_array = self.data_ob.getShapeForData(batch_train_ex_list)
                    batch_eye_pos = np.squeeze(batch_eye_pos)

                    batch_ex_eye_pos = np.squeeze(batch_ex_eye_pos)
                    f_d = {self.input_img: batch_images_array, self.exemplar_images: batch_exem_array,
                           self.img_mask: self.get_Mask(batch_eye_pos), self.exemplar_mask: self.get_Mask(batch_ex_eye_pos), self.lr_decay: lr_decay}

                    # optimize D
                    sess.run(opti_D, feed_dict=f_d)
                    step2 += 1

                # optimize M
                sess.run(opti_M, feed_dict=f_d)
                summary_str = sess.run(summary_op, feed_dict=f_d)
                summary_writer.add_summary(summary_str, step)

                if step % 50 == 0:
                    d_loss,  g_loss = sess.run([self.D_loss, self.G_loss],
                        feed_dict=f_d)
                    print("step %d d_loss = %.4f, g_loss=%.4f" % (step, d_loss, g_loss))

                if np.mod(step, 400) == 0:

                    x_tilde, incomplete_img, local_real, local_fake = sess.run([self.x_tilde, self.incomplete_img, self.local_real_img, self.local_fake_img], feed_dict=f_d)
                    output_concat = np.concatenate([batch_images_array, batch_exem_array, incomplete_img, x_tilde, local_real, local_fake], axis=0)
                    save_images(output_concat, [output_concat.shape[0]/self.batch_size, self.batch_size],
                                            '{}/{:02d}_output.jpg'.format(self.sample_path, step))
                if np.mod(step, 2000) == 0:
                    self.saver.save(sess, os.path.join(self.model_path, 'model_{:06d}.ckpt'.format(step)))

                step += 1

            save_path = self.saver.save(sess, self.model_path)
            print "Model saved in file: %s" % save_path

    def discriminate(self, x_var, x_exemplar, local_x_var, spectural_normed=False, reuse=False):

        with tf.variable_scope("discriminator") as scope:

            if reuse == True:
                scope.reuse_variables()

            conv = tf.concat([x_var, x_exemplar], axis=3)
            for i in range(5):
                output_dim = np.minimum(64 * np.power(2, i+1), 512)
                conv = lrelu(conv2d(conv, spectural_normed=spectural_normed, output_dim=output_dim, name='dis_conv_{}'.format(i)))

            conv = tf.reshape(conv, shape=[self.batch_size, conv.shape[1] * conv.shape[2] * conv.shape[3]])
            ful_global = fully_connect(conv, output_size=output_dim, spectural_normed=spectural_normed, scope='dis_fully1')

            conv = local_x_var
            for i in range(5):
                output_dim = np.minimum(64 * np.power(2, i+1), 512)
                conv = lrelu(conv2d(conv, spectural_normed=spectural_normed, output_dim=output_dim, name='dis_conv_2_{}'.format(i)))

            conv = tf.reshape(conv, shape=[self.batch_size, conv.shape[1] * conv.shape[2] * conv.shape[3]])
            ful_local = fully_connect(conv, output_size=output_dim, spectural_normed=spectural_normed, scope='dis_fully2')

            gan_logits = fully_connect(tf.concat([ful_global, ful_local], axis=1), output_size=1, spectural_normed=spectural_normed, scope='dis_fully3')

            return gan_logits

    def encode_decode(self, x_var, x_exemplar, img_mask, exemplar_mask, reuse=False):

        with tf.variable_scope("encode_decode") as scope:

            if reuse == True:
                scope.reuse_variables()

            x_var = tf.concat([x_var, img_mask, x_exemplar, exemplar_mask], axis=3)

            conv1 = tf.nn.relu(
                instance_norm(conv2d(x_var, output_dim=64, k_w=7, k_h=7, d_w=1, d_h=1, name='e_c1'), scope='e_in1'))
            conv2 = tf.nn.relu(
                instance_norm(conv2d(conv1, output_dim=128, k_w=4, k_h=4, d_w=2, d_h=2, name='e_c2'), scope='e_in2'))
            conv3 = tf.nn.relu(
                instance_norm(conv2d(conv2, output_dim=256, k_w=4, k_h=4, d_w=2, d_h=2, name='e_c3'), scope='e_in3'))

            r1 = Residual(conv3, residual_name='re_1')
            r2 = Residual(r1, residual_name='re_2')
            r3 = Residual(r2, residual_name='re_3')
            r4 = Residual(r3, residual_name='re_4')
            r5 = Residual(r4, residual_name='re_5')
            r6 = Residual(r5, residual_name='re_6')

            g_deconv1 = tf.nn.relu(instance_norm(de_conv(r6, output_shape=[self.batch_size,
                                                                           self.output_size/2, self.output_size/2, 128], name='gen_deconv1'), scope="gen_in"))
            # for 1
            g_deconv_1_1 = tf.nn.relu(instance_norm(de_conv(g_deconv1,
                        output_shape=[self.batch_size, self.output_size, self.output_size, 32], name='g_deconv_1_1'), scope='gen_in_1_1'))

            g_deconv_1_1_x = tf.concat([g_deconv_1_1, x_var], axis=3)
            x_tilde1 = conv2d(g_deconv_1_1_x, output_dim=self.channel, k_w=7, k_h=7, d_h=1, d_w=1, name='gen_conv_1_2')

            return tf.nn.tanh(x_tilde1)

    def get_Mask(self, eye_pos, flag=0):

        eye_pos = eye_pos
        #print eye_pos
        batch_mask = []
        for i in range(self.batch_size):

            current_eye_pos = eye_pos[i]
            #eye
            if flag == 0:
                #left eye, y
                mask = np.zeros(shape=[self.output_size, self.output_size, self.channel])
                scale = current_eye_pos[0] - 25 #current_eye_pos[3] / 2
                down_scale = current_eye_pos[0] + 25 #current_eye_pos[3] / 2
                l1_1 =int(scale)
                u1_1 =int(down_scale)
                #x
                scale = current_eye_pos[1] - 35 #current_eye_pos[2] / 2
                down_scale = current_eye_pos[1] + 35 #current_eye_pos[2] / 2
                l1_2 = int(scale)
                u1_2 = int(down_scale)

                mask[l1_1:u1_1, l1_2:u1_2, :] = 1.0
                #right eye, y
                scale = current_eye_pos[4] - 25 #current_eye_pos[7] / 2
                down_scale = current_eye_pos[4] + 25 #current_eye_pos[7] / 2

                l2_1 = int(scale)
                u2_1 = int(down_scale)

                #x
                scale = current_eye_pos[5] - 35 #current_eye_pos[6] / 2
                down_scale = current_eye_pos[5] + 35 #current_eye_pos[6] / 2
                l2_2 = int(scale)
                u2_2 = int(down_scale)

                mask[l2_1:u2_1, l2_2:u2_2, :] = 1.0

            batch_mask.append(mask)

        return np.array(batch_mask)




