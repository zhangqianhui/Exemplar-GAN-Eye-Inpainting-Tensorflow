import tensorflow as tf
from utils import mkdir_p
from utils import Eyes
from ExemplarGAN import ExemplarGAN

import os
os.environ['CUDA_VISIBLE_DEVICES']= '0'

flags = tf.app.flags
flags.DEFINE_integer("OPER_FLAG", 0, "flag of opertion, test or train")
flags.DEFINE_string("OPER_NAME", "Experiment_6_21_3", "name of the experiment")
flags.DEFINE_string("path", '?', "path of training data")
flags.DEFINE_integer("batch_size", 8, "size of single batch")
flags.DEFINE_integer("max_iters", 100000, "number of total iterations for G")
flags.DEFINE_integer("learn_rate", 0.0001, "learning rate for g and d")
flags.DEFINE_boolean("is_load", False, "whether loading the pretraining model")
flags.DEFINE_boolean("use_sp", True, "whether using spectral normalization")
flags.DEFINE_integer("lam_recon", 1, "weight for recon loss")
flags.DEFINE_integer("lam_gp", 10, "weight for gradient penalty")
flags.DEFINE_integer("beta1", 0.5, "beta1 of Adam optimizer")
flags.DEFINE_integer("beta2", 0.999, "beta2 of Adam optimizer")
flags.DEFINE_integer("n_critic", 1, "iters of g for every d")

FLAGS = flags.FLAGS

if __name__ == "__main__":

    root_log_dir = "./outpout/log/logs{}".format(FLAGS.OPER_FLAG)
    checkpoint_dir = "./outpout/model_gan{}/".format(FLAGS.OPER_NAME)
    sample_path = "./outpout/sample{}/sample_{}".format(FLAGS.OPER_FLAG, FLAGS.OPER_NAME)

    mkdir_p(root_log_dir)
    mkdir_p(checkpoint_dir)
    mkdir_p(sample_path)

    m_ob = Eyes(FLAGS.path)

    semiGan = ExemplarGAN(batch_size= FLAGS.batch_size, max_iters= FLAGS.max_iters,
                      model_path= checkpoint_dir, data_ob= m_ob, sample_path= sample_path , log_dir= root_log_dir,
                      learning_rate=  FLAGS.learn_rate, is_load=FLAGS.is_load, lam_recon=FLAGS.lam_recon, lam_gp=FLAGS.lam_gp,
                    use_sp=FLAGS.use_sp, beta1=FLAGS.beta1, beta2=FLAGS.beta2, n_critic=FLAGS.n_critic)

    if FLAGS.OPER_FLAG == 0:

        semiGan.build_model_GAN()
        semiGan.train()

    if FLAGS.OPER_FLAG == 1:

        semiGan.build_test_model_GAN()
        semiGan.test()


