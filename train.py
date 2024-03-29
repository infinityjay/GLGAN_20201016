import tensorflow as tf
import numpy as np
from config import *
from network import *
from ops import *

def train(args, sess, model):
    #Adam optimizers are used instead of AdaDelta
    d_optimizer = tf.train.AdamOptimizer(args.learning_rate, beta1=args.momentum, name="AdamOptimizer_D").minimize(model.d_loss, var_list=model.d_vars)
    c_optimizer = tf.train.AdamOptimizer(args.learning_rate, beta1=args.momentum, name="AdamOptimizer_C").minimize(model.recon_loss, var_list=model.c_vars)
    
    global_optimizer = tf.train.AdamOptimizer(args.learning_rate, beta1=args.momentum, name="AdamOptimizer_C").minimize(model.loss_all, var_list=model.c_vars)

    epoch = 0
    step = 0
    global_step = 0
    c_data = []
    d_data = []
    g_data = []

    #saver
    saver = tf.train.Saver()        
    if args.continue_training:
        tf.local_variables_initializer().run()
        last_ckpt = tf.train.latest_checkpoint(args.checkpoints_path)
        saver.restore(sess, last_ckpt)
        ckpt_name = str(last_ckpt)
        print("Loaded model file from " + ckpt_name)
        epoch = int(ckpt_name.split('-')[-1])
    else:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()



    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #summary init
    all_summary = tf.summary.merge([model.recon_loss_sum,
                                    model.d_loss_sum,
                                    model.loss_all_sum,
                                    model.input_img_sum, 
                                    model.real_img_sum,
                                    model.recon_img_sum,
                                    model.g_local_imgs_sum,
                                    model.r_local_imgs_sum])
    writer = tf.summary.FileWriter(args.graph_path, sess.graph)


    #training starts here

    #first train completion network
    while epoch < args.train_step:

        #Training Stage 1 (Completion Network)
        if epoch < args.Tc:
            summary, c_loss, _ = sess.run([all_summary, model.recon_loss, c_optimizer])
            writer.add_summary(summary, global_step)
            print("Epoch [%d] Step [%d] C Loss: [%.4f]" % (epoch, step, c_loss))
            c_data.append(c_loss)
        elif epoch < args.Tc + args.Td:
            #Training Stage 2 (Discriminator Network)
            summary, d_loss, _ = sess.run([all_summary, model.d_loss, d_optimizer])
            writer.add_summary(summary, global_step)
            print("Epoch [%d] Step [%d] D Loss: [%.4f]" % (epoch, step, d_loss))
            d_data.append(d_loss)
        else:
            #Training Stage 3 (Completion Network)
            summary, g_loss, _ = sess.run([all_summary, model.loss_all, global_optimizer])
            writer.add_summary(summary, global_step)
            print("Epoch [%d] Step [%d] G Loss: [%.4f]" % (epoch, step, g_loss))
            g_data.append(g_loss)
        

        # Check Test image results every time epoch is finished
        if step*args.batch_size >= model.data_count:
            # saver.save(sess, args.checkpoints_path + "/model", global_step=epoch)
            # test_res_imgs -->network.py line 53
            # res_img = sess.run(model.test_res_imgs, feed_dict={model.single_orig:orig_imgs,
            #                                            model.single_test:test_imgs,
            #                                            model.single_mask:mask}) #原文中code
            # res_img = sess.run(model.test_res_imgs)

            # save test img result
            # img_tile(epoch, args, res_img)
            step = 0
            epoch += 1

        step += 1
        global_step += 1
    saver.save(sess, args.checkpoints_path + "/model", global_step=epoch)# 只保留最后一个model

    np.save('./loss_plot/tc300_td1_trainstep1000/c_data_4.npy', c_data)
    np.save('./loss_plot/tc300_td1_trainstep1000/d_data_4.npy', d_data)
    np.save('./loss_plot/tc300_td1_trainstep1000/g_data_4.npy', g_data)
    coord.request_stop()
    coord.join(threads)
    sess.close()            
    print("Done.")


def main(_):
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    #create graph, images, and checkpoints folder if they don't exist
    if not os.path.exists(args.checkpoints_path):
        os.makedirs(args.checkpoints_path)
    if not os.path.exists(args.graph_path):
        os.makedirs(args.graph_path)
    if not os.path.exists(args.images_path):
        os.makedirs(args.images_path)

    with tf.Session(config=run_config) as sess:
        model = network(args)

        print('Start Training...')
        train(args, sess, model)

main(args)
