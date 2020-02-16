* fc2 s实现了断点续训
    ```py
    # 实现断点继续训
    ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    ```
* fc3 实现了实际图片输入的检验
* fc4 这里是怎么进行制作数据集