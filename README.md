# 参考内容
>epoch计算方式：
>epoch = step * batch_size / count_all_train_pics


本地运行slim框架所用命令行：

使用预训练模型进行inceptionv4等的finetune
```sh
训练：
python3 train_image_classifier.py --dataset_name=quiz --dataset_dir=/path/to/data --checkpoint_path=/path/to/inception_v4.ckpt --model_name=inception_v4 --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits/Aux_logits --train_dir=/path/to/train_ckpt --learning_rate=0.001 --optimizer=rmsprop  --batch_size=32

train集验证：
python3 eval_image_classifier.py --dataset_name=quiz --dataset_dir=/path/to/data --dataset_split_name=train --model_name=inception_v4 --checkpoint_path=/path/to/train_ckpt --eval_dir=/path/to/train_eval --batch_size=32 --max_num_batches=128

validation集验证：
python3 eval_image_classifier.py --dataset_name=quiz --dataset_dir=/path/to/data --dataset_split_name=validation --model_name=inception_v4 --checkpoint_path=/path/to/train_ckpt --eval_dir=/path/to/validation_eval --batch_size=32 --max_num_batches=128

统一脚本：
python3 train_eval_image_classifier.py --dataset_name=quiz --dataset_dir=/path/to/data --checkpoint_path=/path/to/inception_v4.ckpt --model_name=inception_v4 --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits/Aux_logits --optimizer=rmsprop --train_dir=/path/to/log/train_ckpt --learning_rate=0.001 --dataset_split_name=validation --eval_dir=/path/to/eval --max_num_batches=128
```

## cpu训练
本地没有显卡的情况下，使用上述命令进行训练会导致错误。只使用CPU进行训练的话，需要在训练命令或者统一脚本上添加**--clone_on_cpu=True**参数。tinymind上则需要新建一个**clone_on_cpu**的**bool**类型参数并设置为**True**
