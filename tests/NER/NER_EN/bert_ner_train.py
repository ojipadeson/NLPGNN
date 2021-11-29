import numpy as np
import tensorflow as tf

from nlpgnn.datas.checkpoint import LoadCheckpoint
from nlpgnn.datas.dataloader import TFWriter, TFLoader
from nlpgnn.metrics import Metric
from nlpgnn.models import bert
from nlpgnn.optimizers import optim
from nlpgnn.tools import bert_init_weights_from_checkpoint
from sklearn.metrics import classification_report

# 载入参数
# LoadCheckpoint(language='zh', model="bert", parameters="base", cased=True, url=None)
# language: the language you used in your input data
# model: the model you choose,could be bert albert and gpt2
# parameters: can be base large xlarge xxlarge for albert, base medium large for gpt2, base large for BERT.
# cased: True or false, only for bert model.
# url: you can give a link of other checkpoint.
load_check = LoadCheckpoint(language='en', cased=True)
param, vocab_file, model_path = load_check.load_bert_param()

# 定制参数
param.batch_size = 128
param.maxlen = 128
param.label_size = 9
total_epochs = 10


def ner_evaluation(true_label: list, predicts: list, masks: list):
    all_predict = []
    all_true = []
    true_label = [tf.reshape(item, [-1]).numpy() for item in true_label]
    predicts = [tf.reshape(item, [-1]).numpy() for item in predicts]
    masks = [tf.reshape(item, [-1]).numpy() for item in masks]
    for i, j, m in zip(true_label, predicts, masks):
        index = np.argwhere(m == 1)
        all_true.extend(i[index].reshape(-1))
        all_predict.extend(j[index].reshape(-1))
    report = classification_report(all_true, all_predict, digits=4) # paramaters labels
    print(report)


# 构建模型
class BERT_NER(tf.keras.Model):
    def __init__(self, param, **kwargs):
        super(BERT_NER, self).__init__(**kwargs)
        self.batch_size = param.batch_size
        self.maxlen = param.maxlen
        self.label_size = param.label_size

        self.bert = bert.BERT(param)

        self.dense = tf.keras.layers.Dense(self.label_size, activation="relu")

    def call(self, inputs, is_training=True):
        bert = self.bert(inputs, is_training)
        sequence_output = bert.get_sequence_output()  # batch,sequence,768
        pre = self.dense(sequence_output)
        pre = tf.reshape(pre, [self.batch_size, self.maxlen, -1])
        output = tf.math.softmax(pre, axis=-1)
        return output

    def predict(self, inputs, is_training=False):
        output = self(inputs, is_training=is_training)
        return output


model = BERT_NER(param)

model.build(input_shape=(3, param.batch_size, param.maxlen))

model.summary()

# 构建优化器
optimizer_bert = optim.AdamWarmup(learning_rate=2e-5,  # 重要参数
                                  decay_steps=10000,  # 重要参数
                                  warmup_steps=1000, )

# 构建损失函数
sparse_categotical_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

# 初始化参数
bert_init_weights_from_checkpoint(model,
                                  model_path,  # bert_model.ckpt
                                  param.num_hidden_layers,
                                  pooler=False)

# 写入数据 通过check_exist=True参数控制仅在第一次调用时写入
writer = TFWriter(param.maxlen, vocab_file,
                  modes=["train", "valid"], check_exist=False)
ner_load = TFLoader(param.maxlen, param.batch_size, epoch=total_epochs)

# 训练模型
# 使用tensorboard
summary_writer = tf.summary.create_file_writer("./tensorboard")

# Metrics
f1score = Metric.SparseF1Score(average="macro")
precsionscore = Metric.SparsePrecisionScore(average="macro")
recallscore = Metric.SparseRecallScore(average="macro")
accuarcyscore = Metric.SparseAccuracy()

# 保存模型
checkpoint = tf.train.Checkpoint(model=model)
manager = tf.train.CheckpointManager(checkpoint, directory="./save",
                                     checkpoint_name="model.ckpt",
                                     max_to_keep=3)
# For train model
print('Training Begin')
Batch = 0
Epoch = 0

step = 0
for _, _, _, _ in ner_load.load_train():
    step += 1
Epoch_Batch = int(step / total_epochs)
assert Epoch_Batch == step / total_epochs
print('Step for 1 epoch: {}'.format(Epoch_Batch))

train_Batch = 0
train_predicts = []
train_true_label = []
train_masks = []
for X, token_type_id, input_mask, Y in ner_load.load_train():
    with tf.GradientTape() as tape:
        predict = model([X, token_type_id, input_mask])
        loss = sparse_categotical_loss(Y, predict)
        
        train_predict = tf.argmax(predict, -1)
        train_predicts.append(predict)
        train_true_label.append(Y)
        train_masks.append(input_mask)

        if (Batch + 1) % Epoch_Batch == 0:
            print('Epoch {:3d}'.format(Epoch + 1))
            ner_evaluation(train_true_label, train_predicts, train_masks)
            print()
            # print("Epoch:{}\tloss:{:.4f}".format(Epoch + 1, loss.numpy()))
            # print("Epoch:{}\tacc:{:.4f}".format(Epoch + 1, accuracy))
            # print("Epoch:{}\tprecision{:.4f}".format(Epoch + 1, precision))
            # print("Epoch:{}\trecall:{:.4f}".format(Epoch + 1, recall))
            # print("Epoch:{}\tf1score:{:.4f}\n".format(Epoch + 1, f1))
            manager.save(checkpoint_number=Epoch)
            train_Batch = 0
            train_predicts = []
            train_true_label = []
            train_masks = []

            valid_Batch = 0
            valid_predicts = []
            valid_true_label = []
            valid_masks = []
            for valid_X, valid_token_type_id, valid_input_mask, valid_Y in ner_load.load_valid():
                predict = model.predict([valid_X, valid_token_type_id, valid_input_mask])
                predict = tf.argmax(predict, -1)
                valid_predicts.append(predict)
                valid_true_label.append(valid_Y)
                valid_masks.append(input_mask)
            print(writer.label2id())
            ner_evaluation(valid_true_label, valid_predicts, valid_masks)

        f1 = f1score(Y, predict)
        precision = precsionscore(Y, predict)
        recall = recallscore(Y, predict)
        accuracy = accuarcyscore(Y, predict)

        with summary_writer.as_default():
            tf.summary.scalar("loss", loss, step=Batch)
            tf.summary.scalar("acc", accuracy, step=Batch)
            tf.summary.scalar("f1", f1, step=Batch)
            tf.summary.scalar("precision", precision, step=Batch)
            tf.summary.scalar("recall", recall, step=Batch)

    grads_bert = tape.gradient(loss, model.variables)
    optimizer_bert.apply_gradients(grads_and_vars=zip(grads_bert, model.variables))
    Batch += 1
