# coding:utf-8
import MeCab
from model import Seq2Seq
from chainer import training
import chainer
from chainer.training import extensions
import numpy as np
import sys
import codecs
import json
from chainer.datasets import split_dataset_random
import yaml

tagger = MeCab.Tagger('mecabrc')

config = yaml.load(open("config.yml", encoding="utf-8"))
epoch_num = int(config["epoch"])
batch_size = int(config["batch"])
embed_size = int(config["embed"])
unit_size = int(config['unit'])
TRAIN_FILE = config["train_file"]
gpu=int(config["gpu"])

# GPUのセット
if gpu >= 0: # numpyかcuda.cupyか
    xp = chainer.cuda.cupy
    chainer.cuda.get_device(gpu).use()
else:
    xp = np

def parse_sentence(sentence):
    parsed = []
    for chunk in tagger.parse(sentence).splitlines()[:-1]:
        (surface, feature) = chunk.split('\t')
        parsed.append(surface)
    return parsed


def make_dataset(filename):
    dataset=[]
    word2id = {"■": 0}
    id2word = {0: "■"}
    id = 1
    with open(filename, "r") as f:
        lines = f.readlines()

        for line in lines:
            sentences = line.rstrip().split("\t")
            question = ["<start>"] + parse_sentence(sentences[0]) + ["<eos>"]
            answer = parse_sentence(sentences[1]) + ["<eos>"]
            word2id,id2word = make_dict(question,answer,word2id,id2word,id)
            id_question = sentence_to_word_id(question, word2id=word2id)
            id_answer = sentence_to_word_id(answer, word2id=word2id)
            dataset.append((id_question,id_answer))

    return dataset, word2id, id2word


def make_dict(question,answer,word2id,id2word,id):
    sentence = question + answer
    for word in sentence:
        if word not in word2id:
            word2id[word] = id
            id2word[id] = word
            id += 1
    return word2id,id2word


def sentence_to_word_id(split_sentence, word2id):
    id_sentence = []
    for word in split_sentence:
        id = word2id[word]
        id_sentence.append(id)
    return id_sentence


class ParallelSequentialIterator(chainer.dataset.Iterator):
    def __init__(self, dataset, batch_size, repeat=True):
        self.dataset = dataset
        self.batch_size = batch_size  # batch size
        self.epoch = 0
        self.is_new_epoch = False
        self.repeat = repeat
        length = len(dataset)
        self.iteration = 0
        self.epoch_detail = 0

    def __next__(self):
        length = len(self.dataset)
        if not self.repeat and self.iteration * self.batch_size >= length:
            raise StopIteration
        batch_start_index = self.iteration * self.batch_size % length
        batch_end_index = min(batch_start_index + self.batch_size, length)
        questions = [self.dataset[batch_index][0] for batch_index in range(batch_start_index, batch_end_index)]
        answers = [self.dataset[batch_index][1] for batch_index in range(batch_start_index, batch_end_index)]

        self.iteration += 1
        self.epoch_detail = self.calc_epoch_detail

        epoch = self.iteration * self.batch_size // length
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch

        return list(zip(questions, answers))

    def calc_epoch_detail(self):
        # Floating point version of epoch.
        return self.iteration * self.batch_size / len(self.dataset)



    def serialize(self, serializer):
        # It is important to serialize the state to be recovered on resume.
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)

class BPTTUpdater(training.updaters.StandardUpdater):
    def __init__(self, train_iter, optimizer, bprop_len, device):
        super(BPTTUpdater, self).__init__(
            train_iter, optimizer, device=device)
        self.bprop_len = bprop_len

    def update_core(self):
        loss = 0
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')
        for i in range(self.bprop_len):
            batch = train_iter.__next__()
            for question, answer in batch:
                loss += optimizer.target(xp.array(question, dtype=xp.int32),
                                        xp.array(answer, dtype=xp.int32))

        optimizer.target.cleargrads()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()

def translate(trainer,test_iter):
    source, target = test_data[np.random.choice(len(test_data))]
    result = model.generate_sentence([model.xp.array(source)])[0]

    source_sentence = ' '.join([source_words[x] for x in source])
    target_sentence = ' '.join([target_words[y] for y in target])
    result_sentence = ' '.join([target_words[y] for y in result])
    print('# source : ' + source_sentence)
    print('# result : ' + result_sentence)
    print('# expect : ' + target_sentence)



if __name__ == '__main__':
    dataset, word2id, id2word = make_dataset(TRAIN_FILE)
    tr, test_dataset = split_dataset_random(dataset, int(len(dataset) * 0.95), seed=0)
    train_dataset, valid_dataset = split_dataset_random(tr, int(len(tr) * 0.95), seed=0)
    #json.dump(id2word, open("dictionary_i2w.json", "w"))
    #json.dump(word2id, open("dictionary_w2i.json", "w"))
    print('dataset:{}\ntrain:{}\ntest:{}\nvalid:{}'.format(len(dataset),len(train_dataset),len(test_dataset),len(valid_dataset)))
    model = Seq2Seq(input_words=len(word2id),embed_size=embed_size,unit_size=unit_size)
    # GPUのセット
    if gpu >= 0: # numpyかcuda.cupyか
        model.to_gpu()
    optimizer = chainer.optimizers.SGD(lr=1.0)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.GradientClipping(5))

    train_iter = ParallelSequentialIterator(dataset=train_dataset, batch_size=batch_size)
    test_iter = ParallelSequentialIterator(dataset=test_dataset, batch_size=1,repeat=False)
    valid_iter = ParallelSequentialIterator(dataset=valid_dataset, batch_size=1,repeat=False)
    
    updater = BPTTUpdater(train_iter, optimizer,bprop_len=35,device=gpu)
    trainer = training.Trainer(updater, stop_trigger=(epoch_num, 'epoch'),out='result')
    #chainer.serializers.load_npz('./result/snapshot_epoch-0', trainer)
    
    trainer.extend(extensions.LogReport(trigger=(100,'iteration')))
    trainer.extend(extensions.Evaluator(test_iter, model, device=gpu))
    trainer.extend(extensions.Evaluator(valid_iter, model, device=gpu), name='val')
    trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'),trigger=(100,'iteration'))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    trainer.extend(extensions.ProgressBar())
    trainer.run()
    
    model.to_cpu()
    chainer.serializers.save_npz("model.npz", model)
    
