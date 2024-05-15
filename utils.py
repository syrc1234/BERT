import os
#os.environ['CUDA_VISIBLE_DEVICES']='0,1,2'
import random
import numpy as np
import pandas as pd
import torch
from lxml import etree
import xml.etree.ElementTree as ET
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import param
import re


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_id = label_id

class InputFeaturesWithTime(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, label_id, time):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_id = label_id
        self.time = time

def XML2Array(neg_path, pos_path):
    parser = etree.XMLParser(recover=True)
    reviews = []
    negCount = 0
    posCount = 0
    labels = []
    regex = re.compile(r'[\n\r\t+]')

    neg_tree = ET.parse(neg_path, parser=parser)
    neg_root = neg_tree.getroot()

    for rev in neg_root.iter('review_text'):
        text = regex.sub(" ", rev.text)
        reviews.append(text)
        negCount += 1
    labels.extend(np.zeros(negCount, dtype=int))

    pos_tree = ET.parse(pos_path, parser=parser)
    pos_root = pos_tree.getroot()

    for rev in pos_root.iter('review_text'):
        text = regex.sub(" ", rev.text)
        reviews.append(text)
        posCount += 1
    labels.extend(np.ones(posCount, dtype=int))

    reviews = np.array(reviews)
    labels = np.array(labels)

    return reviews, labels


def CSV2Array(path):
    data = pd.read_csv(path, encoding='utf-8')#encoding可能得改latin
    reviews, labels = data.reviews.values.tolist(), data.labels.values.tolist()
    return reviews, labels


def CSV2ArrayWithTime(path):
    data = pd.read_csv(path, encoding='utf-8')#encoding可能得改latin
    reviews, labels, time = data.reviews.values.tolist(), data.labels.values.tolist(), data.creat_time.values.tolist()
    return reviews, labels, time


def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        print("true")
        tensor = tensor.cuda()
    return tensor


def init_random_seed(manual_seed):
    """Init random seed."""
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_model(args, net, restore=None):
    # restore model weights恢复模型权重
    if restore is not None:
        path = os.path.join(param.model_root, args.src, args.model, str(args.seed), restore)#snapshots/i/roberta/42
        if os.path.exists(path):
            net.load_state_dict(torch.load(path))
            print("Restore model from: {}".format(os.path.abspath(path)))

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()
    return net


def save_model(args, net, name):
    """Save trained model."""
    folder = os.path.join(param.model_root, args.src, args.model, str(args.seed))
    path = os.path.join(folder, name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(net.state_dict(), path)
    print("save pretrained model to: {}".format(path))


def save_adpmodel(args, net, name, epoch):
    """Save trained model."""
    folder = os.path.join(param.model_root, args.src, args.model, str(args.seed),str(epoch))
    path = os.path.join(folder, name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(net.state_dict(), path)
    print("save pretrained model to: {}".format(path))

def convert_examples_to_features(reviews, labels, max_seq_length, tokenizer,
                                 cls_token='[CLS]', sep_token='[SEP]',
                                 pad_token=0):
    features = []
    for ex_index, (review, label) in enumerate(zip(reviews, labels)): #zip打包为元组列表，元素个数与最短的列表一致。[(r0,l0),(r2,l2)...]#enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据下标和数据
        if (ex_index + 1) % 200 == 0:
            print("writing example %d of %d" % (ex_index + 1, len(reviews)))
        tokens = tokenizer.tokenize(review)#给reviews分词（变为字？）
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]#切片
        tokens = [cls_token] + tokens + [sep_token]#[SEP] 很好理解，作为一个句子结尾的记号，关键在于对 [CLS] 的理解。 [CLS] 的作用其实就是将整个句对/句子的上层抽象信息作为最终的最高隐层输入softmax中。[SEP] 很好理解，作为一个句子结尾的记号，关键在于对 [CLS] 的理解。 [CLS] 的作用其实就是将整个句对/句子的上层抽象信息作为最终的最高隐层输入softmax中。
        input_ids = tokenizer.convert_tokens_to_ids(tokens)#将tokens序列化（影射成数字）
        input_mask = [1] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)#填充长度
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          label_id=label))
    return features#数据特征


def convert_examples_to_features_with_time(reviews, labels, times,max_seq_length, tokenizer,
                                 cls_token='[CLS]', sep_token='[SEP]',
                                 pad_token=0):
    features = []
    for ex_index, (review, label, time) in enumerate(zip(reviews, labels, times)): #zip打包为元组列表，元素个数与最短的列表一致。[(r0,l0),(r2,l2)...]#enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据下标和数据
        if (ex_index + 1) % 200 == 0:
            print("writing example %d of %d" % (ex_index + 1, len(reviews)))
        tokens = tokenizer.tokenize(review)#给reviews分词（变为字？）
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]#切片
        tokens = [cls_token] + tokens + [sep_token]#[SEP] 很好理解，作为一个句子结尾的记号，关键在于对 [CLS] 的理解。 [CLS] 的作用其实就是将整个句对/句子的上层抽象信息作为最终的最高隐层输入softmax中。[SEP] 很好理解，作为一个句子结尾的记号，关键在于对 [CLS] 的理解。 [CLS] 的作用其实就是将整个句对/句子的上层抽象信息作为最终的最高隐层输入softmax中。
        input_ids = tokenizer.convert_tokens_to_ids(tokens)#将tokens序列化（影射成数字）
        input_mask = [1] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)#填充长度
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        features.append(
            InputFeaturesWithTime(input_ids=input_ids,
                          input_mask=input_mask,
                          label_id=label,time=time))
    return features#数据特征


def roberta_convert_examples_to_features(reviews, labels, max_seq_length, tokenizer,
                                         cls_token='<s>', sep_token='</s>',
                                         pad_token=1):
    features = []
    for ex_index, (review, label) in enumerate(zip(reviews, labels)):
        if (ex_index + 1) % 200 == 0:
            print("writing example %d of %d" % (ex_index + 1, len(reviews)))
        tokens = tokenizer.tokenize(review)
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]
        tokens = [cls_token] + tokens + [sep_token]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          label_id=label))
    return features


def get_data_loader(features, batch_size):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)#张量
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids)# TensorDataset对tensor进行打包
    sampler = RandomSampler(dataset)#随机对数据采样，这里相当于打乱？
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)#sampler(Sampler, optional): 自定义从数据集中取样本的策略，如果指定这个参数，那么shuffle必须为False
    return dataloader

def get_data_loader_withtime(features, batch_size):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)#张量
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_times = torch.tensor([f.time for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids,all_times)# TensorDataset对tensor进行打包
    sampler = RandomSampler(dataset)#随机对数据采样，这里相当于打乱？
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)#sampler(Sampler, optional): 自定义从数据集中取样本的策略，如果指定这个参数，那么shuffle必须为False
    return dataloader

def MMD(source, target):
    mmd_loss = torch.exp(-1 / (source.mean(dim=0) - target.mean(dim=0)).norm())
    return mmd_loss


"""
def Clean_txt(coms):
    for i, com in enumerate(coms):
        #com = zhconv.convert(com, 'zh-hant')
        com = re.sub(r"(回复)?(//)?\s*@\S*?\s*(:| |$)", " ", com)  # 去除正文中的@和回复/转发中的用户名
        #com = re.sub(r"\[\S+\]", "", com)  # 去除表情符号
        com = re.sub(r"\[", "", com)
        com = re.sub(r"\]", "", com)
        #com = com.replace("", "")
        #com = com.replace("]", "")
        #com = re.sub(r"#\S+#", "", com)      # 保留话题内容
        com = re.sub(r"\#\S+\#", "", com)
        URL_REGEX = re.compile(
            r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
            re.IGNORECASE)
        com = re.sub(URL_REGEX, "", com)  # 去除网址
        com = com.replace("转发微博", "")  # 去除无意义的词语
        #com = com.replace("我在这里:", "")
        com = com.replace(" ","")
        com = re.sub(r"\s+ ", " ", com)  # 合并正文中过多的空格
        coms[i] = com
        print(coms[i])
    return coms
"""