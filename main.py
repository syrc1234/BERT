"""Main script for ADDA."""

import param
from train import pretrain, adapt, evaluate, eval
from model import (BertEncoder, DistilBertEncoder, DistilRobertaEncoder,BertChineseEncoder,
                   BertClassifier, Discriminator, RobertaEncoder, RobertaClassifier)
from utils import XML2Array, CSV2Array,CSV2ArrayWithTime, convert_examples_to_features, \
    roberta_convert_examples_to_features, get_data_loader, init_model, convert_examples_to_features_with_time, get_data_loader_withtime
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, RobertaTokenizer
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import random
import argparse


def parse_arguments():
    # argument parsing
    parser = argparse.ArgumentParser(description="Specify Params for Experimental Setting")#1.创建一个ArgumentParser类的对象parser
    # description里面的字符串内容可以随便填，就是描述你这个对象ArgumentParser类的对象a是用来干什么的
    # 2.一系列的parser.add_argument()
    parser.add_argument('--src', type=str, default="scr",
                        choices=["books", "dvd", "electronics", "kitchen", "blog", "airline", "imdb","i"],
                        help="Specify src dataset")

    parser.add_argument('--tgt', type=str, default="tgt",
                        choices=["books", "dvd", "electronics", "kitchen", "blog", "airline", "imdb","a"],
                        help="Specify tgt dataset")
    #预训练
    parser.add_argument('--pretrain', default=True, action='store_true',
                        help='Force to pretrain source encoder/classifier')
    #适应
    parser.add_argument('--adapt', default=True, action='store_true',
                        help='Force to adapt target encoder')

    parser.add_argument('--seed', type=int, default=42,
                        help="Specify random state")

    parser.add_argument('--train_seed', type=int, default=42,
                        help="Specify random state")
    #action：当输入--load时，波动开关，值为True，否则默认为False
    parser.add_argument('--load', default=False, action='store_true',
                        help="Load saved model")

    parser.add_argument('--model', type=str, default="bertchinese",
                        choices=["bert", "distilbert", "roberta", "distilroberta","bertchinese"],
                        help="Specify model type")
    #最大句长
    parser.add_argument('--max_seq_length', type=int, default=256,
                        help="Specify maximum sequence length")#默认128

    parser.add_argument('--alpha', type=float, default=1.0,
                        help="Specify adversarial weight")

    parser.add_argument('--beta', type=float, default=1.0,
                        help="Specify KD loss weight")

    parser.add_argument('--temperature', type=int, default=20,
                        help="Specify temperature")

    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")#目标编码器的梯度范数

    parser.add_argument("--clip_value", type=float, default=0.01,
                        help="lower and upper clip value for disc. weights")#鉴别器的梯度裁剪值，使对抗性训练更稳定
    #批大小一次训练所选取的样本数，原定16
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Specify batch size")

    parser.add_argument('--pre_epochs', type=int, default=3,
                        help="Specify the number of epochs for pretrain")
    #为预训练指定日志步长，默认为1，即预训练时每步都会打印日志
    parser.add_argument('--pre_log_step', type=int, default=1,
                        help="Specify log step size for pretrain")

    parser.add_argument('--num_epochs', type=int, default=3,
                        help="Specify the number of epochs for adaptation")

    parser.add_argument('--log_step', type=int, default=1,
                        help="Specify log step size for adaptation")

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)#种子一样使用 random() 生成的随机数将会是同一个
    torch.manual_seed(seed)#为CPU中设置种子，生成随机数：
    if torch.cuda.device_count() > 0:#有gpu
        torch.cuda.manual_seed_all(seed)#为所有GPU设置种子，生成随机数：


def main():
    args = parse_arguments()#3.调用对象parser的parse_args()方法，得到一个新的对象
    # argument setting   #4.输出b对象的属性
    print("=== Argument Setting ===")
    print("src: " + args.src)
    print("tgt: " + args.tgt)
    print("seed: " + str(args.seed))
    print("train_seed: " + str(args.train_seed))
    print("model_type: " + str(args.model))
    print("max_seq_length: " + str(args.max_seq_length))
    print("batch_size: " + str(args.batch_size))
    print("pre_epochs: " + str(args.pre_epochs))
    print("num_epochs: " + str(args.num_epochs))
    print("AD weight: " + str(args.alpha))
    print("KD weight: " + str(args.beta))
    print("temperature: " + str(args.temperature))
    print("gpu:"+str(torch.cuda.device_count()))
    set_seed(args.train_seed)#设置各种随机数种子

    if args.model in ['roberta', 'distilroberta']:
        tokenizer = RobertaTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')#原版为RobertaTokenizer.from_pretrained'roberta-base'
    elif args.model in['bertchinese']:
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')#调用 from_pretrained 将从网上获取模型。当我们加载 bert-base-uncased时，我们会在日志中看到打印的模型定义。该模型是一个12层的深度神经网络！

    # preprocess data
    print("=== Processing datasets ===")
    #注意：针对源数据，如果有与目标数据相近领域的数据，使用一个领域迁移到另一个领域的方法；如果没有，则可以采用多个领域的数据作为源数据，这时需要自己创建一个excel文件，把所有数据放里面
    #至于哪个方法有效，可以两个都试试；要统一数据格式，建议采用“文本，标签”的格式，类似blog.csv
    # if args.src in ['blog', 'airline', 'imdb', 'source', 'i']:
    src_x, src_y = CSV2Array(os.path.join('data', args.src, 'train_638.csv'))#os.path.join()函数：连接两个或更多的路径名组件#拼接路径data/i/train.csv
    #src_x = Clean_txt(src_x)
    src_test_x, src_test_y = CSV2Array(os.path.join('data', args.src, 'test_638.csv'))
    #src_test_x = Clean_txt(src_test_x)
    # elif args.src in ['source']:
    #     src_x, src_y = CSV2Array(os.path.join('data', args.tgt, 'train.csv'))
    #     src_test_x, src_test_y = CSV2Array(os.path.join('data', args.tgt, 'test.csv'))
    # else:
    #     src_x, src_y = XML2Array(os.path.join('data', args.src, 'negative.review'),
    #                            os.path.join('data', args.src, 'positive.review'))#将xml数据转化为数组

    # src_x, src_test_x, src_y, src_test_y = train_test_split(src_x, src_y,
    #                                                         test_size=0.2,
    #                                                         stratify=src_y,
    #                                                         random_state=args.seed)#将源目标数据拆分为训练集和验证集(测试集)，8：2
    #注意：源训练集和目标数据集数量要相同，如果目标数据少的话，可以删去一些源数据；tokenizer函数是对文本标签做处理的，为统一使用，可以为我们的目标数据统一标注为0
    # if args.tgt in ['blog', 'airline', 'imdb']:
    #     tgt_x, tgt_y = CSV2Array(os.path.join('data', args.tgt, args.tgt + '.csv'))
    # elif args.tgt in ['target']:
    tgt_x, tgt_y, tgt_time= CSV2ArrayWithTime(os.path.join('data', args.tgt, 'tgt1_1.csv'))#目标领域所有数据#拼接路径data/a/train.csv,a为args.tgt默认值
    #tgt_x = Clean_txt(tgt_x)
    tgt_test_x, tgt_test_y = CSV2Array(os.path.join('data', args.tgt, 'test638o.csv'))#目标领域挑选几十条自己打标签，用作测试数据
    #tgt_test_x = Clean_txt(tgt_test_x)
    # else:
    #     tgt_x, tgt_y = XML2Array(os.path.join('data', args.tgt, 'negative.review'),
    #                              os.path.join('data', args.tgt, 'positive.review'))
    #
    # tgt_train_x, tgt_test_y, tgt_train_y, tgt_test_y = train_test_split(tgt_x, tgt_y,
    #                                                                     test_size=0.2,
    #                                                                     stratify=tgt_y,
    #                                                                     random_state=args.seed)

    if args.model in ['roberta', 'distilroberta']:
        src_features = roberta_convert_examples_to_features(src_x, src_y, args.max_seq_length, tokenizer)
        src_test_features = roberta_convert_examples_to_features(src_test_x, src_test_y, args.max_seq_length, tokenizer)
        tgt_test_features = roberta_convert_examples_to_features(tgt_test_x, tgt_test_y, args.max_seq_length, tokenizer)
        tgt_features = roberta_convert_examples_to_features(tgt_x, tgt_y, args.max_seq_length, tokenizer)
    else:#对数据分词编码，生成input_ids,input_mask,label_id,注此处获得的是列表
        src_features = convert_examples_to_features(src_x, src_y, args.max_seq_length, tokenizer)
        src_test_features = convert_examples_to_features(src_test_x, src_test_y, args.max_seq_length, tokenizer)
        tgt_test_features = convert_examples_to_features(tgt_test_x, tgt_test_y, args.max_seq_length, tokenizer)
        tgt_features = convert_examples_to_features_with_time(tgt_x, tgt_y, tgt_time, args.max_seq_length, tokenizer)

    # load dataset，先将list转换成tensor类型，再生成dataset，最后生成dataloader

    src_data_loader = get_data_loader(src_features, args.batch_size)
    src_data_eval_loader = get_data_loader(src_test_features, args.batch_size)
    tgt_data_test_loader = get_data_loader(tgt_test_features, args.batch_size)
    tgt_data_all_loader = get_data_loader_withtime(tgt_features, args.batch_size)

    # load models
    if args.model == 'bert':
        src_encoder = BertEncoder()
        tgt_encoder = BertEncoder()
        src_classifier = BertClassifier()
    elif args.model == 'distilbert':
        src_encoder = DistilBertEncoder()
        tgt_encoder = DistilBertEncoder()
        src_classifier = BertClassifier()
    elif args.model == 'roberta':
        src_encoder = RobertaEncoder()
        tgt_encoder = RobertaEncoder()
        src_classifier = RobertaClassifier()
    elif args.model == 'bertchinese':
        src_encoder = BertChineseEncoder()
        tgt_encoder = BertChineseEncoder()
        src_classifier = BertClassifier()
    else:
        src_encoder = DistilRobertaEncoder()
        tgt_encoder = DistilRobertaEncoder()
        src_classifier = RobertaClassifier()
    discriminator = Discriminator()
    evaluate(tgt_encoder, src_classifier, tgt_data_test_loader)
    evaluate(src_encoder, src_classifier, tgt_data_test_loader)

    if not args.load:#若load值为Ture时，有训练好的模型则载入  load的开关拨动见readme
        src_encoder = init_model(args, src_encoder, restore=param.src_encoder_path)#restore恢复模型，restore默认为空，若不为空且模型存在则载入训练好的模型
        src_classifier = init_model(args, src_classifier, restore=param.src_classifier_path)
        tgt_encoder = init_model(args, tgt_encoder, restore=param.tgt_encoder_path)
        discriminator = init_model(args, discriminator, restore=param.d_model_path)
    else:#初始化模型参数
        src_encoder = init_model(args, src_encoder)#load默认是False
        src_classifier = init_model(args, src_classifier)
        tgt_encoder = init_model(args, tgt_encoder)
        discriminator = init_model(args, discriminator)

    #evaluate(src_encoder, src_classifier, tgt_data_test_loader)#评估目标测试集a/test
    # train source model
    print("=== Training classifier for source domain ===")
    if not args.pretrain:#类似load,默认为ture
        src_encoder, src_classifier = pretrain(
            args, src_encoder, src_classifier, src_data_loader)#预训练源i/train.csv???

    # eval source model
    print("=== Evaluating classifier for source domain ===")
    #evaluate(src_encoder, src_classifier, src_data_loader)#源训练集
    #evaluate(src_encoder, src_classifier, src_data_eval_loader)#源验证集（测试集）
    evaluate(tgt_encoder, src_classifier, tgt_data_test_loader)
    evaluate(src_encoder, src_classifier, tgt_data_test_loader)#目标测试集
    #固定参数，不进行梯度更新，以便用源模型参数初始化目标模型
    for params in src_encoder.parameters():
        params.requires_grad = False

    for params in src_classifier.parameters():
        params.requires_grad = False

    # train target encoder by GAN
    print("=== Training encoder for target domain ===")
    if args.adapt:#类似pretrain,默认ture
        tgt_encoder.load_state_dict(src_encoder.state_dict())#将预训练的参数权重加载到新的模型之中
        tgt_encoder = adapt(args, src_encoder, tgt_encoder, discriminator,
                            src_classifier, src_data_loader, tgt_data_test_loader, tgt_data_all_loader)#返回目标编码器

    # eval target encoder on lambda0.1 set of target dataset
    print("=== Evaluating classifier for encoded target domain ===")
    print(">>> source only <<<")
    #evaluate(src_encoder, src_classifier, tgt_data_test_loader)
    print(">>> domain adaption <<<")
    #evaluate(tgt_encoder, src_classifier, tgt_data_test_loader)
    print(">>> final prediction <<<")
    #eval(tgt_encoder, src_classifier, tgt_data_all_loader)
    #eval(src_encoder, src_classifier, tgt_data_all_loader)
   


if __name__ == '__main__':
    main()
