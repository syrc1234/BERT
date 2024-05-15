"""Adversarial adaptation to train target encoder."""

import torch
from sklearn.metrics import f1_score, accuracy_score, recall_score

from utils import make_cuda
import torch.nn.functional as F
import torch.nn as nn
import param
import torch.optim as optim
from utils import save_model, save_adpmodel
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import os

def pretrain(args, encoder, classifier, data_loader):
    """Train classifier for source domain."""

    # setup criterion and optimizer
    optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()),
                           lr=param.c_learning_rate)
    CELoss = nn.CrossEntropyLoss()#交叉熵损失函数，值越小则为该类的可能性越大

    # set train state for Dropout and BN layers 设置Dropout层和BN层的训练状态
    encoder.train()
    classifier.train()
    writer = SummaryWriter("try148")
    for epoch in range(args.pre_epochs):
        for step, (reviews, mask, labels) in enumerate(data_loader):
            reviews = make_cuda(reviews)
            mask = make_cuda(mask)
            labels = make_cuda(labels)

            # zero gradients for optimizer 优化器的零梯度
            optimizer.zero_grad()#清空过往梯度；
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            # compute loss for discriminator 计算鉴别器的损耗
            feat = encoder(reviews, mask)
            preds = classifier(feat)
            cls_loss = CELoss(preds, labels)

            # optimize source classifier 优化源分类器
            cls_loss.backward()#根据loss来计算网络参数的梯度,反向传播，计算当前梯度
            optimizer.step()#根据梯度更新网络参数,用来更新优化器的学习率的，一般是按照epoch为单位进行更换，即多少个epoch后更换一次学习率

            # print step info
            if (step + 1) % args.pre_log_step == 0:
                print("Epoch [%.2d/%.2d] Step [%.3d/%.3d]: cls_loss=%.4f"
                      % (epoch + 1,
                         args.pre_epochs,
                         step + 1,
                         len(data_loader),
                         cls_loss.item()))
                writer.add_scalar('pretrain/loss', cls_loss.item(), len(data_loader)*epoch+step+1)
    writer.close()
    # save final model
    save_model(args, encoder, param.src_encoder_path)
    save_model(args, classifier, param.src_classifier_path)

    return encoder, classifier


def adapt(args, src_encoder, tgt_encoder, discriminator,
          src_classifier, src_data_loader, tgt_data_test_loader, tgt_data_all_loader):
    """Train encoder for target domain."""

    # set train state for Dropout and BN layers
    src_encoder.eval()
    src_classifier.eval()
    tgt_encoder.train()
    discriminator.train()

    # setup criterion and optimizer
    BCELoss = nn.BCELoss()#二分类损失函数，对输出向量的每个元素单独使用交叉熵损失函数，然后计算平均值
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')#KL散度，又叫相对熵，用于衡量两个分布（离散分布和连续分布）之间的距离 kl散度的本质是交叉熵减信息熵，即，使用估计分布编码真实分布所需的bit数，与编码真实分布所需的最少bit数的差。当且仅当估计分布与真实分布相同时，kl散度为0。因此可以作为两个分布差异的衡量方法。
    optimizer_G = optim.Adam(tgt_encoder.parameters(), lr=param.d_learning_rate)#将参数分别放进优化器
    optimizer_D = optim.Adam(discriminator.parameters(), lr=param.d_learning_rate)
    len_data_loader = min(len(src_data_loader), len(tgt_data_all_loader))
    writer = SummaryWriter("tgt638")
    for epoch in range(args.num_epochs):
        # zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_all_loader))
        for step, ((reviews_src, src_mask, _), (reviews_tgt, tgt_mask, _, tgt_time)) in data_zip:
            reviews_src = make_cuda(reviews_src)
            src_mask = make_cuda(src_mask)

            reviews_tgt = make_cuda(reviews_tgt)
            tgt_mask = make_cuda(tgt_mask)

            # zero gradients for optimizer
            optimizer_D.zero_grad()

            # extract and concat features
            with torch.no_grad():
                feat_src = src_encoder(reviews_src, src_mask)
            feat_src_tgt = tgt_encoder(reviews_src, src_mask)#源数据输入目标编码器的输出(相当于模型图中第二部分source bert的输出)
            feat_tgt = tgt_encoder(reviews_tgt, tgt_mask)#(b,768)
            feat_concat = torch.cat((feat_src_tgt, feat_tgt), 0)#(2b,768) #把多个tensor进行拼接

            # predict on discriminator
            pred_concat = discriminator(feat_concat.detach())

            # prepare real and fake label
            label_src = make_cuda(torch.ones(feat_src_tgt.size(0))).unsqueeze(1)#领域标签源为1，目标为0
            label_tgt = make_cuda(torch.zeros(feat_tgt.size(0))).unsqueeze(1)#torch.size(0)中的0表示第0维度的数据数量,unsqueeze升维度
            label_concat = torch.cat((label_src, label_tgt), 0)

            # compute loss for discriminator
            dis_loss = BCELoss(pred_concat, label_concat)#nn.BCELoss()为二元交叉熵损失函数，只能解决二分类问题，需要在该层前面加上Sigmoid函数
            dis_loss.backward()

            for p in discriminator.parameters():
                p.data.clamp_(-args.clip_value, args.clip_value)#将判别器所有的参数截断到一个区间内，min:-args.clip_value，max:args.clip_value
            # optimize discriminator更新鉴别器参数 #parameters里存的就是weight，parameters()会返回一个生成器（迭代器）
            optimizer_D.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])#去掉维数为1的的维度max(-1)中的-1表示按照最后一个维度（行）求最大值，即求每一个样本（每一行）概率的最大值。然后pred.max(-1)[1]中的方括号[1]则表示返回最大值的索引，即返回0或者1
            acc = (pred_cls == label_concat).float().mean()

            # zero gradients for optimizer
            optimizer_G.zero_grad()
            T = args.temperature

            # predict on discriminator
            pred_tgt = discriminator(feat_tgt)

            # logits for KL-divergence
            with torch.no_grad():
                src_prob = F.softmax(src_classifier(feat_src) / T, dim=-1)
            tgt_prob = F.log_softmax(src_classifier(feat_src_tgt) / T, dim=-1)
            kd_loss = KLDivLoss(tgt_prob, src_prob.detach()) * T * T

            # compute loss for target encoder
            gen_loss = BCELoss(pred_tgt, label_src)
            loss_tgt = args.alpha * gen_loss + args.beta * kd_loss
            loss_tgt.backward()
            torch.nn.utils.clip_grad_norm_(tgt_encoder.parameters(), args.max_grad_norm)#设置一个梯度剪切的阈值，如果在更新梯度的时候，梯度超过这个阈值，则会将其限制在这个范围之内，防止梯度爆炸。
            # optimize target encoder
            optimizer_G.step()

            if (step + 1) % args.log_step == 0:
                print("Epoch [%.2d/%.2d] Step [%.3d/%.3d]: "
                      "acc=%.4f g_loss=%.4f d_loss=%.4f kd_loss=%.4f"
                      % (epoch + 1,
                         args.num_epochs,
                         step + 1,
                         len_data_loader,
                         acc.item(),
                         gen_loss.item(),#生成网络损失函数
                         dis_loss.item(),#判别网络损失函数
                         kd_loss.item()))
                writer.add_scalars('adapt/loss', {'g_loss':gen_loss.item(),#生成网络损失函数
                         'd_loss':dis_loss.item(),#判别网络损失函数
                         'k_loss':kd_loss.item()}, epoch*len_data_loader+step+1)

        loss, acc, f1_macro = evaluate(tgt_encoder, src_classifier, tgt_data_test_loader)
        #tag = os.path.join('adapt/evaluate', str(epoch+1))
        writer.add_scalars('adapt/evaluate',{'loss':loss,#loss, acc, f1_macro
                         'acc':acc,
                         'f1_marco':f1_macro},epoch+1)
        save_adpmodel(args, tgt_encoder, param.tgt_encoder_path, epoch)
    writer.close()
    return tgt_encoder

#目标数据集上的源分类器对目标编码器的评估。
def evaluate(encoder, classifier, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()
    all_pred = None
    all_label = None
    # evaluate network
    for (reviews, mask, labels) in data_loader:
        reviews = make_cuda(reviews)
        mask = make_cuda(mask)
        labels = make_cuda(labels)

        with torch.no_grad():
            feat = encoder(reviews, mask)
            preds = classifier(feat)
        loss += criterion(preds, labels).item()
        pred_cls = preds.data.max(1)[1]#.data.max用于找概率最大的下标,(1):行，[1]:每行最大位置的下标。
        acc += pred_cls.eq(labels.data).cpu().sum().item()
        pred_cls = pred_cls.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        if all_pred is None:
            all_pred = pred_cls
        else:
            all_pred = np.concatenate((all_pred, pred_cls), 0)
        if all_label is None:
            all_label = labels
        else:
            all_label = np.concatenate((all_label, labels), 0)
    loss /= len(data_loader)
    acc /= len(data_loader.dataset)
    # A = accuracy_score(labels.detach().cpu().numpy(), pred_cls.detach().cpu().numpy())
    # R = recall_score(labels.detach().cpu().numpy(), pred_cls.detach().cpu().numpy(), labels=[0, 1])

    f1_macro = f1_score(all_label, all_pred, labels=[0, 1], average='macro')


    print("Avg Loss = %.4f, Avg Accuracy = %.4f, Avg F1 = %.4f" % (loss, acc, f1_macro))
    # print("Avg Loss = %.4f, Avg Accuracy = %.4f" % (loss, acc))
    # test = pd.DataFrame(columns=['pred'], data=all_pred)
    # test.to_csv('save1.csv', index=False, sep=',')
    return loss, acc, f1_macro
#目标数据集上的源分类器对目标编码器的评估。
def eval(encoder, classifier, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    #criterion = nn.CrossEntropyLoss() 好像不会用到
    all_pred = None
    all_time = None
    # all_label = None
    # evaluate network
    for (reviews, mask, labels, times) in data_loader:
        reviews = make_cuda(reviews)
        mask = make_cuda(mask)
        times = make_cuda(times)
        #labels = make_cuda(labels)  好像不会用到

        with torch.no_grad():
            feat = encoder(reviews, mask)
            preds = classifier(feat)
        # loss += criterion(preds, labels).item()
        pred_cls = preds.data.max(1)[1]
        # acc += pred_cls.eq(labels.data).cpu().sum().item()
        pred_cls = pred_cls.detach().cpu().numpy()
        times = times.detach().cpu().numpy()
        if all_pred is None:
            all_pred = pred_cls
        else:
            all_pred = np.concatenate((all_pred, pred_cls), 0)#concatenate拼接
        if all_time is None:
            all_time = times
        else:
            all_time = np.concatenate((all_time, times), 0)
        # if all_label is None:
        #     all_label = labels
        # else:
        #     all_label = np.concatenate((all_label, labels), 0)
    # loss /= len(data_loader)
    # acc /= len(data_loader.dataset)
    # A = accuracy_score(labels.detach().cpu().numpy(), pred_cls.detach().cpu().numpy())
    # R = recall_score(labels.detach().cpu().numpy(), pred_cls.detach().cpu().numpy(), labels=[0, 1])

    # f1_macro = f1_score(all_label, all_pred, labels=[0, 1], average='macro')
    #
    #
    # print("Avg Loss = %.4f, Avg Accuracy = %.4f, Avg F1 = %.4f" % (loss, acc, f1_macro))
    # print("Avg Loss = %.4f, Avg Accuracy = %.4f" % (loss, acc))
    #test = pd.DataFrame(columns=['pred'], data=all_pred)
    dataDict = {'pres': all_pred,
                'time': all_time}
    test = pd.DataFrame(dataDict)
    test.to_csv('savetgt1_1.csv', index=False, sep=',')
    # return acc
    # return all_pred