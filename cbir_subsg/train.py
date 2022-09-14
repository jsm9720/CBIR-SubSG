from cbir_subsg.config import parse_encoder
from cbir_subsg.test import validation
from common import utils
from common import models
from common import data
import torch.optim as optim
import torch.nn as nn
import torch
from sklearn.manifold import TSNE
import os
import argparse


def build_model(args):
    if args.method_type == "gnn":
        model = models.GnnEmbedder(1, args.hidden_dim, args)
    # elif args.method_type == "mlp":
    #     model = models.BaselineMLP(1, args.hidden_dim, args)
    model.to(utils.get_device())

    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path,
                                         map_location=utils.get_device()))
    return model


def make_data_source(args):
    if args.dataset == "scene":
        data_source = data.SceneDataSource("scene")
    return data_source


def train(args, model, dataset, data_source):
    """Train the embedding model.
    args: Commandline arguments
    dataset: Dataset of batch size
    data_source: DataSource class
    """
    scheduler, opt = utils.build_optimizer(args, model.parameters())
    if args.method_type == "gnn":
        clf_opt = optim.Adam(model.clf_model.parameters(), lr=args.lr)

    model.train()   # dorpout 및 batchnomalization 활성화
    model.zero_grad()   # 학습하기위한 Grad 저장할 변수 초기화
    pos_a, pos_b, pos_label = data_source.gen_batch(
        dataset, True)

    emb_as, emb_bs = model.emb_model(pos_a), model.emb_model(pos_b)
    labels = torch.tensor(pos_label).to(utils.get_device())

    intersect_embs = None
    pred = model(emb_as, emb_bs)
    loss = model.criterion(pred, intersect_embs, labels)
    print("loss", loss)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if scheduler:
        scheduler.step()

    # 분류하기 위해서
    if args.method_type == "gnn":
        with torch.no_grad():
            pred = model.predict(pred)  # 해당 부분은 학습에 반영하지 않겠다
        model.clf_model.zero_grad()
        pred = model.clf_model(pred.unsqueeze(1)).view(-1)
        criterion = nn.MSELoss()
        clf_loss = criterion(pred.float(), labels.float())
        clf_loss.backward()
        clf_opt.step()

    # acc = torch.mean((pred == labels).type(torch.float))

    return pred, labels, loss.item()


def train_loop(args):
    if not os.path.exists(os.path.dirname(args.model_path)):
        os.makedirs(os.path.dirname(args.model_path))
    if not os.path.exists("plots/"):
        os.makedirs("plots/")

    model = build_model(args)

    data_source = make_data_source(args)
    loaders = data_source.gen_data_loaders(
        args.batch_size, train=False)

    val = []
    batch_n = 0
    epoch = 1
    for e in range(epoch):
        for dataset in loaders:
            if args.test:
                mae = validation(args, model, dataset, data_source)
                val.append(mae)
            else:
                pred, labels, loss = train(
                    args, model, dataset, data_source)

                if batch_n % 100 == 0:
                    print(pred, pred.shape, sep='\n')
                    print(labels, labels.shape, sep='\n')
                    print("epoch :", e, "batch :", batch_n,
                          "loss :", loss)

                batch_n += 1

        if not args.test:
            print("Saving {}".format(args.model_path[:-5]+"_e"+str(e+1)+".pt"))
            torch.save(model.state_dict(),
                       args.model_path[:-5]+"_e"+str(e+1)+".pt")
        else:
            print(len(loaders))
            print(sum(val)/len(loaders))


def main(force_test=False):
    parser = argparse.ArgumentParser(description='Embedding arguments')

    utils.parse_optimizer(parser)
    parse_encoder(parser)
    args = parser.parse_args()

    if force_test:
        args.test = True

    train_loop(args)


if __name__ == '__main__':
    main()
