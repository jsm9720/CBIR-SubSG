from common import utils
from collections import defaultdict
from datetime import datetime
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import torch
import numpy as np
import sys

USE_ORCA_FEATS = False  # whether to use orca motif counts along with embeddings
MAX_MARGIN_SCORE = 1e9  # a very large margin score to given orca constraints


def validation(args, model, dataset, data_source):
    # test on new motifs
    model.eval()
    # all_raw_preds, all_preds, all_labels, all_pre_preds = [], [], [], []
    pos_a, pos_b, pos_label = data_source.gen_batch(
        dataset, True)

    with torch.no_grad():
        emb_as, emb_bs = model.emb_model(pos_a), model.emb_model(pos_b)

        labels = torch.tensor(pos_label).to(utils.get_device())

        pred = model(emb_as, emb_bs)
        raw_pred = model.predict(pred)
        pre_pred = raw_pred.clone().detach()

    #         if args.method_type == "order":
    #             pred = model.clf_model(raw_pred.unsqueeze(1)).view(-1)
    #             # pred = pred.argmax(dim=-1)
    #             raw_pred *= -1
    #         elif args.method_type == "ensemble":
    #             pred = torch.stack([m.clf_model(
    #                 raw_pred.unsqueeze(1)).argmax(dim=-1) for m in model.models])
    #             for i in range(pred.shape[1]):
    #                 print(pred[:, i])
    #             pred = torch.min(pred, dim=0)[0]
    #             raw_pred *= -1
    #         elif args.method_type == "mlp":
    #             raw_pred = raw_pred[:, 1]
    #             pred = pred.argmax(dim=-1)
    #     all_raw_preds.append(raw_pred)
    #     all_pre_preds.append(pre_pred)
    #     all_preds.append(pred)
    #     all_labels.append(labels)
    # pre_pred = torch.cat(all_pre_preds, dim=-1)
    # pred = torch.cat(all_preds, dim=-1)
    # labels = torch.cat(all_labels, dim=-1)
    # print(pre_pred.shape)
    # print(pre_pred)
    # print(pred.shape)
    # print(pred)
    # print(labels.shape)
    # print(labels)
    # print("loss :", torch.sum(torch.abs(labels-pre_pred)).item())
    mae = mean_absolute_error(labels.cpu(), pre_pred.cpu())
    # print("loss(MAE) :", mae)
    return mae
    '''
    raw_pred = torch.cat(all_raw_preds, dim=-1)
    acc = torch.mean((pred == labels).type(torch.float))
    prec = (torch.sum(pred * labels).item() / torch.sum(pred).item() if
            torch.sum(pred) > 0 else float("NaN"))
    recall = (torch.sum(pred * labels).item() /
              torch.sum(labels).item() if torch.sum(labels) > 0 else
              float("NaN"))
    labels = labels.detach().cpu().numpy()
    raw_pred = raw_pred.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    # auroc = roc_auc_score(labels, raw_pred)
    auroc = 0.01
    # avg_prec = average_precision_score(labels, raw_pred)
    avg_prec = 0.01
    tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()
    if verbose:
        import matplotlib.pyplot as plt
        precs, recalls, threshs = precision_recall_curve(labels, raw_pred)
        plt.plot(recalls, precs)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.savefig("plots/precision-recall-curve.png")
        print("Saved PR curve plot in plots/precision-recall-curve.png")

    print("\n{}".format(str(datetime.now())))
    print("Validation. Epoch {}. Acc: {:.4f}. "
          "P: {:.4f}. R: {:.4f}. AUROC: {:.4f}. AP: {:.4f}.\n     "
          "TN: {}. FP: {}. FN: {}. TP: {}".format(epoch,
                                                  acc, prec, recall, auroc, avg_prec,
                                                  tn, fp, fn, tp))

    if not args.test:
        # logger.add_scalar("Accuracy/test", acc, batch_n)
        # logger.add_scalar("Precision/test", prec, batch_n)
        # logger.add_scalar("Recall/test", recall, batch_n)
        # logger.add_scalar("AUROC/test", auroc, batch_n)
        # logger.add_scalar("AvgPrec/test", avg_prec, batch_n)
        # logger.add_scalar("TP/test", tp, batch_n)
        # logger.add_scalar("TN/test", tn, batch_n)
        # logger.add_scalar("FP/test", fp, batch_n)
        # logger.add_scalar("FN/test", fn, batch_n)
        print("Saving {}".format(args.model_path))
        torch.save(model.state_dict(), args.model_path)


    if verbose:
        conf_mat_examples = defaultdict(list)
        idx = 0
        for pos_a, pos_b, neg_a, neg_b in test_pts:
            if pos_a:
                pos_a = pos_a.to(utils.get_device())
                pos_b = pos_b.to(utils.get_device())
            neg_a = neg_a.to(utils.get_device())
            neg_b = neg_b.to(utils.get_device())
            for list_a, list_b in [(pos_a, pos_b), (neg_a, neg_b)]:
                if not list_a:
                    continue
                for a, b in zip(list_a.G, list_b.G):
                    correct = pred[idx] == labels[idx]
                    conf_mat_examples[correct, pred[idx]].append((a, b))
                    idx += 1
    '''


if __name__ == "__main__":
    from subgraph_matching.train import main
    main(force_test=True)
