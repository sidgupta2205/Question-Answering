"""Test a model and generate submission CSV.

Usage:
    > python test.py --split SPLIT --load_path PATH --name NAME
    where
    > SPLIT is either "dev" or "test"
    > PATH is a path to a checkpoint (e.g., save/train/model-01/best.pth.tar)
    > NAME is a name to identify the test run

''
"""

import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import util

from args import get_test_args
from collections import OrderedDict
from json import dumps
from models import BiDAF, QANet
from milestone_bidaf import MilestoneBiDAF
from torch_attn_bidaf import KaimingBiDAF
from cnn_0309_qanet import CNNQANet0309
from os.path import join
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD


def main(args):
    # Set up logging
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=False)
    log = util.get_logger(args.save_dir, args.name)
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    device, gpu_ids = util.get_available_devices()
    args.batch_size *= max(1, len(gpu_ids))

    # Get embeddings
    log.info('Loading embeddings...')
    word_vectors = util.torch_from_json(args.word_emb_file)
    char_vectors = util.torch_from_json(args.char_emb_file)

    # Get model
    log.info('Building model...')

    models = [
        MilestoneBiDAF(char_vectors=char_vectors,
                       word_vectors=word_vectors,
                       hidden_size=args.hidden_size),
        QANet(char_vectors=char_vectors,
              word_vectors=word_vectors,
              hidden_size=128,
              project=True,
              use_char_cnn=False,
              use_seq=False),
        CNNQANet0309(char_vectors=char_vectors,
                     word_vectors=word_vectors,
                     hidden_size=128,
                     project=True,
                     use_char_cnn=True),
        BiDAF(char_vectors=char_vectors,
              word_vectors=word_vectors,
              hidden_size=args.hidden_size,
              use_char_cnn=False),
        KaimingBiDAF(char_vectors=char_vectors,
                     word_vectors=word_vectors,
                     hidden_size=128,
                     use_char_cnn=False,
                     num_heads=8),
        BiDAF(char_vectors=char_vectors,
              word_vectors=word_vectors,
              hidden_size=args.hidden_size,
              use_char_cnn=False),
        QANet(char_vectors=char_vectors,
              word_vectors=word_vectors,
              hidden_size=128,
              project=True,
              use_char_cnn=False,
              use_seq=True),
    ]

    model_load_paths = [
        'save/train/bidaf-char-emb-200-proj-01/best.pth.tar',
        'save/train/qanet-cnn0311202210/qanet-base-06/best.pth.tar',
        'save/train/qanet-cnn-0310202209/cnn/qanet-base-01/best.pth.tar',
        'save/train/qanet-cnn0311202210/bidaf/nocnn/no-kaiming-init/hs100-02/best.pth.tar',
        'save/train/qanet-cnn0311202210/bidaf/nocnn/kaiming-init-02/best.pth.tar',
        'save/train/ensemble/bidaf/nocnn/hs100-01/best.pth.tar',
        'save/train/ensemble/nocnn/qanet-base-stochastic-depth-03/best.pth.tar',
    ]

    for model, model_load_path in zip(models, model_load_paths):
        model = nn.DataParallel(model, gpu_ids)
        model = util.load_model(model, model_load_path, gpu_ids, return_step=False)
        model = model.to(device)
        model.eval()

    # Get data loader
    log.info('Building dataset...')
    record_file = vars(args)[f'{args.split}_record_file']
    dataset = SQuAD(record_file, args.use_squad_v2)
    data_loader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn)

    # Evaluate
    log.info(f'Evaluating on {args.split} split...')
    nll_meter = util.AverageMeter()
    pred_dict = {}  # Predictions for TensorBoard
    sub_dict = {}   # Predictions for submission
    eval_file = vars(args)[f'{args.split}_eval_file']
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), \
            tqdm(total=len(dataset)) as progress_bar:
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            cc_idxs = cc_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            qc_idxs = qc_idxs.to(device)
            batch_size = cw_idxs.size(0)

            # Forward
            p1s, p2s = [], []

            for model in models:
                log_p1, log_p2 = model(cw_idxs, cc_idxs, qw_idxs, qc_idxs)
                y1, y2 = y1.to(device), y2.to(device)
                loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
                nll_meter.update(loss.item(), batch_size)

                # Get F1 and EM scores
                p1, p2 = log_p1.exp(), log_p2.exp()
                p1s.append(p1)
                p2s.append(p2)

            starts, ends = util.discretize_ensemble(p1s, p2s, args.max_ans_len, args.use_squad_v2)

            # Log info
            progress_bar.update(batch_size)
            if args.split != 'test':
                # No labels for the test set, so NLL would be invalid
                progress_bar.set_postfix(NLL=nll_meter.avg)

            idx2pred, uuid2pred = util.convert_tokens(gold_dict,
                                                      ids.tolist(),
                                                      starts.tolist(),
                                                      ends.tolist(),
                                                      args.use_squad_v2)
            pred_dict.update(idx2pred)
            sub_dict.update(uuid2pred)

    # Log results (except for test set, since it does not come with labels)
    if args.split != 'test':
        results = util.eval_dicts(gold_dict, pred_dict, args.use_squad_v2, args.save_dir)
        results_list = [('NLL', nll_meter.avg),
                        ('F1', results['F1']),
                        ('EM', results['EM'])]
        if args.use_squad_v2:
            results_list.append(('AvNA', results['AvNA']))
        results = OrderedDict(results_list)

        # Log to console
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
        log.info(f'{args.split.title()} {results_str}')

        # Log to TensorBoard
        tbx = SummaryWriter(args.save_dir)
        util.visualize(tbx,
                       pred_dict=pred_dict,
                       eval_path=eval_file,
                       step=0,
                       split=args.split,
                       num_visuals=args.num_visuals)

    # Write submission file
    sub_path = join(args.save_dir, args.split + '_' + args.sub_file)
    log.info(f'Writing submission file to {sub_path}...')
    with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
        csv_writer = csv.writer(csv_fh, delimiter=',')
        csv_writer.writerow(['Id', 'Predicted'])
        for uuid in sorted(sub_dict):
            csv_writer.writerow([uuid, sub_dict[uuid]])


if __name__ == '__main__':
    main(get_test_args())
