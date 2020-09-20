import argparse
import logging
import os
import pickle
import random
import timeit
import json
import gc
import re

from utils import load_examples, get_jaccard_and_pred_ans, SentencePieceTokenizer

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    AdamW,
    AutoConfig,
    get_linear_schedule_with_warmup,
)

from tokenizers import ByteLevelBPETokenizer, Tokenizer, BertWordPieceTokenizer

from model import QuestionAnswering

from sklearn.model_selection import StratifiedKFold

# from transformers.data.metrics.squad_metrics import (
#     compute_predictions_log_probs,
#     compute_predictions_logits,
#     squad_evaluate,
# )
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def train(args, train_dataset, eval_dataset, model, tokenizer, n_fold):
    fold_best_model_dir = os.path.join(args.best_model_dir, f"fold-{n_fold}")
    if not os.path.exists(fold_best_model_dir):
        os.makedirs(fold_best_model_dir)
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    best_jac = 0.

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch"
    )

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):

            model.train()
            batch = tuple(t.to(args.device) for _, t in list(batch.items())[:5])

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart"]:
                del inputs["token_type_ids"]

            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
                if args.version_2_with_negative:
                    inputs.update({"is_impossible": batch[7]})
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                    )

            loss = model(**inputs, use_jaccard_soft=args.use_jaccard_soft)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

        jaccard_score = evaluate(args, model, eval_dataset)
        logger.info(f'fold-{n_fold}, global step {global_step} jaccard score: {jaccard_score:.4f},'
                    f' current best jaccard score {best_jac:.4f}')
        if jaccard_score > best_jac:
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(fold_best_model_dir)
            tokenizer_file = os.path.join(fold_best_model_dir, 'tokenizer.json')
            tokenizer.save(tokenizer_file)

            torch.save(args, os.path.join(fold_best_model_dir, "training_args.bin"))
            logger.info("Saving model checkpoint to %s", fold_best_model_dir)
            best_jac = jaccard_score

    # oof predict
    model = QuestionAnswering()(args.model_type).from_pretrained(fold_best_model_dir)
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    # inference!
    logger.info("***** Running oof inference *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    total_start_probs = []
    total_end_probs = []
    offsets = []
    for batch in tqdm(eval_dataloader, desc="inference"):
        model.eval()
        batch_for_forward = tuple(t.to(args.device) for _, t in list(batch.items())[:3])

        with torch.no_grad():
            inputs = {
                "input_ids": batch_for_forward[0],
                "attention_mask": batch_for_forward[1],
                "token_type_ids": batch_for_forward[2],
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart"]:
                del inputs["token_type_ids"]

            start_logits, end_logits = model(**inputs)
            start_probs = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().detach().numpy()
            total_start_probs.append(start_probs)
            total_end_probs.append(end_probs)
            offsets.append(batch['offsets'].numpy())

    total_start_probs = np.concatenate(total_start_probs, axis=0)
    total_end_probs = np.concatenate(total_end_probs, axis=0)
    total_probs = np.stack([total_start_probs, total_end_probs], axis=-1)
    offsets = np.concatenate(offsets, axis=0)
    assert total_probs.ndim == 3, total_probs.shape

    return best_jac, total_probs, offsets


def evaluate(args, model, dataset):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_jaccards = []
    start_time = timeit.default_timer()

    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch_for_forward = tuple(t.to(args.device) for _, t in list(batch.items())[:3])

        with torch.no_grad():
            inputs = {
                "input_ids": batch_for_forward[0],
                "attention_mask": batch_for_forward[1],
                "token_type_ids": batch_for_forward[2],
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart"]:
                del inputs["token_type_ids"]

            # XLNet and XLM use more arguments for their predictions
            # TODO 注意对 batch 的修改，batch 已经是 dict 了
            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                # for lang_id-sensitive xlm models
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                    )
            start_logits, end_logits = model(**inputs)
            # 这里只是分别根据最大的 logit 取出对应的索引，还存在一种可能这些索引超出了 context 所在的长度范围，直接使用它们
            # 可能拿到的不是最可能的答案范围
            start_idxs = start_logits.argmax(dim=-1)
            end_idxs = end_logits.argmax(dim=-1)
            for i in range(len(batch['orig_text'])):
                orig_text = batch['orig_text'][i]
                orig_selected_text = batch['orig_selected_text'][i]
                sentiment = batch['sentiment'][i]
                _, jaccard_score = get_jaccard_and_pred_ans(start_idxs[i], end_idxs[i],
                                                            batch['offsets'][i],
                                                            orig_text,
                                                            orig_selected_text,
                                                            sentiment)
                all_jaccards.append(jaccard_score)

    avg_jaccard_score = sum(all_jaccards) / len(all_jaccards) * 100

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    return avg_jaccard_score


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--first_level_models",
        default='./first_level_models/',
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--model_type",
        default='bert',
        type=str,
        help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
    )
    parser.add_argument(
        "--model_name_or_path",
        default='../models/bert/',
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--model_arch",
        default='bert-base-cased',
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--data_dir",
        default='/xhp/TweetSentimentExtr/data',
        type=str,
        help="The input data dir. Should contain the .json files for the task."
             + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--train_file",
        default='clean_train.csv',
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
             + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default='clean_valid.csv',
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
             + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=180,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
             "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=45, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=45, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=6.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--splits", default=5, type=int, help="cross validation folds"
    )
    parser.add_argument(
        "--alpha",
        default=0.3,
        type=float,
        help="used in jaccard-based soft labels"
    )
    parser.add_argument("--use_jaccard_soft", action="store_true", help="whether to use jaccard-based soft labels.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
             "A number of warnings are expected for a normal SQuAD evaluation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    args.best_model_dir = os.path.join(args.first_level_models, args.model_arch)
    best_model_dir_pat = re.compile(args.model_arch + r'.*')
    count = 0
    for d in os.listdir(args.first_level_models):
        if best_model_dir_pat.match(d):
            count += 1
    args.best_model_dir += '-' + str(count)
    os.makedirs(args.best_model_dir)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        output_hidden_states=True,
    )
    if args.model_type == 'roberta':
        tokenizer = ByteLevelBPETokenizer(
            vocab_file=os.path.join(args.model_name_or_path, 'vocab.json'),
            merges_file=os.path.join(args.model_name_or_path, 'merges.txt'),
            add_prefix_space=True,
            lowercase=args.do_lower_case
        )
    elif args.model_type in ['bert', 'distilbert']:
        tokenizer = BertWordPieceTokenizer(
            vocab_file=os.path.join(args.model_name_or_path, 'vocab.txt'),
            lowercase=args.do_lower_case
        )
    elif 'spiece.model' in os.listdir(args.model_name_or_path):
        tokenizer = SentencePieceTokenizer(args.model_name_or_path)

    logger.info("Training/evaluation parameters %s", args)

    train_df = pd.read_csv(os.path.join(args.data_dir, args.train_file))
    valid_df = pd.read_csv(os.path.join(args.data_dir, args.predict_file))
    total_df = pd.concat([train_df, valid_df], axis=0).reset_index(drop=True)

    score = 0
    oof_pred_probs = {}
    oof_offsets = {}
    splits = StratifiedKFold(n_splits=args.splits, random_state=args.seed,
                             shuffle=True).split(total_df, total_df.sentiment)
    for f, (train_idx, valid_idx) in enumerate(splits, start=1):
        logger.info(f' -> fold {f} starting ...')
        args.seed += f
        set_seed(args)
        train_df = total_df.iloc[train_idx]
        valid_df = total_df.iloc[valid_idx]
        valid_ids = valid_df.textID.tolist()
        args.train_file = os.path.join(args.data_dir, f'fold{f:d}_train.csv')
        args.predict_file = os.path.join(args.data_dir, f'fold{f:d}_valid.csv')
        train_df.to_csv(args.train_file, index=False, encoding='utf8')
        valid_df.to_csv(args.predict_file, index=False, encoding='utf8')

        train_dataset = load_examples(args, tokenizer, evaluate=False)
        eval_dataset = load_examples(args, tokenizer, evaluate=True)
        model = QuestionAnswering()(args.model_type).from_pretrained(
            args.model_name_or_path,
            config=config
        )
        model.to(args.device)
        fold_jaccard, fold_pred_probs, offsets = train(args, train_dataset, eval_dataset, model, tokenizer, f)
        oof_pred_probs.update({
            example_id: pred_probs for example_id, pred_probs in zip(valid_ids, fold_pred_probs)
        })
        oof_offsets.update({
            example_id: offset for example_id, offset in zip(valid_ids, offsets)
        })
        score += fold_jaccard / args.splits

        # del model, train_dataset, eval_dataset, train_df, valid_df
        # torch.cuda.empty_cache()
        # gc.collect()

    logger.info(' -> saving first level predict probability')

    with open(os.path.join(args.best_model_dir, 'start_end_probs.pickle'), 'wb') as f:
        pickle.dump(oof_pred_probs, f)

    with open(os.path.join(args.best_model_dir, 'offsets.pickle'), 'wb') as f:
        pickle.dump(oof_offsets, f)

    logger.info(f' -> cv jaccard score : {score:.3f}')
    eval_result = {'jaccard_score': score}
    with open(os.path.join(args.best_model_dir, 'eval_result.json'), 'w', encoding='utf8') as f:
        json.dump(eval_result, f)


if __name__ == "__main__":
    main()
