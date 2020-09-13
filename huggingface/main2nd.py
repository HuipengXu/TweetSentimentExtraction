import argparse
import glob
import logging
import os
import re
import pickle
import json
import random
import timeit

from model import CharRNN, TweetCharModel, CharCNN
from utils import (
    load_char_level_examples,
    get_jaccard_and_pred_ans,
    first_level_inference
)

import numpy as np
import pandas as pd
import torch
from torchcontrib.optim import SWA
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm, trange

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def train(args, train_dataset, model, tokenizer):
    base_output_dir = args.model_arch + '-result'
    output_dir_pattern = re.compile(base_output_dir + r'.*')
    count = 0
    for d in os.listdir(args.output_dir):
        if output_dir_pattern.match(d):
            count += 1
    args.output_dir += base_output_dir + '-' + str(count)
    os.makedirs(args.output_dir, exist_ok=True)

    """ Train the model """
    tb_writer = SummaryWriter(log_dir=args.output_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // len(train_dataloader) + 1
    else:
        t_total = len(train_dataloader) * args.num_train_epochs
    args.logging_steps = int(args.logging_step_ratio * t_total)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    # optimizer = SWA(optimizer)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel) = %d",
        args.train_batch_size
    )
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch"
    )
    # Added here for reproductibility
    set_seed(args)

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):

            model.train()
            batch = tuple(t.to(args.device) for _, t in list(batch.items()))

            inputs = {
                "start_probs": batch[0],
                "end_probs": batch[1],
                "char_ids": batch[2],
                "sentiment_ids": batch[3],
                "start_positions": batch[4],
                "end_positions": batch[5]
            }

            loss = model(**inputs)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training

            loss.backward()

            tr_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            # Log metrics
            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                # Only evaluate when single GPU otherwise metrics may not average well
                if args.evaluate_during_training:
                    results = evaluate(args, model, tokenizer)
                    for key, value in results.items():
                        tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                logging_loss = tr_loss

            # Save model checkpoint
            if global_step / t_total > 0.85 and global_step % args.save_steps == 0:
                output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                # Take care of distributed/parallel training
                model_to_save = model.module if hasattr(model, "module") else model
                if not os.path.exists(output_dir): os.mkdir(output_dir)
                torch.save(model_to_save.state_dict(),
                           os.path.join(output_dir, 'weight.pt'))

                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving model checkpoint to %s", output_dir)

                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        # if epoch + 1 >= args.swa_first_epoch:
        #     optimizer.update_swa()
        #     optimizer.swap_swa_sgd()
        #
        # if epoch + 1 >= args.swa_first_epoch:
        #     optimizer.swap_swa_sgd()

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    dataset = load_char_level_examples(args, tokenizer, evaluate=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = {}
    all_jaccards = []
    example_id_jaccard_map = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch_for_forward = tuple(t.to(args.device) for _, t in list(batch.items())[:4])

        with torch.no_grad():
            inputs = {
                "start_probs": batch_for_forward[0],
                "end_probs": batch_for_forward[1],
                "char_ids": batch_for_forward[2],
                "sentiment_ids": batch_for_forward[3]
            }

            start_logits, end_logits = model(**inputs)
            # 这里只是分别根据最大的 logit 取出对应的索引，还存在一种可能这些索引超出了 context 所在的长度范围，直接使用它们
            # 可能拿到的不是最可能的答案范围
            start_idxs = start_logits.argmax(dim=-1)
            end_idxs = end_logits.argmax(dim=-1)
            for i in range(len(batch['orig_text'])):
                orig_text = batch['orig_text'][i]
                orig_selected_text = batch['orig_selected_text'][i]
                sentiment = batch['sentiment'][i]
                example_id = batch['example_id'][i]
                pred_selected_text, jaccard_score = get_jaccard_and_pred_ans(start_idxs[i], end_idxs[i],
                                                                             None, orig_text,
                                                                             orig_selected_text,
                                                                             sentiment)
                all_results.update({
                    example_id: {
                        'text': orig_text,
                        'selected_text': orig_selected_text,
                        'pred_selected_text': pred_selected_text,
                        'sentiment': sentiment,
                        'jaccard_score': jaccard_score
                    }
                })
                example_id_jaccard_map.append((example_id, jaccard_score))
                all_jaccards.append(jaccard_score)

    avg_jaccard_score = sum(all_jaccards) / len(all_jaccards) * 100
    example_id_jaccard_map.sort(key=lambda x: x[1])
    topk_bad_cases = {id_: all_results[id_] for id_, _ in example_id_jaccard_map[:args.topk]}
    bad_cases_file = os.path.join(args.output_dir, f'bad_cases_{prefix}.json')
    logger.info(f"Saving topk bad cases in {bad_cases_file}")
    with open(bad_cases_file, 'w', encoding='utf8') as f:
        json.dump(topk_bad_cases, f, indent=4)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    return {f'jaccard': avg_jaccard_score}


def get_1st_data(first_level_model_dir, filename, train_ids, valid_ids):
    with open(os.path.join(first_level_model_dir, filename), 'rb') as f:
        data = pickle.load(f)
        train_data = {}
        valid_data = {}
        for id_ in train_ids:
            train_data.update({id_: data[id_]})
        for id_ in valid_ids:
            valid_data.update({id_: data[id_]})
        with open(os.path.join(first_level_model_dir,
                               'train_' + filename), 'wb') as f1:
            pickle.dump(train_data, f1)
        with open(os.path.join(first_level_model_dir,
                               'valid_' + filename), 'wb') as f2:
            pickle.dump(valid_data, f2)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--first_level_models",
        default='./first_level_models/',
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--model_type",
        default='',
        type=str,
        help="first level model type",
    )
    parser.add_argument(
        "--model_arch",
        default='cnn',
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        default='./char_level_results/',
        type=str,
        help="The output directory where the model checkpoints and predictions will be written.",
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
        "--topk",
        default=50,
        type=int,
        help="top k bad cases"
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument("--char_embed_dim", default=128, type=int)
    parser.add_argument("--n_models", default=1, type=int)
    parser.add_argument("--lstm_hidden_size", default=256, type=int)
    parser.add_argument("--cnn_dim", default=32, type=int)
    parser.add_argument("--kernel_size", default=3, type=int)
    parser.add_argument("--sentiment_dim", default=128, type=int)
    parser.add_argument("--encode_size", default=256, type=int)

    parser.add_argument("--per_gpu_train_batch_size", default=45, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=45, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=6.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--swa_first_epoch", default=5.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--alpha",
        default=0.3,
        type=float,
        help="used in jaccard-based soft labels"
    )
    parser.add_argument("--use_jaccard_soft", action="store_true", help="whether to use jaccard-based soft labels.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_step_ratio", type=float, default=0.1, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Set seed
    set_seed(args)

    args.cv_probs = True

    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    if not os.path.exists(args.data_dir): os.makedirs(args.data_dir)

    logger.info("starting use 1st level model to inference to get position probability")
    first_level_model = os.path.join(args.first_level_models, args.model_type)
    args.first_level_model = first_level_model
    if not args.cv_probs:
        first_level_inference(first_level_model, data_type='train')
        first_level_inference(first_level_model, data_type='valid')
    else:
        train_df = pd.read_csv(os.path.join(args.data_dir, args.train_file))
        valid_df = pd.read_csv(os.path.join(args.data_dir, args.predict_file))
        train_ids = train_df.textID.tolist()
        valid_ids = valid_df.textID.tolist()
        get_1st_data(args.first_level_model, 'start_end_probs.pickle', train_ids, valid_ids)
        get_1st_data(args.first_level_model, 'offsets.pickle', train_ids, valid_ids)

    tokenizer = Tokenizer(num_words=None, char_level=True, oov_token='UNK', lower=True)
    train_texts = pd.read_csv(os.path.join(args.data_dir, args.train_file)).text.tolist()
    tokenizer.fit_on_texts(train_texts)
    args.vocab_size = len(tokenizer.word_index) + 1

    logger.info("Training/evaluation parameters %s", args)

    if args.model_arch == 'LSTM':
        model = CharRNN(char_vocab_size=args.vocab_size,
                        char_embed_dim=args.char_embed_dim,
                        n_models=args.n_models,
                        lstm_hidden_size=args.lstm_hidden_size,
                        sentiment_dim=args.sentiment_dim,
                        encode_size=args.encode_size)
    elif args.model_arch == 'CNN':
        model = CharCNN(char_vocab_size=args.vocab_size,
                        char_embed_dim=args.char_embed_dim,
                        n_models=args.n_models,
                        cnn_dim=args.cnn_dim,
                        sentiment_dim=args.sentiment_dim,
                        encode_size=args.encode_size,
                        kernel_size=args.kernel_size)
    model.to(args.device)

    # Training
    if args.do_train:
        train_dataset = load_char_level_examples(args, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    output_dir = ''
    # Save the trained model and the tokenizer
    if args.do_train:
        output_dir = os.path.join(args.output_dir, "checkpoint-final")
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        if not os.path.exists(output_dir): os.mkdir(output_dir)
        model_weight_file = os.path.join(output_dir, 'weight.pt')
        torch.save(model_to_save.state_dict(), model_weight_file)
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model.load_state_dict(torch.load(model_weight_file))
        model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    if args.do_eval:
        results = {}
        if args.do_train:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = [output_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c)
                    for c in sorted(glob.glob(args.output_dir + "/**/" + 'weight.pt', recursive=True))
                )
                logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
        else:
            logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
            checkpoints = [args.model_name_or_path]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            logger.info(f"Evaluate the checkpoint: {checkpoint}")
            model.load_state_dict(torch.load(os.path.join(checkpoint, 'weight.pt')))
            model.to(args.device)

            # Evaluate
            args.output_dir = checkpoint
            result = evaluate(args, model, tokenizer, prefix=global_step)
            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)

        eval_result = "Results: {}".format(results)
        args.output_dir = output_dir
        logger.info(eval_result)
        with open(os.path.join(args.output_dir, 'eval_result.txt'), 'w', encoding='utf8') as f:
            f.write(eval_result)

        return results


if __name__ == "__main__":
    main()
