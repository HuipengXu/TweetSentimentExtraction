import argparse
import glob
import logging
import os
import re
import json
import random
import gc
import timeit

from utils import (
    load_examples,
    get_jaccard_and_pred_ans,
    SentencePieceTokenizer
)

import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    get_linear_schedule_with_warmup,
)

from tokenizers import (
    ByteLevelBPETokenizer,
    BertWordPieceTokenizer,
)

from model import QuestionAnswering

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

# os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6'

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


def train(args, train_dataset, eval_dataset, model, tokenizer):
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

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    args.logging_steps = int(args.logging_step_ratio * t_total)

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
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch"
    )
    # Added here for reproductibility
    set_seed(args)

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

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                # Save model checkpoint
                if global_step / t_total > 0.85 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    if not os.path.exists(output_dir): os.mkdir(output_dir)
                    model_to_save.save_pretrained(output_dir)
                    if args.model_type != 'albert':
                        tokenizer_file = os.path.join(output_dir, 'tokenizer.json')
                        tokenizer.save(tokenizer_file)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

    return global_step, tr_loss / global_step


def evaluate(args, model, eval_dataset, prefix=""):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = {}
    all_jaccards = []
    example_id_jaccard_map = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
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
                                                                             batch['offsets'][i],
                                                                             orig_text,
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
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(eval_dataset))

    return {f'jaccard': avg_jaccard_score}


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
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
        "--output_dir",
        default='./result1st/',
        type=str,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument(
        "--eval_model_dir",
        default=None,
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
        default='train.json',
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
             + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default='valid.json',
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
             + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
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
        "--alpha",
        default=0.3,
        type=float,
        help="used in jaccard-based soft labels"
    )
    parser.add_argument("--use_jaccard_soft", action="store_true", help="whether to use jaccard-based soft labels.")
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

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    # Set seed
    set_seed(args)

    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    if not os.path.exists(args.data_dir): os.makedirs(args.data_dir)

    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
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
    model = QuestionAnswering()(args.model_type).from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Training
    if args.do_train:
        train_dataset = load_examples(args, tokenizer, evaluate=False)
        eval_dataset = load_examples(args, tokenizer, evaluate=True)
        global_step, tr_loss = train(args, train_dataset, eval_dataset, model, tokenizer)
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
        model_to_save.save_pretrained(output_dir)
        if args.model_type != 'albert':
            tokenizer_file = os.path.join(output_dir, 'tokenizer.json')
            tokenizer.save(tokenizer_file)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = QuestionAnswering()(args.model_type).from_pretrained(output_dir)
        model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    if args.do_eval:
        if not args.do_train:
            args.output_dir = args.eval_model_dir
            eval_dataset = load_examples(args, tokenizer, evaluate=True)
        results = {}
        if args.do_train:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = [output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c)
                for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            logger.info(f"Evaluate the checkpoint: {checkpoint}")
            model = QuestionAnswering()(args.model_type).from_pretrained(checkpoint)
            model.to(args.device)

            # Evaluate
            args.output_dir = checkpoint
            result = evaluate(args, model, eval_dataset, prefix=global_step)
            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)

            del model
            torch.cuda.empty_cache()
            gc.collect()

        eval_result = "Results: {}".format(results)
        args.output_dir = output_dir
        logger.info(eval_result)
        with open(os.path.join(args.output_dir, 'eval_result.txt'), 'w', encoding='utf8') as f:
            f.write(eval_result)

        return results


if __name__ == "__main__":
    main()
