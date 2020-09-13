import re
import string
import collections
import logging
import math
import json
import os
import timeit
from tqdm import tqdm
import pickle

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from tensorflow.keras.preprocessing.text import Tokenizer as KTokenizer
from tokenizers import Tokenizer as TTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from transformers.tokenization_bert import BasicTokenizer
from transformers import AutoConfig

from model import QuestionAnswering

logger = logging.getLogger(__name__)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_jaccard(a_gold, a_pred):
    # gold_toks = get_tokens(a_gold)
    # pred_toks = get_tokens(a_pred)
    gold_toks = set(a_gold.strip().split())
    pred_toks = set(a_pred.strip().split())
    common = gold_toks & pred_toks
    num_same = len(common)
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    jaccard = float(num_same) / (len(gold_toks) + len(pred_toks) - num_same)
    return jaccard


def get_raw_scores(examples, preds, head_jaccard_file, tail_jaccard_file):
    """
    Computes the exact and jaccard scores from the examples and the model predictions
    """
    exact_scores = {}
    jaccard_scores = {}

    for example in examples:
        qas_id = example.qas_id
        gold_answers = [answer["text"] for answer in example.answers if normalize_answer(answer["text"])]

        if not gold_answers:
            # For unanswerable questions, only correct answer is empty string
            gold_answers = [""]

        if qas_id not in preds:
            print("Missing prediction for %s" % qas_id)
            continue

        prediction = preds[qas_id]['pred_selected']
        exact_scores[qas_id] = max(compute_exact(a, prediction) for a in gold_answers)
        jaccard_scores[qas_id] = max(compute_jaccard(a, prediction) for a in gold_answers)

    sorted_jaccard_scores = sorted(jaccard_scores.items(), key=lambda x: x[1])
    head_jaccard = {}
    tail_jaccard = {}
    for id_, score in sorted_jaccard_scores[:100]:
        preds[id_]['jaccard_score'] = score
        tail_jaccard[id_] = preds[id_]

    for id_, score in sorted_jaccard_scores[-100:]:
        preds[id_]['jaccard_score'] = score
        head_jaccard[id_] = preds[id_]

    with open(head_jaccard_file, "w") as writer:
        writer.write(json.dumps(head_jaccard, indent=4) + "\n")

    with open(tail_jaccard_file, "w") as writer:
        writer.write(json.dumps(tail_jaccard, indent=4) + "\n")

    return exact_scores, jaccard_scores


def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
    new_scores = {}
    for qid, s in scores.items():
        pred_na = na_probs[qid] > na_prob_thresh
        if pred_na:
            new_scores[qid] = float(not qid_to_has_ans[qid])
        else:
            new_scores[qid] = s
    return new_scores


def make_eval_dict(exact_scores, jaccard_scores, qid_list=None):
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores.values()) / total),
                ("jaccard", 100.0 * sum(jaccard_scores.values()) / total),
                ("total", total),
            ]
        )
    else:
        total = len(qid_list)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores[k] for k in qid_list) / total),
                ("jaccard", 100.0 * sum(jaccard_scores[k] for k in qid_list) / total),
                ("total", total),
            ]
        )


def merge_eval(main_eval, new_eval, prefix):
    for k in new_eval:
        main_eval["%s_%s" % (prefix, k)] = new_eval[k]


def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    for _, qid in enumerate(qid_list):
        if qid not in scores:
            continue
        if qid_to_has_ans[qid]:
            diff = scores[qid]
        else:
            if preds[qid]:
                diff = -1
            else:
                diff = 0
        cur_score += diff
        if cur_score > best_score:
            best_score = cur_score
            best_thresh = na_probs[qid]
    return 100.0 * best_score / len(scores), best_thresh


def find_all_best_thresh(main_eval, preds, exact_raw, jaccard_raw, na_probs, qid_to_has_ans):
    best_exact, exact_thresh = find_best_thresh(preds, exact_raw, na_probs, qid_to_has_ans)
    best_jaccard, jaccard_thresh = find_best_thresh(preds, jaccard_raw, na_probs, qid_to_has_ans)

    main_eval["best_exact"] = best_exact
    main_eval["best_exact_thresh"] = exact_thresh
    main_eval["best_jaccard"] = best_jaccard
    main_eval["best_jaccard_thresh"] = jaccard_thresh


def squad_evaluate(examples, preds, head_jaccard_file, tail_jaccard_file,
                   no_answer_probs=None, no_answer_probability_threshold=1.0):
    qas_id_to_has_answer = {example.qas_id: bool(example.answers) for example in examples}
    has_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if has_answer]
    no_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if not has_answer]

    if no_answer_probs is None:
        no_answer_probs = {k: 0.0 for k in preds}

    exact, jaccard = get_raw_scores(examples, preds, head_jaccard_file, tail_jaccard_file)

    exact_threshold = apply_no_ans_threshold(
        exact, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold
    )
    jaccard_threshold = apply_no_ans_threshold(jaccard, no_answer_probs, qas_id_to_has_answer,
                                               no_answer_probability_threshold)

    evaluation = make_eval_dict(exact_threshold, jaccard_threshold)

    if has_answer_qids:
        has_ans_eval = make_eval_dict(exact_threshold, jaccard_threshold, qid_list=has_answer_qids)
        merge_eval(evaluation, has_ans_eval, "HasAns")

    if no_answer_qids:
        no_ans_eval = make_eval_dict(exact_threshold, jaccard_threshold, qid_list=no_answer_qids)
        merge_eval(evaluation, no_ans_eval, "NoAns")

    if no_answer_probs:
        find_all_best_thresh(evaluation, preds, exact, jaccard, no_answer_probs, qas_id_to_has_answer)

    return evaluation


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info("Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'", orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position: (orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def compute_predictions_logits(
        all_examples,
        all_features,
        all_results,
        n_best_size,
        max_answer_length,
        do_lower_case,
        do_test,
        output_prediction_file,
        submission_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        verbose_logging,
        version_2_with_negative,
        null_score_diff_threshold,
        tokenizer,
):
    """Write final predictions to the json file and log-odds of null if needed."""
    if output_prediction_file:
        logger.info(f"Writing predictions to: {output_prediction_file}")
    if output_nbest_file:
        logger.info(f"Writing nbest to: {output_nbest_file}")
    if output_null_log_odds_file and version_2_with_negative:
        logger.info(f"Writing null_log_odds to: {output_null_log_odds_file}")

    # 每一个 example 对应的所有 features
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    # 每一个 id 对应的 result，应该是和 all_features 等长的
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_logit", "end_logit"]
    )

    all_predictions = collections.OrderedDict()
    sub_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        example_text = ' '.join(example.doc_tokens)
        features = example_index_to_features[example_index]

        # 每一个 example 可能的预测值
        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            # 获取 最大的 n_best 个 logits 对应的索引
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    # token_to_orig_map 保证了 start_index 不在 question 范围内
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    # score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"] ，
                    # 以此来判断当前 token 位置是否是最大上下文位置
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                        )
                    )
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit,
                )
            )

        # 因为概率由 exp 函数转换而来，start_p * end_p 最大也就意味着 start_logit + end_logit 最大
        prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "gold_selected", "start_logit", "end_logit"]
        )

        seen_predictions = {}
        nbest = []
        # 一个 example 的 prediction 选择
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index: (pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start: (orig_doc_end + 1)]

                tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

                # tok_text = " ".join(tok_tokens)
                #
                # # De-tokenize WordPieces that have been split off.
                # tok_text = tok_text.replace(" ##", "")
                # tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(_NbestPrediction(text=final_text, gold_selected=example.answers[0]['text'],
                                          start_logit=pred.start_logit, end_logit=pred.end_logit))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(_NbestPrediction(text="", gold_selected=example.answers[0]['text'],
                                              start_logit=null_start_logit, end_logit=null_end_logit))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(0, _NbestPrediction(text="empty", gold_selected=example.answers[0]['text'],
                                                 start_logit=0.0, end_logit=0.0))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text="empty", gold_selected=example.answers[0]['text'],
                                          start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)  # 计算的概率是 nbest 中的分布概率，并不是基于整个段落

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output['text'] = example_text
            output['sentiment'] = example.question_text
            output['gold_selected'] = entry.gold_selected
            if example.question_text == 'neutral':
                output["pred_selected"] = example_text
            else:
                output["pred_selected"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]
            sub_predictions[example.qas_id] = nbest_json[0]['pred_selected']
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
                sub_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = nbest_json[0]
                sub_predictions[example.qas_id] = best_non_null_entry.text
        all_nbest_json[example.qas_id] = nbest_json

    if do_test:
        submission_df = pd.DataFrame(sub_predictions.items(), columns=['textID', 'selected_text'])
        submission_df.to_csv(submission_prediction_file, index=False, encoding='utf8')
    else:
        if output_prediction_file:
            with open(output_prediction_file, "w") as writer:
                writer.write(json.dumps(all_predictions, indent=4) + "\n")

        if output_nbest_file:
            with open(output_nbest_file, "w") as writer:
                writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

        if output_null_log_odds_file and version_2_with_negative:
            with open(output_null_log_odds_file, "w") as writer:
                writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions


def compute_predictions_log_probs(
        all_examples,
        all_features,
        all_results,
        n_best_size,
        max_answer_length,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        start_n_top,
        end_n_top,
        version_2_with_negative,
        tokenizer,
        verbose_logging,
):
    """ XLNet write prediction logic (more complex than Bert's).
        Write final predictions to the json file and log-odds of null if needed.

        Requires utils_squad_evaluate.py
    """
    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_log_prob", "end_log_prob"]
    )

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_log_prob", "end_log_prob"]
    )

    logger.info("Writing predictions to: %s", output_prediction_file)
    # logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive

        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]

            cur_null_score = result.cls_logits

            # if we could have irrelevant answers, get the min score of irrelevant
            score_null = min(score_null, cur_null_score)

            for i in range(start_n_top):
                for j in range(end_n_top):
                    start_log_prob = result.start_logits[i]
                    start_index = result.start_top_index[i]

                    j_index = i * end_n_top + j

                    end_log_prob = result.end_logits[j_index]
                    end_index = result.end_top_index[j_index]

                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= feature.paragraph_len - 1:
                        continue
                    if end_index >= feature.paragraph_len - 1:
                        continue

                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue

                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_log_prob=start_log_prob,
                            end_log_prob=end_log_prob,
                        )
                    )

        prelim_predictions = sorted(
            prelim_predictions, key=lambda x: (x.start_log_prob + x.end_log_prob), reverse=True
        )

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]

            # XLNet un-tokenizer
            # Let's keep it simple for now and see if we need all this later.
            #
            # tok_start_to_orig_index = feature.tok_start_to_orig_index
            # tok_end_to_orig_index = feature.tok_end_to_orig_index
            # start_orig_pos = tok_start_to_orig_index[pred.start_index]
            # end_orig_pos = tok_end_to_orig_index[pred.end_index]
            # paragraph_text = example.paragraph_text
            # final_text = paragraph_text[start_orig_pos: end_orig_pos + 1].strip()

            # Previously used Bert untokenizer
            tok_tokens = feature.tokens[pred.start_index: (pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start: (orig_doc_end + 1)]
            tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            if hasattr(tokenizer, "do_lower_case"):
                do_lower_case = tokenizer.do_lower_case
            else:
                do_lower_case = tokenizer.do_lowercase_and_remove_accent

            final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)

            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(text=final_text, start_log_prob=pred.start_log_prob, end_log_prob=pred.end_log_prob)
            )

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text="", start_log_prob=-1e6, end_log_prob=-1e6))

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_log_prob + entry.end_log_prob)
            if not best_non_null_entry:
                best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_log_prob"] = entry.start_log_prob
            output["end_log_prob"] = entry.end_log_prob
            nbest_json.append(output)

        assert len(nbest_json) >= 1
        assert best_non_null_entry is not None

        score_diff = score_null
        scores_diff_json[example.qas_id] = score_diff
        # note(zhiliny): always predict best_non_null_entry
        # and the evaluation script will search for the best threshold
        all_predictions[example.qas_id] = best_non_null_entry.text

        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions


def one_hot(idx, seq_len):
    idxs = [0.] * seq_len
    idxs[idx] = 1.
    return idxs


def jaccard_based_soft_labels(alpha, tok_text_length, orig_start, orig_end, tok_selected_text_ids_str, tok_text_ids,
                              example_id, add_square_item=True):
    jaccard_start_scores = []
    jaccard_end_scores = []
    orig_starts = one_hot(orig_start, tok_text_length)
    orig_ends = one_hot(orig_end, tok_text_length)

    for i in range(tok_text_length):
        jaccard_start_score = compute_jaccard(tok_selected_text_ids_str,
                                              ' '.join(str(id_) for id_ in tok_text_ids[i: orig_end + 1]))
        jaccard_end_score = compute_jaccard(tok_selected_text_ids_str,
                                            ' '.join(str(id_) for id_ in tok_text_ids[orig_start: i]))
        if add_square_item:
            jaccard_start_score += jaccard_start_score ** 2
            jaccard_end_score += jaccard_end_score ** 2
        jaccard_start_scores.append(jaccard_start_score)
        jaccard_end_scores.append(jaccard_end_score)

    total_jaccard_start_score = sum(jaccard_start_scores)
    total_jaccard_end_score = sum(jaccard_end_scores)

    soft_start_labels = []
    soft_end_labels = []
    for i, (jaccard_start_score, jaccard_end_score) in enumerate(
            zip(jaccard_start_scores, jaccard_end_scores)):
        try:
            soft_start_label = alpha * orig_starts[i] + (1 - alpha) * jaccard_start_score / total_jaccard_start_score
            soft_start_labels.append(soft_start_label)
            soft_end_label = alpha * orig_ends[i] + (1 - alpha) * jaccard_end_score / total_jaccard_end_score
            soft_end_labels.append(soft_end_label)
        except ZeroDivisionError:
            print(example_id)

    soft_start_labels = np.array(soft_start_labels)
    soft_start_labels /= soft_start_labels.sum()
    soft_end_labels = np.array(soft_end_labels)
    soft_end_labels /= soft_end_labels.sum()

    return soft_start_labels, soft_end_labels


# TODO 用自己的方法试试，对 text 和 selected_text 同时 tokenize，然后在 text 中 find
def process_data(tokenizer, text, selected_text, extended_selected_text, sentiment, example_id,
                 model_type, max_length, alpha=1., use_jaccard_soft=False):
    text = ' '.join(str(text).split())
    selected_text = ' '.join(str(selected_text).split())
    if model_type == 'roberta':
        text = ' ' + text
        selected_text = ' ' + selected_text
    selected_text_length = len(selected_text)

    start_idx = end_idx = None
    for i, c in enumerate(text):
        if model_type == 'roberta' and c == selected_text[1] and \
                ' ' + text[i: i + selected_text_length - 1] == selected_text:
            start_idx = i
            end_idx = i + selected_text_length - 2
            break
        elif model_type == 'bert' and c == selected_text[0] and \
                text[i: i + selected_text_length] == selected_text:
            start_idx = i
            end_idx = i + selected_text_length - 1
            break

    char_targets = np.zeros(len(text))
    if start_idx is not None and end_idx is not None:
        char_targets[start_idx: end_idx + 1] = 1

    tok_text = tokenizer.encode(text)
    if model_type == 'roberta':
        input_ids_orig = tok_text.ids
        text_offsets = tok_text.offsets
    elif model_type == 'bert':
        input_ids_orig = tok_text.ids[1:-1]
        text_offsets = tok_text.offsets[1:-1]

    target_idxs = []
    for i, (offset1, offset2) in enumerate(text_offsets):
        if char_targets[offset1: offset2].sum() > 0:
            target_idxs.append(i)

    tok_start_idx = target_idxs[0]
    tok_end_idx = target_idxs[-1]

    if model_type == 'roberta':
        sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}
        input_ids = [0] + [sentiment_id[sentiment]] + [2] + [2] + input_ids_orig + [2]
        text_offsets = [(0, 0)] * 4 + text_offsets + [(0, 0)]  # 0 用来表示特殊符号的范围，比如 <s></s>，(0, 0) 表示不可用
        token_type_ids = [0] * len(input_ids)
        tok_start_idx += 4
        tok_end_idx += 4
    elif model_type == 'bert':
        sentiment_id = {'positive': 3112, 'negative': 4366, 'neutral': 8795}
        input_ids = [101] + [sentiment_id[sentiment]] + [102] + input_ids_orig + [102]
        token_type_ids = [0] * 3 + [1] * (len(input_ids_orig) + 1)
        text_offsets = [(0, 0)] * 3 + text_offsets + [(0, 0)]
        tok_start_idx += 3
        tok_end_idx += 3

    attention_mask = [1] * len(input_ids)

    padding_length = max_length - len(input_ids)
    if padding_length > 0:
        if model_type == 'roberta':
            pad_id = 1
        elif model_type == 'bert':
            pad_id = 0
        input_ids += [pad_id] * padding_length
        attention_mask += [0] * padding_length
        token_type_ids += [0] * padding_length
        text_offsets += [(0, 0)] * padding_length
    elif padding_length < 0:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        token_type_ids = token_type_ids[:max_length]
        text_offsets = text_offsets[:max_length]
        tok_end_idx = min(tok_end_idx, max_length - 1)
        tok_start_idx = min(tok_start_idx, max_length - 1)

    data = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'text_offsets': text_offsets,
        'selected_text': selected_text,
        'text': text,
        'sentiment': sentiment,
        'tok_start_idx': tok_start_idx,
        'tok_end_idx': tok_end_idx
    }

    if use_jaccard_soft:
        tok_selected_text_ids = tokenizer.encode(extended_selected_text).ids
        tok_selected_text_ids_str = ' '.join(str(id_) for id_ in tok_selected_text_ids)
        soft_start_labels, soft_end_labels = jaccard_based_soft_labels(
            alpha, len(input_ids), tok_start_idx, tok_end_idx, tok_selected_text_ids_str, input_ids, example_id
        )
        data.update({
            'tok_start_idx': soft_start_labels,
            'tok_end_idx': soft_end_labels
        })

    return data


class TweetData(Dataset):

    def __init__(self, tokenizer, example_ids, texts, sentiments, selected_texts, extended_selected_texts,
                 model_type, max_length=128, evaluate=False, alpha=1., use_jaccard_soft=False):
        super(TweetData, self).__init__()
        self.tokenizer = tokenizer
        self.example_ids = example_ids
        self.texts = texts
        self.sentiments = sentiments
        self.selected_texts = selected_texts
        self.extended_selected_texts = extended_selected_texts
        self.model_type = model_type
        self.max_length = max_length
        self.evaluate = evaluate
        self.use_jaccard_soft = use_jaccard_soft
        self.alpha = alpha

    def __getitem__(self, item):
        data = process_data(self.tokenizer,
                            self.texts[item],
                            self.selected_texts[item],
                            self.extended_selected_texts[item],
                            self.sentiments[item],
                            self.example_ids[item],
                            self.model_type,
                            self.max_length,
                            self.alpha,
                            self.use_jaccard_soft)
        target_dtype = torch.float if self.use_jaccard_soft else torch.long
        example = {
            'input_ids': torch.tensor(data['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(data['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(data['token_type_ids'], dtype=torch.long),
            'start_positions': torch.tensor(data['tok_start_idx'], dtype=target_dtype),
            'end_positions': torch.tensor(data['tok_end_idx'], dtype=target_dtype)
        }

        if not self.evaluate:
            return example
        else:
            example.update({
                'offsets': torch.tensor(data['text_offsets'], dtype=torch.long),
                'orig_selected_text': data['selected_text'],
                'sentiment': data['sentiment'],
                'orig_text': data['text'],
                'example_id': self.example_ids[item]
            })
        return example

    def __len__(self):
        return len(self.sentiments)


def load_examples(args, tokenizer, evaluate):
    filename = args.predict_file if evaluate else args.train_file
    if evaluate: args.use_jaccard_soft = False
    df = pd.read_csv(os.path.join(args.data_dir, filename))
    texts = df.text.tolist()
    selected_texts = df.selected_text.tolist()
    extended_selected_texts = df.extended_selected_text.tolist()
    sentiments = df.sentiment.tolist()
    ids = df.textID.tolist()
    dataset = TweetData(tokenizer, ids, texts, sentiments, selected_texts, extended_selected_texts,
                        args.model_type, args.max_seq_length, evaluate, args.alpha, args.use_jaccard_soft)
    return dataset


def get_jaccard_and_pred_ans(start_idx, end_idx, offsets, orig_text, orig_selected_text, sentiment):
    pred_selected_text = ''

    if sentiment == 'neutral' or len(orig_text.split()) < 2 or start_idx > end_idx:
        pred_selected_text = orig_text

    elif offsets is not None:
        for idx in range(start_idx, end_idx + 1):
            token = orig_text[offsets[idx][0]: offsets[idx][1]]
            pred_selected_text += token
            if idx + 1 < len(offsets) and offsets[idx + 1][0] > offsets[idx][1]:
                pred_selected_text += ' '
    else:
        pred_selected_text = orig_text[start_idx: end_idx + 1]

    jaccard_score = compute_jaccard(orig_selected_text, pred_selected_text)

    return pred_selected_text, jaccard_score


######################### 2nd level model #########################
class CharDataset(Dataset):

    def __init__(self, tokenizer, example_ids, texts, sentiments, selected_texts, extended_selected_texts,
                 start_end_probs, offsets, max_length=128, evaluate=False, alpha=1., use_jaccard_soft=False):
        self.tokenizer = tokenizer
        self.example_ids = example_ids
        self.texts = texts
        self.char_ids = pad_sequences(self.tokenizer.texts_to_sequences(texts),
                                      maxlen=max_length, padding='post', truncating='post')
        self.sentiments = sentiments
        self.sentiment2id = {'neutral': 0, 'positive': 1, 'negative': 2}
        self.selected_texts = selected_texts
        self.extended_selected_texts = extended_selected_texts
        self.start_end_probs = start_end_probs
        self.offsets = offsets
        self.evaluate = evaluate
        self.use_jaccard_soft = use_jaccard_soft
        self.alpha = alpha

    def __len__(self):
        return len(self.sentiments)

    def __getitem__(self, item):
        example_id = self.example_ids[item]
        text = self.texts[item]
        offsets = self.offsets[example_id]
        selected_text = self.selected_texts[item]
        start_end_probs = self.start_end_probs[example_id]
        char_ids = self.char_ids[item]
        char_start_probs = np.zeros(len(char_ids))
        char_end_probs = np.zeros(len(char_ids))
        for i, (offset1, offset2) in enumerate(offsets):
            if offset1 or offset2:  # 全零，不赋予概率
                char_start_probs[offset1: offset2] = start_end_probs[i, 0]
                char_end_probs[offset1: offset2] = start_end_probs[i, 1]

        sentiment_id = self.sentiment2id[self.sentiments[item]]

        start_position = end_position = None
        for i, ch in enumerate(text):
            if ch == selected_text[0] and text[i: i + len(selected_text)] == selected_text:
                start_position = i
                end_position = i + len(selected_text) - 1
                break
        assert selected_text == text[start_position: end_position + 1], \
            f'"{text[start_position: end_position + 1]}" instead of "{selected_text}" in "{text}"'

        data = {
            'start_probs': torch.tensor(char_start_probs, dtype=torch.float),
            'end_probs': torch.tensor(char_end_probs, dtype=torch.float),
            'char_ids': torch.tensor(char_ids, dtype=torch.long),
            'sentiment_ids': torch.tensor(sentiment_id, dtype=torch.long),
        }

        if self.evaluate:
            data.update({
                'sentiment': self.sentiments[item],
                'orig_selected_text': selected_text,
                'orig_text': text,
                'example_id': example_id
            })
        else:
            data.update({
                'start_positions': torch.tensor(start_position, dtype=torch.long),
                'end_positions': torch.tensor(end_position, dtype=torch.long)
            })
        return data


def first_level_inference(model_path, data_type):
    args = torch.load(os.path.join(model_path, 'training_args.bin'))
    # args.train_file = 'debug_train.csv'
    # args.predict_file = 'debug_valid.csv'
    if data_type == 'train': args.predict_file = args.train_file
    tokenizer = TTokenizer.from_file(os.path.join(model_path, 'bpe.tokenizer.json'))
    model = QuestionAnswering()(args.model_type).from_pretrained(model_path).to(args.device)
    dataset = load_examples(args, tokenizer, evaluate=True)
    batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=12)
    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info(f"***** Running inference on {data_type} data *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", batch_size)

    start_end_probs = {}
    offsets = {}
    all_jaccards = []
    start_time = timeit.default_timer()

    for batch in tqdm(dataloader, desc="inference"):
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
            start_probs = torch.softmax(start_logits, dim=-1)
            end_probs = torch.softmax(end_logits, dim=-1)
            start_idxs = start_probs.argmax(dim=-1)
            end_idxs = end_probs.argmax(dim=-1)
            probs = torch.stack([start_probs, end_probs], dim=-1).cpu().numpy()
            for i in range(len(batch['orig_text'])):
                orig_text = batch['orig_text'][i]
                orig_selected_text = batch['orig_selected_text'][i]
                sentiment = batch['sentiment'][i]
                example_id = batch['example_id'][i]
                offset = batch['offsets'][i]
                _, jaccard_score = get_jaccard_and_pred_ans(start_idxs[i], end_idxs[i],
                                                            offset, orig_text,
                                                            orig_selected_text,
                                                            sentiment)
                start_end_probs.update({example_id: probs[i]})
                offsets.update({example_id: offset})
                all_jaccards.append(jaccard_score)

    avg_jaccard_score = sum(all_jaccards) / len(all_jaccards) * 100

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))
    logger.info(f"Result: {avg_jaccard_score:.5f}")

    with open(os.path.join(model_path, f'{data_type}_start_end_probs.pickle'), 'wb') as f:
        pickle.dump(start_end_probs, f)

    with open(os.path.join(model_path, f'{data_type}_offsets.pickle'), 'wb') as f:
        pickle.dump(offsets, f)


def load_char_level_examples(args, tokenizer, evaluate):
    filename = args.predict_file if evaluate else args.train_file
    df = pd.read_csv(os.path.join(args.data_dir, filename))
    example_ids = df.textID.tolist()
    texts = df.text.tolist()
    sentiments = df.sentiment.tolist()
    selected_texts = df.selected_text.tolist()
    extended_selected_texts = df.extended_selected_text.tolist()
    data_type = 'train' if not evaluate else 'valid'
    with open(os.path.join(args.first_level_model, f'{data_type}_start_end_probs.pickle'), 'rb') as f:
        start_end_probs = pickle.load(f)
    with open(os.path.join(args.first_level_model, f'{data_type}_offsets.pickle'), 'rb') as f:
        offsets = pickle.load(f)
    dataset = CharDataset(
        tokenizer, example_ids, texts, sentiments, selected_texts, extended_selected_texts,
        start_end_probs, offsets, args.max_seq_length, evaluate, args.alpha, args.use_jaccard_soft
    )
    return dataset


if __name__ == '__main__':
    from tokenizers import ByteLevelBPETokenizer, BertWordPieceTokenizer
    import pandas as pd
    import argparse

    tokenizer = ByteLevelBPETokenizer(vocab_file='../models/roberta-base/vocab.json',
                                      merges_file='../models/roberta-base/merges.txt',
                                      add_prefix_space=True,
                                      lowercase=True)
    # tokenizer = BertWordPieceTokenizer(
    #     vocab_file=os.path.join('../models/bert-base-cased', 'vocab.txt'),
    #     lowercase=False
    # )
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default='debug_train.csv', type=str)
    parser.add_argument('--model_type', default='roberta', type=str)
    parser.add_argument('--predict_file', default='debug_valid.csv', type=str)
    parser.add_argument('--data_dir', default='../data/', type=str)
    parser.add_argument('--max_seq_length', default=180, type=int)
    parser.add_argument(
        "--alpha",
        default=0.3,
        type=float,
        help="used in jaccard-based soft labels"
    )
    parser.add_argument("--use_jaccard_soft", action="store_false", help="whether to use jaccard-based soft labels.")
    args = parser.parse_args()
    args.first_level_model = './first_level_models/roberta-base-last2h-conv/'
    tokenizer = Tokenizer(num_words=None, char_level=True, oov_token='UNK', lower=True)
    train_texts = pd.read_csv(os.path.join(args.data_dir, args.train_file)).text.tolist()
    tokenizer.fit_on_texts(train_texts)
    args.vocab_size = len(tokenizer.word_index) + 1

    train_dataset = load_char_level_examples(args, tokenizer, False)
    #
    # dataset = load_examples(args, tokenizer)
    # print(dataset[4])
    # train_df = pd.read_csv('../data/clean_train.csv')
    # max_length = 0
    # for text in train_df.text.tolist():
    #     max_length = max(len(tokenizer.encode(text).ids), max_length)
    # print(max_length + 5)
    # max length 105
    # train_dataset = load_examples(args, tokenizer, evaluate=False)
    from torch.utils.data import DataLoader

    dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
    for batch in dataloader:
        print('hah')
    # train_df = pd.read_csv('../data/debug_train.csv')
    # text = train_df.text.tolist()
    # tokenizer = KTokenizer(num_words=None, char_level=True, oov_token='UNK', lower=True)
    # tokenizer.fit_on_texts(text)
    # ids = tokenizer.texts_to_sequences(['you are so beautiful'])
    # print(len(ids[0]))
    # print(len('you are so beautiful'))
    # print(ids)




