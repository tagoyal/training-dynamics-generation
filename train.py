import argparse
import json
import logging
import os
import random
from typing import Dict, List, Tuple
import numpy as np
import torch
import math
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm, trange
import train_utils
from torch.nn.utils.rnn import pad_sequence

from torch import nn

from transformers import (
    AdamW,
    PreTrainedModel,
    PreTrainedTokenizer,
    BartConfig,
    BartTokenizer,
    PegasusConfig,
    PegasusTokenizer,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)
MODEL_CLASSES = {"bart": (BartConfig, train_utils.ConditionalGenerationCustomBart, BartTokenizer),
                 "pegasus": (PegasusConfig, train_utils.ConditionalGenerationCustomPegasus, PegasusTokenizer), }


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def save_checkpoints(args, output_dir, model, tokenizer, suffix=None):
    if suffix is not None:
        output_dir = os.path.join(output_dir, f'ckp_{suffix}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)


def evaluate(args, eval_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, suffix="") -> Dict:
    eval_output_dir = os.path.join(args.output_dir, 'outputs')

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    print(eval_dataset)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    softmax_function = nn.Softmax(dim=-1)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss_sentence = 0.0
    eval_steps = 0

    if args.dump_posteriors:
        f_out = open(os.path.join(eval_output_dir, 'prob_out%s.txt' % suffix), 'w')

    with torch.no_grad():
        model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(args.device) for t in batch)

            input_ids, attention, decoder_ids, decoder_attention = batch[0], batch[1], batch[2], batch[3]

            inputs = {'input_ids': input_ids, 'attention_mask': attention, 'decoder_input_ids': decoder_ids,
                      'decoder_attention_mask': decoder_attention, 'generate': False}

            outputs = model(**inputs)
            tmp_eval_loss_sentence = outputs.loss
            eval_loss_sentence += tmp_eval_loss_sentence.item()
            eval_steps += 1

            if args.dump_posteriors:
                logits = outputs[1]
                softmax_scores = softmax_function(logits)

                decode_ids_shifted_left = train_utils.shift_tokens_left(decoder_ids, tokenizer.pad_token_id)
                decoder_softmax_scores = torch.gather(softmax_scores, dim=2,
                                                      index=decode_ids_shifted_left.unsqueeze(2)).detach().cpu().numpy()
                decoder_softmax_scores = decoder_softmax_scores.squeeze(2)

                for j in range(decoder_softmax_scores.shape[0]):
                    uncleaned_tokens = tokenizer.convert_ids_to_tokens(decode_ids_shifted_left[j],
                                                                       skip_special_tokens=False)
                    input = tokenizer.decode(input_ids[j]).replace('<pad>', '')
                    output = tokenizer.decode(decoder_ids[j]).replace('<pad>', '')
                    f_out.write(input + '\n')
                    f_out.write(output + '\n')

                    for k in range(len(uncleaned_tokens)):
                        f_out.write(uncleaned_tokens[k] + '\t' + str(decoder_softmax_scores[j][k]) + '\n')
                        if uncleaned_tokens[k] == '</s>':
                            break
                    f_out.write('\n\n')
            """
            if eval_steps == 50:
                break"""

    eval_loss_sentence = eval_loss_sentence / eval_steps
    result = {'loss': eval_loss_sentence}
    print(result)

    if args.generate:
        f_out = open(os.path.join(eval_output_dir, 'dev_out%s.txt' % suffix), 'w')
        f_out_posterior = open(os.path.join(eval_output_dir, 'dev_out_prob%s.txt' % suffix), 'w')
        print(eval_output_dir)
        k = 0

        with torch.no_grad():
            model.eval()
            for batch in eval_dataloader:
                batch = tuple(t.to(args.device) for t in batch)
                input_ids, input_attention_mask, decoder_ids = batch[0], batch[1], batch[2]
                ids = batch[4]

                if args.dataset == 'xsum':
                    output_ids = model.generate(input_ids,
                                                attention_mask=input_attention_mask,
                                                num_beams=6, length_penalty=1, no_repeat_ngram_size=3, max_length=200,
                                                min_length=12, num_return_sequences=1,
                                                decoder_start_token_id=tokenizer.bos_token_id)
                elif args.dataset == 'cnndm':
                    if args.model_type == 'pegasus':
                        output_ids = model.generate(input_ids,
                                                    attention_mask=input_attention_mask,
                                                    num_beams=4, length_penalty=0.8, no_repeat_ngram_size=3,
                                                    max_length=128,
                                                    min_length=20, num_return_sequences=1,
                                                    decoder_start_token_id=tokenizer.bos_token_id)
                    else:
                        output_ids = model.generate(input_ids,
                                                    attention_mask=input_attention_mask,
                                                    num_beams=4, length_penalty=2, no_repeat_ngram_size=3,
                                                    max_length=200,
                                                    min_length=20, num_return_sequences=1,
                                                    decoder_start_token_id=tokenizer.bos_token_id)

                else:
                    output_ids = model.generate(input_ids,
                                                attention_mask=input_attention_mask,
                                                num_beams=4, length_penalty=2, no_repeat_ngram_size=3, max_length=200,
                                                min_length=10, num_return_sequences=1,
                                                decoder_start_token_id=tokenizer.bos_token_id)

                for j in range(len(input_ids)):
                    input = tokenizer.decode(input_ids[j], skip_special_tokens=True)
                    gold = tokenizer.decode(decoder_ids[j], skip_special_tokens=True)

                    f_out.write(tokenizer.decode(ids[j]) + '\n')
                    f_out.write(input.strip() + '\n')
                    f_out.write(gold.strip() + '\n')

                    gen = tokenizer.decode(output_ids[j], skip_special_tokens=True, clean_up_tokenization_spaces=False)

                    f_out.write(gen.strip() + '\n\n')
                    print(gen)

                # TODO: clean this up later
                # output_ids = pad_sequence(output_ids, batch_first=True)
                output_ids = output_ids.squeeze(1)
                inputs = {'input_ids': input_ids, 'attention_mask': input_attention_mask,
                          'decoder_input_ids': output_ids, 'generate': False}

                outputs = model(**inputs)

                logits = outputs[1]
                softmax_scores = softmax_function(logits)

                decode_ids_shifted_left = train_utils.shift_tokens_left(output_ids, tokenizer.pad_token_id)
                decoder_softmax_scores = torch.gather(softmax_scores, dim=2,
                                                      index=decode_ids_shifted_left.unsqueeze(
                                                          2)).detach().cpu().numpy()
                decoder_softmax_scores = decoder_softmax_scores.squeeze(2)

                for j in range(decoder_softmax_scores.shape[0]):
                    uncleaned_tokens = tokenizer.convert_ids_to_tokens(decode_ids_shifted_left[j],
                                                                       skip_special_tokens=False)
                    input = tokenizer.decode(input_ids[j]).replace('<pad>', '')
                    output = tokenizer.decode(output_ids[j]).replace('<pad>', '')
                    f_out_posterior.write(input + '\n')
                    f_out_posterior.write(output + '\n')

                    for idx in range(len(uncleaned_tokens)):
                        f_out_posterior.write(uncleaned_tokens[idx] + '\t' + str(decoder_softmax_scores[j][idx]) + '\n')
                        if uncleaned_tokens[idx] == '</s>':
                            break

                    f_out_posterior.write('\n\n')

                k += 1
                if args.per_gpu_eval_batch_size * k >= 800:
                    break

            f_out.close()

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("%s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
        writer.write('\n')
    return result


def train(args, train_dataset, eval_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
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

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps,
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0

    tr_loss = 0.0
    logging_loss_sent = 0.0

    model.zero_grad()
    train_iterator = trange(0, int(args.num_train_epochs), desc="Epoch")
    set_seed(args)

    torch.cuda.empty_cache()

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):

            batch = tuple(t.to(args.device) for t in batch)
            input_ids, attention, decoder_ids, decoder_attention = batch[0], batch[1], batch[2], batch[3]
            inputs = {'input_ids': input_ids, 'attention_mask': attention, 'decoder_input_ids': decoder_ids,
                      'decoder_attention_mask': decoder_attention, 'generate': False}

            if args.token_level_loss_truncation and global_step > args.loss_based_comp_or_trunc_steps:
                inputs['token_level_loss_truncation'] = True
            elif args.summary_level_loss_truncation and global_step > args.loss_based_comp_or_trunc_steps:
                inputs['summary_level_loss_truncation'] = True

            outputs = model(**inputs)
            loss = outputs.loss

            tr_loss += loss.item()

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    logs = {}
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    loss_scalar_sent = (tr_loss - logging_loss_sent) / args.save_steps
                    logs["loss_sent"] = loss_scalar_sent
                    logging_loss_sent = tr_loss

                    print(json.dumps({**logs, **{"step": global_step}}))
                    logger.info(json.dumps({**logs, **{"step": global_step}}))

                    # Evaluation
                    evaluate(args, eval_dataset, model, tokenizer, str(global_step))
                    save_checkpoints(args, args.output_dir, model, tokenizer, str(global_step))

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break

        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type choose from bart or pegasus (others may be available)",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Check path to pre-trained model or shortcut name",
    )
    parser.add_argument(
        "--input_dir",
        default=None,
        type=str,
        help="Check path to pre-trained model or shortcut name",
    )
    parser.add_argument(
        "--input_dir_weak_model",
        default=None,
        type=str,
        help="The input folder for the weak model. Only required when using example reweighting"
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        required=True,
        help="Evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--train_data_file",
        default=None,
        type=str,
        required=True,
        help="The input training data file (a text file)."
    )

    # Other parameters
    parser.add_argument(
        "--max_seq_length",
        default=1024,
        type=int,
        help="The maximum total input sequence length after tokenization.",
    )
    parser.add_argument(
        "--max_decoder_length",
        default=128,
        type=int,
        help="The maximum total decoder sequence length after tokenization.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int, help="Batch size training.", )
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int, help="Batch size evaluation.", )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs", )
    parser.add_argument("--gpu_device", type=int, default=0, help="gpu device")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the output directory", )
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached data sets", )
    parser.add_argument("--generate", action="store_true", help="Generate summaries for dev set", )
    parser.add_argument("--dump_posteriors", action="store_true", help="Dump posterior probs at intermediate steps", )
    parser.add_argument("--seed", type=int, default=100, help="random seed for initialization")

    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: total no. of train steps. Overrides "
                                                                  "num_train_epochs.", )
    parser.add_argument("--save_steps", type=int, default=0, help="Save checkpoint every X updates steps.")
    parser.add_argument("--dataset", default='xsum', help="dataset")

    parser.add_argument("--token_level_loss_truncation", action="store_true", help="Token-level LT")
    parser.add_argument("--summary_level_loss_truncation", action="store_true", help="Summary-level LT")
    parser.add_argument("--percentile_loss_truncation", type=float, default=0.5)
    parser.add_argument("--loss_based_comp_or_trunc_steps", type=int, default=3000,
                        help="Mod. train data after X steps for loss trunc.")
    parser.add_argument("--truncation_mode", default='fact')

    args = parser.parse_args()

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.n_gpu = 1
    device = torch.device("cuda", args.gpu_device)
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        filename=os.path.join(args.output_dir, 'model.log')
    )

    # Set seed
    set_seed(args)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    logger.info("Training/evaluation parameters %s", args)

    if args.do_eval:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
        eval_dataset = train_utils.load_and_cache_examples(args, tokenizer, evaluate=True)
        for i in range(3000, 3001, 1000):
            if i == 0:
                config = config_class.from_pretrained(args.model_name_or_path)
                model = model_class.from_pretrained(args.model_name_or_path, config=config)
            else:
                model_dir = os.path.join(args.input_dir, f'ckp_{i}')
                model = model_class.from_pretrained(model_dir)
            model.to(args.device)
            evaluate(args, eval_dataset, model, tokenizer, str(i))
        exit()

    if args.do_train:
        if args.input_dir is not None:
            print('loading model')
            tokenizer = tokenizer_class.from_pretrained(args.input_dir)
            model = model_class.from_pretrained(args.input_dir)
        else:
            config = config_class.from_pretrained(args.model_name_or_path)
            tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
            model = model_class.from_pretrained(args.model_name_or_path, config=config)

        model.add_dropper_properties(percentile=args.percentile_loss_truncation, mode=args.truncation_mode)
        model.to(args.device)

        eval_dataset = train_utils.load_and_cache_examples(args, tokenizer, evaluate=True)
        evaluate(args, eval_dataset, model, tokenizer, '0')

        train_dataset = train_utils.load_and_cache_examples(args, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, eval_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


if __name__ == "__main__":
    main()
