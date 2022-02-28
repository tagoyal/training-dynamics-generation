import os, csv, torch, copy
import logging
from torch.utils.data import TensorDataset
from transformers import BartForConditionalGeneration, BartConfig
from transformers import PegasusForConditionalGeneration, PegasusConfig
from transformers.modeling_outputs import Seq2SeqLMOutput
from torch import nn
import numpy as np
from loss_dropper import LossDropper

logger = logging.getLogger(__name__)


def _read_tsv(input_file, quoting=csv.QUOTE_MINIMAL):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=quoting)
        lines = []
        for line in reader:
            lines.append(line)
        return lines


def get_train_examples(data_dir):
    """See base class."""
    return _read_tsv(os.path.join(data_dir, "train.tsv"))


def get_dev_examples(data_dir):
    """See base class."""
    return _read_tsv(os.path.join(data_dir, "dev.tsv"))


class InputFeatures(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def shift_tokens_left(input_ids, pad_token_id):
    """Shift input ids one token to the left"""
    prev_output_tokens = input_ids.clone()
    prev_output_tokens[:, :-1] = input_ids[:, 1:]
    prev_output_tokens[:, -1] = pad_token_id
    return prev_output_tokens


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """Shift input ids one token to the right."""
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class LossDropperToken(nn.Module):
    def __init__(
            self,
            percentile=0.4,
            min_count=5000,
            recompute=5000,
            verbose=True,
            mode='fact',
    ):
        super().__init__()
        self.keepc = 1. - percentile
        self.mode = mode
        self.count = 0
        self.min_count = min_count

        self.recompute = recompute
        self.last_computed = 0
        self.percentile_val = 100000000.
        self.cur_idx = 0

        self.verbose = verbose

        self.vals = np.zeros(self.recompute, dtype=np.float32)

    def forward(self, loss):
        if loss is None:
            return loss

        non_zero_loss_elems = loss[loss.nonzero(as_tuple=True)]
        self.last_computed += non_zero_loss_elems.numel()
        self.count += non_zero_loss_elems.numel()
        if self.count < len(self.vals):
            self.vals[
            self.count - non_zero_loss_elems.numel():self.count] = non_zero_loss_elems.detach().cpu().numpy().flatten()
            self.cur_idx += non_zero_loss_elems.numel()
            return (loss < np.inf).type(loss.dtype)
        else:
            for idx, item in enumerate(non_zero_loss_elems):
                self.vals[self.cur_idx] = item
                self.cur_idx += 1
                if self.cur_idx >= len(self.vals):
                    self.cur_idx = 0
        if self.count < self.min_count:
            return (loss < np.inf).type(loss.dtype)

        if self.last_computed > self.recompute:
            self.percentile_val = np.percentile(self.vals, self.keepc * 100)
            if self.verbose:
                print('Using cutoff', self.percentile_val)
            self.last_computed = 0

        if self.mode == 'fact':
            mask = (loss < self.percentile_val).type(loss.dtype)
        else:
            mask = (loss > self.percentile_val).type(loss.dtype)
        return mask


class ConditionalGenerationCustom(nn.Module):
    def add_dropper_properties(self, percentile=0.2, mode='fact'):
        self.dropper = LossDropperToken(percentile=percentile,
                                        mode=mode)  # 0.5 for factuality experiments, 0.2 for abstractiveness
        self.summarydroper = LossDropper(dropc=percentile)  # 0.5 for factuality experiments

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            generate=True,
            token_level_loss_truncation=False,
            summary_level_loss_truncation=False,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = True if generate else False

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            # cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if not generate:
            if token_level_loss_truncation:  # token level loss trunc slightly modifying kang and hashimoto
                lm_labels = shift_tokens_left(decoder_input_ids, self.config.pad_token_id)
                loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id, reduction='none')
                masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), lm_labels.view(-1))
                mask = self.dropper(masked_lm_loss)
                masked_lm_loss *= mask
                masked_lm_loss = masked_lm_loss.mean()

            elif summary_level_loss_truncation:
                batch_size = input_ids.shape[0]
                lm_labels = shift_tokens_left(decoder_input_ids, self.config.pad_token_id)
                loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id, reduction='none')
                masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), lm_labels.view(-1))
                masked_lm_loss = masked_lm_loss.view(batch_size, -1)
                masked_lm_loss = masked_lm_loss.mean(dim=1)

                mask = self.dropper(masked_lm_loss)
                masked_lm_loss *= mask
                masked_lm_loss = masked_lm_loss.mean()

            else:
                lm_labels = shift_tokens_left(decoder_input_ids, self.config.pad_token_id)
                loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
                masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), lm_labels.view(-1))

        if generate:
            return Seq2SeqLMOutput(
                loss=masked_lm_loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )
        else:
            return Seq2SeqLMOutput(
                loss=masked_lm_loss,
                logits=lm_logits)


class ConditionalGenerationCustomBart(ConditionalGenerationCustom, BartForConditionalGeneration):
    def __init__(self, config: BartConfig):
        super().__init__(config)


class ConditionalGenerationCustomPegasus(ConditionalGenerationCustom, PegasusForConditionalGeneration):
    def __init__(self, config: PegasusConfig):
        super().__init__(config)


def convert_examples_to_features(examples, tokenizer, max_length=1024, max_decoder_length=128):
    features = []
    pad_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % ex_index)

        input = example['article']
        output = example['summary']

        id = tokenizer.encode(example['id'], max_length=40)
        id = id + ([pad_id] * (40 - len(id)))

        if input == '' or output == '':
            continue

        input_ids = tokenizer.encode(input, add_prefix_space=True)
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length - 1]
        padding_length_a = max_length - len(input_ids)
        input_attention_mask = [1] * len(input_ids) + ([0] * padding_length_a)
        input_ids = input_ids + ([pad_id] * padding_length_a)

        decoder_ids = tokenizer.encode(output, add_prefix_space=True)
        if len(decoder_ids) > max_decoder_length:
            decoder_ids = decoder_ids[:max_decoder_length - 1]
        padding_length_b = max_decoder_length - len(decoder_ids)
        decoder_attention_mask = [1] * len(decoder_ids) + ([0] * padding_length_b)
        decoder_ids = decoder_ids + ([pad_id] * padding_length_b)

        features.append(InputFeatures(input_ids=input_ids,
                                      attention=input_attention_mask,
                                      decoder_attention=decoder_attention_mask,
                                      decoder_ids=decoder_ids,
                                      id=id))
    print(len(features))
    return features


def load_and_cache_examples(args, tokenizer, evaluate):
    if evaluate:
        data_dir = '/'.join(args.eval_data_file.split('/')[:-1])
        file_name = args.eval_data_file.split('/')[-1].split('.')[0]
    else:
        data_dir = '/'.join(args.train_data_file.split('/')[:-1])
        file_name = args.train_data_file.split('/')[-1].split('.')[0]

    model_type = args.model_type
    cached_features_file = os.path.join(
        data_dir,
        "cached_{}_{}_{}".format(
            file_name,
            model_type,
            str(args.max_seq_length)
        ),
    )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", data_dir)
        examples = (get_dev_examples(data_dir) if evaluate else get_train_examples(data_dir))
        features = convert_examples_to_features(
            examples,
            tokenizer,
            max_length=args.max_seq_length,
            max_decoder_length=args.max_decoder_length,
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_attention_mask = torch.tensor([f.attention for f in features], dtype=torch.long)
    decoder_ids = torch.tensor([f.decoder_ids for f in features], dtype=torch.long)
    decoder_attention_mask = torch.tensor([f.decoder_attention for f in features], dtype=torch.long)
    ids = torch.tensor([f.id for f in features])

    dataset = TensorDataset(input_ids, input_attention_mask, decoder_ids, decoder_attention_mask, ids)

    return dataset
