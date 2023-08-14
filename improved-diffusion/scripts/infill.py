"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import copy
import os, json, sys
import stanza
import spacy_stanza
import numpy as np
import torch as th
import wandb

from transformers import set_seed, AutoConfig
import torch.distributed as dist
from improved_diffusion.rounding import rounding_func, load_models, load_tokenizer
from improved_diffusion.test_util import get_weights, denoised_fn_round
from functools import partial
from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

sys.path.insert(0, 'diffusion_lm/transformers/examples/pytorch/language-modeling')
from custom_trainer import Classifier_GPT2, Classifier_Times, Classifier_POS, Classifier_Tree, \
    Classifier_TONE, Classifier_VOWEL, Classifier_LINE, Classifier_RHYME
from infill_util import langevin_fn3, get_score, langevin_fn3_compose, langevin_fn1, langevin_fn4, langevin_fn_tree, \
    langevin_fn_length, langevin_fn_tone_vowel_length, langevin_fn_tone_length, langevin_fn_tone, langevin_fn_line, \
    langevin_fn_line_rhyme
from spacy.lang.en import English

name2val = {}
mid_steps = []


def dumpkvs(kv_dict):
    for key in kv_dict:
        name2val[key] = kv_dict[key]
    # wandb.log({**name2val})


def main():
    set_seed(101)
    args = create_argparser().parse_args()

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    args.__dict__.update(training_args)

    args.noise_level = 0.0
    args.sigma_small = True

    if args.eval_task_.startswith('control_'):
        args.diffusion_steps = args.change_num_steps  # 200  # 500  # DEBUG
    dist_util.setup_dist()
    logger.configure()
    print(args.clip_denoised, 'clip_denoised')
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(th.load(args.model_path))
    # print("model:", model)
    model.to(dist_util.dev())
    model.eval()

    logger.log("load embedding models")
    print(os.path.split(args.model_path)[0])
    model_embs, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                        os.path.split(args.model_path)[0])
    if args.training_mode.startswith('e2e'):
        print('e2e, load the right model embeddings', '*' * 80)  # √
        model_embs.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())
    model_embs = model_embs.cuda()
    model3 = get_weights(model_embs, args)
    logger.log('load the partial sequences')
    if args.partial_seq:
        logger.info(f"args.partial_seq: {args.partial_seq}")
        partial_seq = [args.partial_seq]
        partial_seq_idx = ['0']
    elif args.partial_seq_file:
        logger.info(f"args.partial_seq_file: {args.partial_seq_file}")
        # implies that we should read from the files
        nlp = English()
        tokenizer_spacy = nlp.tokenizer
        print(f'reading from the file {args.partial_seq_file}', '-*' * 20)
        with open(args.partial_seq_file, 'r') as f:
            sent_lst = json.load(f)
        partial_seq = []
        partial_seq_idx = []
        for idx, (key, val) in enumerate(sent_lst.items()):
            if idx < int(args.start_idx) or idx > int(args.end_idx):
                continue
            partial_seq_ = f"{val['obs1']} " + "PAD " * 10 + f"{val['obs2']}"
            word_lst = [x.text for x in tokenizer_spacy(partial_seq_)]
            partial_seq_ = " ".join(word_lst)
            print(partial_seq_, idx)
            partial_seq.append(partial_seq_)
            partial_seq_idx.append(str(idx))
    else:
        logger.info("else")  # √
        partial_seq = ['A kid friendly venue named Alimentum is located on the riverside .',
                       'Alimentum , situated by the river , is quite child friendly .']
        partial_seq_idx = ['0', '1']
    # else:  generate them by randomly preturbing the inputs data.
    logger.info(f"args.modality: {args.modality}")  # e2e-tgt
    if args.modality in ['synth', 'pos']:
        tokens2id = {v: k for k, v in tokenizer.items()}
        todo_pad_token = tokens2id['END']
        print(f'pad token = {todo_pad_token}')
        encoded_partial_seq = [th.LongTensor([tokens2id[x] for x in seq.split()]) for seq in partial_seq]
        print(encoded_partial_seq[0], len(encoded_partial_seq[0]))
    elif args.modality in ['e2e-tgt', 'roc', 'roc-aug']:  # √
        tokens2id = {v: k for k, v in tokenizer.items()}
        todo_pad_token = -1
        pad_token = tokens2id['PAD']
        encoded_partial_seq = [th.LongTensor([tokens2id.get(x, tokens2id['UNK']) for x in seq.split()]) for seq in
                               partial_seq]
        logger.info(f"args.eval_task_: {args.eval_task_}")  # control_tree
        if args.eval_task_ == 'infill':
            todo_pad_token = tokens2id['PAD']
            print(f'pad token = {todo_pad_token}')
            partial_seq = [(b, a) for (a, b) in zip(partial_seq, partial_seq_idx)]
            pass
        elif args.eval_task_ == 'l2r':
            # right_length= args.image_size ** 2 - len(encoded_partial_seq[0])
            right_length = args.tgt_len - len(encoded_partial_seq[0])
            right_pad = th.empty(right_length).fill_(todo_pad_token).long()
            encoded_partial_seq = [th.cat([seq, right_pad], dim=0) for seq in encoded_partial_seq]

        elif args.eval_task_ == 'r2l':
            # right_length= args.image_size ** 2 - len(encoded_partial_seq[0])
            # right_length= args.image_size ** 2 - len(encoded_partial_seq[0])
            right_length = args.tgt_len - len(encoded_partial_seq[0])
            right_pad = th.empty(right_length).fill_(todo_pad_token).long()
            encoded_partial_seq = [th.cat([right_pad, seq], dim=0) for seq in encoded_partial_seq]

        elif args.eval_task_ == 'length':
            right_length = args.image_size ** 2 - len(encoded_partial_seq[0])
            # right_length = args.tgt_len - len(encoded_partial_seq[0])
            # assert args.tgt_len > len(encoded_partial_seq[0])
            right_pad = th.empty(right_length).fill_(todo_pad_token).long()
            encoded_partial_seq = [th.cat([seq, right_pad], dim=0) for seq in encoded_partial_seq]
            encoded_partial_seq[0][args.tgt_len - 1] = tokens2id['END']
            encoded_partial_seq[0][args.tgt_len] = tokens2id['START']
            # encoded_partial_seq[0][args.tgt_len+1:] = tokens2id['PAD']

        elif args.eval_task_ == 'word':
            right_length = args.tgt_len // 2

            right_pad = th.empty(right_length).fill_(todo_pad_token).long()
            encoded_partial_seq = [th.cat([right_pad, seq, right_pad], dim=0) for seq in encoded_partial_seq]
        elif args.eval_task_.startswith('control'):
            # right_pad = th.empty(args.tgt_len+2).fill_(pad_token).long()
            # TO FIX... IMPORTANT.
            if 'length' not in args.eval_task_:
                if 'line' in args.eval_task_:
                    args.image_size = 14
                else:
                    args.image_size = 8
                right_pad = th.empty(args.image_size**2).fill_(pad_token).long()
                encoded_partial_seq = [th.cat([right_pad], dim=0)]
                encoded_partial_seq[0][0] = tokens2id['START']
                encoded_partial_seq[0][args.tgt_len] = tokens2id['END']

            if args.eval_task_ == 'control_attribute':
                # model_control = Classifier_GPT2.from_pretrained('predictability/diff_models/e2e-back_e=6_b=10_m=gpt2_wikitext-103-raw-v1_101_wp_full_multi16_t_aware').cuda()
                model_control = Classifier_GPT2.from_pretrained(
                    os.path.split(args.model_path)[0] +
                    '/../../../classifier_models/'
                    'e2e-tgt-tree_e=6_b=10_m=bert-base-uncased_wikitext-103-raw-v1_101_wp_None').cuda()

                control_label_lst = []
                # with open('diffusion_lm/improved-diffusion/control_gen/target_attribute.json', 'r') as controlf:
                with open(os.path.split(args.model_path)[0] + '/../../control_gen/target_pos.json', 'r') as controlf:
                    for line in controlf:
                        control_label_lst.append(json.loads(line))
                # print(control_label_lst[:5])
                control_constraints = []
                for label_class in control_label_lst:
                    # assert encoded_partial_seq[0].size(0)  == 64
                    label = [-100] * 64 + [tokens2id.get(x, tokens2id['UNK']) for x in label_class]
                    label_ids = th.tensor(label).unsqueeze(0)
                    debug_lst = []
                    langevin_fn_selected = partial(langevin_fn3, debug_lst, model_control, model3.cuda(),
                                                   label_ids.expand(args.batch_size, -1), 0.1)
                    control_constraints.append((langevin_fn_selected, label_class))

                partial_seq = control_constraints
                # print(control_constraints)
                encoded_partial_seq = [encoded_partial_seq[0] for _ in range(len(partial_seq))]
                assert len(partial_seq) == len(encoded_partial_seq)
                print(f'RUNNING FOR {len(partial_seq)} constraints.', '*-' * 20)
                # # label_class =  ['price', ':', 'cheap']
                # # label_class =  ['name', ':', 'The', 'Vaults']
                # # label_class =  ['food', ':', 'French']
                # label_class = ['price', ':', 'none']
                # label_class = ['near', ':', 'riverside']
                # label_class = ['name', ':','The', 'Vaults'] #98%
                # label_class = ['name', ':', 'The', 'Cricketers'] #92%
                # label_class = ['name', ':','Green', 'Man'] #96%
                # label_class = ['price', ':', 'cheap'] #84%
                # label_class = ['price', ':', 'moderate'] #78%
                # label_class = ['area', ':', 'riverside'] #90%
                # label_class = ['UNK', ':', 'coffee', 'shop']#98%
                # label_class = ['UNK', ':', 'pub']  # 90%
                # label_class = ['name', ':', 'The', 'Rice', 'Boat'] # 100%
                # label_class = ['food', ':', 'French'] #82%
                # label_class = ['customer', 'rating', ':', 'average'] #78%
                # label_class = ['customer', 'rating', ':', '3', 'out', 'of', '5']  # 0.54
                # label_class = ['customer', 'rating', ':', '5', 'out', 'of', '5']  # 0.68
                # label_class = ['name', ':', 'Green', 'Man']  # 96% --> 82%
                # # label_class = ['price', ':', 'less', 'than', '£', '20'] # 90%
                # # label_class = ['price', ':', 'cheap']  # 84%
                # label = [-100] * encoded_partial_seq[0].size(0) + [tokens2id[x] for x in label_class]
                # label_ids = th.tensor(label).unsqueeze(0)
                #
                # debug_lst = []
                # langevin_fn_selected = partial(langevin_fn3, debug_lst, model_control, model3.cuda(),
                #                                label_ids.expand(args.batch_size, -1), 0.1)
                # # langevin_fn_selected = partial(langevin_fn1, debug_lst, model_control, model3.cuda(),
                # #                                label_ids.expand(args.batch_size, -1), 0.1)

            if args.eval_task_ == 'control_attribute_compose':
                # model_control = Classifier_GPT2.from_pretrained('predictability/diff_models/e2e-bac'
                #                                                 'k_e=6_b=10_m=gpt2_wikitext-103-raw-v1_101_wp_'
                #                                                 'full_multi16_t_aware').cuda()
                model_control = Classifier_GPT2.from_pretrained(
                    os.path.split(args.model_path)[0] +
                    '/../../../classifier_models/'
                    'e2e-tgt-tree_e=6_b=10_m=bert-base-uncased_wikitext-103-raw-v1_101_wp_None').cuda()
                label_ids_lst = []

                label_class = ['price', ':', 'none']
                label_class = ['near', ':', 'riverside']
                label_class = ['name', ':', 'The', 'Vaults']  # 98%
                label_class = ['name', ':', 'The', 'Cricketers']  # 92%
                label_class = ['name', ':', 'Green', 'Man']  # 96%
                label_class = ['price', ':', 'cheap']  # 84%
                label_class = ['price', ':', 'moderate']  # 78%
                label_class = ['area', ':', 'riverside']  # 90%
                label_class = ['UNK', ':', 'coffee', 'shop']  # 98%
                label_class = ['UNK', ':', 'pub']  # 90%
                label_class = ['name', ':', 'The', 'Rice', 'Boat']  # 100%
                label_class = ['food', ':', 'French']  # 82%
                label_class = ['customer', 'rating', ':', 'average']  # 78%
                label_class = ['customer', 'rating', ':', '3', 'out', 'of', '5']  # 0.54
                # label_class = ['customer', 'rating', ':', '5', 'out', 'of', '5']  # 0.68
                label_class1 = ['name', ':', 'Green', 'Man']  # 96% --> 82%
                # label_class1 = ['price', ':', 'less', 'than', '£', '20'] # 90%
                # label = [-100] * encoded_partial_seq[0].size(0) + [tokens2id[x] for x in label_class1]
                # label_ids = th.tensor(label).unsqueeze(0).cuda()
                # label_ids_lst = [label_ids]

                label_ids_lst = []
                label_class2 = ['name', ':', 'Green', 'Man']  # 96% --> 82%
                label_class1 = ['price', ':', 'less', 'than', '£', '20']  # 90%
                for label_class in [label_class1, label_class2]:
                    label = [-100] * encoded_partial_seq[0].size(0) + [tokens2id[x] for x in label_class]
                    label_ids = th.tensor(label).unsqueeze(0).cuda()
                    label_ids_lst.append(label_ids)

                debug_lst = []
                langevin_fn_selected = partial(langevin_fn3_compose, debug_lst, model_control, model3.cuda(),
                                               [label_ids.expand(args.batch_size, -1) for label_ids in label_ids_lst],
                                               0.1)

            elif args.eval_task_ == 'control_pos':
                # model_control = Classifier_POS.from_pretrained('predictability/diff_models/e2e-tgt-pos_e=6_b=10_m=bert-'
                #                                                'base-uncased_wikitext-103-raw-v1_101_wp_full_multi16_v2').cuda()
                model_control = Classifier_POS.from_pretrained(
                    os.path.split(args.model_path)[0] +
                    '/../../../classifier_models/'
                    'e2e-tgt-pos_e=6_b=10_m=bert-base-uncased_wikitext-103-raw-v1_101_wp_None').cuda()

                pos_vocab = {'START': 0, 'END': 1, 'UNK': 2, 'PAD': 3}

                pos_lst = ['ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB',
                           'ADP', 'AUX', 'CCONJ', 'DET', 'NUM', 'PART', 'PRON', 'SCONJ',
                           'PUNCT', 'SYM', 'X']
                for x in pos_lst:
                    pos_vocab[x] = len(pos_vocab)
                pos_vocab_rev = {v: k for k, v in pos_vocab.items()}

                ################33
                control_label_lst = []
                with open(os.path.split(args.model_path)[0] + '/../../' + 'control_gen/target_pos.json',
                          'r') as controlf:
                    for line in controlf:
                        control_label_lst.append(json.loads(line))
                print(control_label_lst[:5])
                control_constraints = []
                for label_class_dict in control_label_lst[:50]:  # control_label_lst[:100]:
                    label_class = label_class_dict['pos']
                    words_ = label_class_dict['words_']
                    label_class = [pos_vocab.get(x, pos_vocab['UNK']) for x in label_class]
                    label_class = label_class + [pos_vocab['PAD']] * (64 - len(label_class))
                    label_ids = th.LongTensor(label_class).unsqueeze(0)
                    debug_lst = []
                    langevin_fn_selected = partial(langevin_fn4, debug_lst, model_control, model3.cuda(),
                                                   label_ids.expand(args.batch_size, -1),
                                                   0.1)
                    control_constraints.append((langevin_fn_selected, label_class_dict['pos']))

                partial_seq = control_constraints
                # print(control_constraints)
                encoded_partial_seq = [encoded_partial_seq[0] for _ in range(len(partial_seq))]
                assert len(partial_seq) == len(encoded_partial_seq)
                print(f'RUNNING FOR {len(partial_seq)} constraints.', '*-' * 20)

                # toy1 = ['START', 'The', 'Vaults', 'pub', 'near', 'Café', 'Adriatic', 'has', 'a', '5', 'star', 'rating', '.',
                #        'Prices', 'start', 'at', '£', '30', '.', '\n', 'END']
                # # toy1 = 'START The Mill is a coffee shop with an expensive menu near The Sorrento . \n END'.split()
                # toy = toy1 + (64 - len(toy1)) * ['PAD']
                # input_ids = th.tensor([tokens2id[x] for x in toy]).unsqueeze(0)
                #
                # model_out = model_control(input_ids.to(model_control.device), t=200)
                # print(model_out.logits.shape)
                # pred_pos = th.argmax(model_out.logits, dim=-1)
                # print(pred_pos.shape, pred_pos)
                # print('predicted pos', [pos_vocab_rev[x.item()] for x in pred_pos[0]])
                # model_out = model_control(input_ids.to(model_control.device), pos_ids=pred_pos, t=200)
                # print('predicted score', model_out.loss)
                #
                # nlp = spacy_stanza.load_pipeline("en", processors={"tokenize": "spacy"})
                # sent_full = " ".join(toy1[1:-1])
                # doc = nlp(sent_full)
                # doc_token_pos = [(token.text, token.pos_,) for token in doc]
                # print(doc_token_pos)
                # doc_token_pos = ['START'] + [x[1] for x in doc_token_pos] + ['END']
                # print(doc_token_pos, 'target POS tagging sequences')
                # label_class = [pos_vocab.get(x, pos_vocab['UNK']) for x in doc_token_pos]
                # label_class = label_class + [pos_vocab['PAD']] * (encoded_partial_seq[0].size(0)-len(label_class))
                # print(label_class)
                # label_ids = th.LongTensor(label_class).unsqueeze(0)
                # label_ids[:, 3:] = -100
                # label_ids[:, :1] = -100
                #
                # debug_lst = []
                # langevin_fn_selected = partial(langevin_fn4, debug_lst, model_control, model3.cuda(),
                #                                label_ids.expand(args.batch_size, -1),
                #                                0.1)

            elif args.eval_task_ == 'control_tone_vowel_length':
                # model_control = Classifier_POS.from_pretrained('predictability/diff_models/e2e-tgt-pos_e=6_b=10_m=bert-'
                #                                                'base-uncased_wikitext-103-raw-v1_101_wp_full_multi16_v2').cuda()
                model_control_tone = Classifier_TONE.from_pretrained(
                    os.path.split(args.model_path)[0] +
                    '/../../../classifier_models/' + args.classifier_model_name).cuda()
                model_control_vowel = Classifier_VOWEL.from_pretrained(
                    os.path.split(args.model_path)[0] +
                    '/../../../classifier_models/' + args.classifier_model_name_2).cuda()

                tone_vocab = {'START': 0, 'END': 1, 'UNK': 2, 'PAD': 3, 'NONE': 4, 'PING': 5, 'ZE': 6}
                vowel_vocab = {'START': 0, 'END': 1, 'UNK': 2, 'PAD': 3}
                vowel_lst = [str(x) for x in range(18)]
                for x in vowel_lst:
                    vowel_vocab[x] = len(vowel_vocab)

                control_label_lst = []
                with open(os.path.split(args.model_path)[0] + "/../../" + "control_gen/" + args.tgt_file,
                          'r') as controlf:
                    for line in controlf:
                        control_label_lst.append(json.loads(line))
                # print(control_label_lst[:2])
                control_constraints = []
                for label_class_dict in control_label_lst:  # control_label_lst[:100]:
                    # tone:
                    tone_label_class = label_class_dict['tone'][:64]
                    tone_label_class = [tone_vocab.get(x, tone_vocab['UNK']) for x in tone_label_class]
                    tone_label_class = tone_label_class + [tone_vocab['PAD']] * (64 - len(tone_label_class))
                    tone_label_ids = th.LongTensor(tone_label_class).unsqueeze(0)
                    # vowel:
                    vowel_label_ids = label_class_dict['vowel'][:64]
                    tgt_key = {x: [] for x in range(18)}
                    yunjiao = 0
                    for i in range(min(64, len(vowel_label_ids))):
                        if len(vowel_label_ids) > i + 1 and vowel_label_ids[i + 1] == '17':
                            for y in vowel_label_ids[i].split():
                                y = int(y)
                                tgt_key[y].append(i)
                                yunjiao = y if len(tgt_key[y]) > len(tgt_key[yunjiao]) else yunjiao
                            vowel_label_ids[i] = vowel_vocab['PAD']
                        elif vowel_label_ids[i] == '17':
                            vowel_label_ids[i] = vowel_vocab['17']
                        else:
                            vowel_label_ids[i] = vowel_vocab['PAD']
                    for pos in tgt_key[yunjiao]:
                        vowel_label_ids[pos] = yunjiao
                    vowel_label_ids = vowel_label_ids + [vowel_vocab['PAD']] * (64 - len(vowel_label_ids))
                    # logger.info(f"vowel_label_ids: {vowel_label_ids}")
                    vowel_label_ids = th.LongTensor(vowel_label_ids).unsqueeze(0)
                    # length:
                    words_ = label_class_dict['words_']
                    encoded_partial_seq = [th.LongTensor([-1])]
                    right_length = args.image_size ** 2 - len(encoded_partial_seq[0])
                    right_pad = th.empty(right_length).fill_(todo_pad_token).long()
                    encoded_partial_seq = [th.cat([seq, right_pad], dim=0) for seq in encoded_partial_seq]
                    for i in range(min(len(words_), 64)):
                        if words_[i] in ["，", "。", "？", "！"]:
                            encoded_partial_seq[0][i] = tokens2id[words_[i]]
                    partial_mask = (encoded_partial_seq[0] == todo_pad_token).unsqueeze(0).expand(args.batch_size, -1)
                    length_label_ids = th.tensor(encoded_partial_seq[0]).unsqueeze(0)
                    length_label_ids = length_label_ids.masked_fill(length_label_ids == todo_pad_token, 3)

                    # logger.info(f' --> tone_label_ids: {tone_label_ids}')
                    # logger.info(f' --> vowel_label_ids: {vowel_label_ids}')
                    tgt_embs = model3.cuda()(length_label_ids.cuda())

                    langevin_fn_selected = partial(langevin_fn_tone_vowel_length, diffusion,
                                                   model_control_tone, model_control_vowel, model,
                                                   tgt_embs.expand(args.batch_size, -1, -1), partial_mask,
                                                   tone_label_ids.expand(args.batch_size, -1),
                                                   vowel_label_ids.expand(args.batch_size, -1), 0.1)
                    control_constraints.append((langevin_fn_selected, label_class_dict['tone']))

                partial_seq = control_constraints
                # print(control_constraints)
                encoded_partial_seq = [encoded_partial_seq[0] for _ in range(len(partial_seq))]
                assert len(partial_seq) == len(encoded_partial_seq)
                print(f'RUNNING FOR {len(partial_seq)} constraints.', '*-' * 20)

            elif args.eval_task_ == 'control_tone_length':
                # model_control = Classifier_POS.from_pretrained('predictability/diff_models/e2e-tgt-pos_e=6_b=10_m=bert-'
                #                                                'base-uncased_wikitext-103-raw-v1_101_wp_full_multi16_v2').cuda()
                model_control = Classifier_TONE.from_pretrained(
                    os.path.split(args.model_path)[0] +
                    '/../../../classifier_models/' + args.classifier_model_name).cuda()

                tone_vocab = {'START': 0, 'END': 1, 'UNK': 2, 'PAD': 3, 'NONE': 4, 'PING': 5, 'ZE': 6}
                tone_vocab_rev = {v: k for k, v in tone_vocab.items()}

                ################33
                control_label_lst = []
                with open(os.path.split(args.model_path)[0] + "/../../" + "control_gen/" + args.tgt_file,
                          'r', encoding='utf-8') as controlf:
                    for line in controlf:
                        control_label_lst.append(json.loads(line))
                # print(control_label_lst[:2])
                control_constraints = []
                for label_class_dict in control_label_lst:  # control_label_lst[:100]:
                    # tone:
                    label_class = label_class_dict['tone']
                    label_class = [tone_vocab.get(x, tone_vocab['UNK']) for x in label_class][:64]
                    label_class = label_class + [tone_vocab['PAD']] * (64 - len(label_class))
                    label_ids = th.LongTensor(label_class).unsqueeze(0)
                    # length:
                    words_ = label_class_dict['words_']
                    encoded_partial_seq = [th.LongTensor([-1])]
                    right_length = args.image_size ** 2 - len(encoded_partial_seq[0])
                    right_pad = th.empty(right_length).fill_(todo_pad_token).long()
                    encoded_partial_seq = [th.cat([seq, right_pad], dim=0) for seq in encoded_partial_seq]
                    for i in range(min(len(words_), 64)):
                        if words_[i] in ["，", "。", "？", "！"]:
                            encoded_partial_seq[0][i] = tokens2id[words_[i]]
                    partial_mask = (encoded_partial_seq[0] == todo_pad_token).unsqueeze(0).expand(args.batch_size, -1)
                    length_label_ids = th.tensor(encoded_partial_seq[0]).unsqueeze(0)
                    length_label_ids = length_label_ids.masked_fill(length_label_ids == todo_pad_token, 3)
                    # logger.info(f' --> label_ids: {label_ids}')
                    tgt_embs = model3.cuda()(length_label_ids.cuda())

                    langevin_fn_selected = partial(langevin_fn_tone_length, diffusion, model_control,
                                                   model3.cuda(), model,
                                                   tgt_embs.expand(args.batch_size, -1, -1), partial_mask,
                                                   label_ids.expand(args.batch_size, -1), args.coef)
                    control_constraints.append((langevin_fn_selected, label_class_dict['tone']))

                partial_seq = control_constraints
                # print(control_constraints)
                encoded_partial_seq = [encoded_partial_seq[0] for _ in range(len(partial_seq))]
                assert len(partial_seq) == len(encoded_partial_seq)
                print(f'RUNNING FOR {len(partial_seq)} constraints.', '*-' * 20)

            elif args.eval_task_ == 'control_tone':
                # model_control = Classifier_POS.from_pretrained('predictability/diff_models/e2e-tgt-pos_e=6_b=10_m=bert-'
                #                                                'base-uncased_wikitext-103-raw-v1_101_wp_full_multi16_v2').cuda()
                model_control = Classifier_TONE.from_pretrained(
                    os.path.split(args.model_path)[0] +
                    '/../../../classifier_models/' + args.classifier_model_name).cuda()

                tone_vocab = {'START': 0, 'END': 1, 'UNK': 2, 'PAD': 3, 'NONE': 4, 'PING': 5, 'ZE': 6}
                tone_vocab_rev = {v: k for k, v in tone_vocab.items()}

                ################33
                control_label_lst = []
                with open(os.path.split(args.model_path)[0] + "/../../" + "control_gen/" + args.tgt_file,
                          'r') as controlf:
                    for line in controlf:
                        control_label_lst.append(json.loads(line))
                print(control_label_lst[:2])
                control_constraints = []
                for label_class_dict in control_label_lst:  # control_label_lst[:100]:
                    # tone:
                    label_class = label_class_dict['tone']
                    label_class = [tone_vocab.get(x, tone_vocab['UNK']) for x in label_class][:64]
                    label_class = label_class + [tone_vocab['PAD']] * (64 - len(label_class))
                    label_ids = th.LongTensor(label_class).unsqueeze(0)

                    langevin_fn_selected = partial(langevin_fn_tone, diffusion, model_control,
                                                   model3.cuda(), model,
                                                   label_ids.expand(args.batch_size, -1), 0.1)
                    control_constraints.append((langevin_fn_selected, label_class_dict['tone']))

                partial_seq = control_constraints
                # print(control_constraints)
                encoded_partial_seq = [encoded_partial_seq[0] for _ in range(len(partial_seq))]
                assert len(partial_seq) == len(encoded_partial_seq)
                print(f'RUNNING FOR {len(partial_seq)} constraints.', '*-' * 20)

            elif args.eval_task_ == 'control_line':
                # model_control = Classifier_POS.from_pretrained('predictability/diff_models/e2e-tgt-pos_e=6_b=10_m=bert-'
                #                                                'base-uncased_wikitext-103-raw-v1_101_wp_full_multi16_v2').cuda()
                model_control = Classifier_LINE.from_pretrained(
                    os.path.split(args.model_path)[0] +
                    '/../../../self_classifier/sonnet_n_line/save_models/' + args.classifier_model_name).cuda()

                line_map = {str(i+1): i for i in range(14)}

                ################33
                control_label_lst = [['14']]*50
                control_constraints = []
                for label_class_dict in control_label_lst:  # control_label_lst[:100]:
                    # print('label_class_dict:', label_class_dict)
                    label_ids = [line_map[x] for x in label_class_dict]
                    label_ids = th.LongTensor(label_ids).unsqueeze(0)
                    langevin_fn_selected = partial(langevin_fn_line, diffusion, model_control,
                                                   model3.cuda(), model,
                                                   label_ids, 0.1)
                    control_constraints.append((langevin_fn_selected, label_class_dict))

                partial_seq = control_constraints
                # print(control_constraints)
                encoded_partial_seq = [encoded_partial_seq[0] for _ in range(len(partial_seq))]
                assert len(partial_seq) == len(encoded_partial_seq)
                print(f'RUNNING FOR {len(partial_seq)} constraints.', '*-' * 20)

            elif args.eval_task_ == 'control_line_rhyme':
                # model_control = Classifier_POS.from_pretrained('predictability/diff_models/e2e-tgt-pos_e=6_b=10_m=bert-'
                #                                                'base-uncased_wikitext-103-raw-v1_101_wp_full_multi16_v2').cuda()
                model_control_line = Classifier_LINE.from_pretrained(
                    os.path.split(args.model_path)[0] +
                    '/../../../self_classifier/sonnet_n_line/save_models/' + args.classifier_model_name).cuda()
                line_map = {str(i+1): i for i in range(14)}

                model_control_rhyme = Classifier_RHYME.from_pretrained(
                    os.path.split(args.model_path)[0] +
                    '/../../../classifier_models/' + args.classifier_model_name_2).cuda()
                rhyme_vocab = {'START': 0, 'END': 1, 'UNK': 2, 'PAD': 3}
                rhyme_lst = [str(x) for x in range(6220)]
                for x in rhyme_lst:
                    rhyme_vocab[x] = len(rhyme_vocab)


                ################33
                # control_label_lst = [['14']]*50
                control_label_lst = []
                with open(os.path.split(args.model_path)[0] + "/../../" + "control_gen/" + args.tgt_file,
                          'r') as controlf:
                    for line in controlf:
                        control_label_lst.append(json.loads(line))
                control_constraints = []
                for label_class_dict in control_label_lst[:50]:  # control_label_lst[:100]:
                    # print('label_class_dict:', label_class_dict)
                    line_label_ids = [line_map['14']]
                    line_label_ids = th.LongTensor(line_label_ids).unsqueeze(0)

                    rhyme_label_ids = label_class_dict['rhyme'][:196]
                    for i in range(min(196, len(rhyme_label_ids))):
                        if len(rhyme_label_ids) > i + 3 and \
                                rhyme_label_ids[i + 1] == rhyme_label_ids[i + 2] == rhyme_label_ids[i + 3] == '-1':
                            rhyme_label_ids[i] = rhyme_vocab[rhyme_label_ids[i]]
                        elif rhyme_label_ids[i] == '-1':
                            rhyme_label_ids[i] = rhyme_vocab['6219']
                        else:
                            rhyme_label_ids[i] = rhyme_vocab['PAD']
                    rhyme_label_ids = rhyme_label_ids + [rhyme_vocab['PAD']] * (196 - len(rhyme_label_ids))
                    # logger.info(f"vowel_label_ids: {vowel_label_ids}")
                    rhyme_label_ids = th.LongTensor(rhyme_label_ids).unsqueeze(0)

                    langevin_fn_selected = partial(langevin_fn_line_rhyme, diffusion,
                                                   model_control_line, model_control_rhyme, model,
                                                   line_label_ids, rhyme_label_ids.expand(args.batch_size, -1), 0.1)
                    control_constraints.append((langevin_fn_selected, label_class_dict['rhyme']))

                partial_seq = control_constraints
                # print(control_constraints)
                encoded_partial_seq = [encoded_partial_seq[0] for _ in range(len(partial_seq))]
                assert len(partial_seq) == len(encoded_partial_seq)
                print(f'RUNNING FOR {len(partial_seq)} constraints.', '*-' * 20)

            elif args.eval_task_ == 'control_tree':
                # model_control = Classifier_Tree.from_pretrained(
                #     'predictability/diff_models/e2e-tgt-tree_e=20_b=32_m=bert-base-uncased_'
                #     'wikitext-103-raw-v1_101_wp_full_multi16_v2').cuda()
                model_control = Classifier_Tree.from_pretrained(
                    os.path.split(args.model_path)[0] +
                    '/../../../'
                    'classifier_models/e2e-tgt-tree_e=6_b=10_m=bert-base-uncased_wikitext-103-raw-v1_101_wp_None'
                ).cuda()

                # print(model_control)

                import benepar
                from tree_helper import chart_from_tree, pad_charts, padded_chart_from_spans
                parser = benepar.Parser("benepar_en3")
                tree_vocab = parser._parser.config["label_vocab"]
                tree_vocab_rev = {v: k for k, v in tree_vocab.items()}

                ###############
                control_label_lst = []
                with open(os.path.split(args.model_path)[0] + '/../../' + 'control_gen/target_tree.json',
                          'r') as controlf:
                    for line in controlf:
                        control_label_lst.append(json.loads(line))
                # print(control_label_lst[:1])
                control_constraints = []
                for label_class_dict in control_label_lst[100:]:
                    padded_chart = th.LongTensor(label_class_dict['padded_chart'])
                    words_ = label_class_dict['words_']
                    label_ids = padded_chart
                    langevin_fn_selected = partial(langevin_fn_tree, 0.0005, model_control, model3.cuda(),
                                                   label_ids.expand(args.batch_size, -1, -1),
                                                   0.1)
                    control_constraints.append((langevin_fn_selected, [label_class_dict['tree']]))

                partial_seq = control_constraints
                # print(control_constraints)
                encoded_partial_seq = [encoded_partial_seq[0] for _ in range(len(partial_seq))]
                assert len(partial_seq) == len(encoded_partial_seq)
                print(f'RUNNING FOR {len(partial_seq)} constraints.', '*-' * 20)  # 100

            elif args.eval_task_ == 'control_span':
                model_control = Classifier_Tree.from_pretrained(
                    os.path.split(args.model_path)[0] +
                    '/../../../'
                    'classifier_models/e2e-tgt-tree_e=6_b=10_m=bert-base-uncased_wikitext-103-raw-v1_101_wp_None'
                ).cuda()

                import benepar
                from tree_helper import chart_from_tree, pad_charts, padded_chart_from_spans
                parser = benepar.Parser("benepar_en3")
                tree_vocab = parser._parser.config["label_vocab"]
                tree_vocab_rev = {v: k for k, v in tree_vocab.items()}

                ###############
                control_label_lst = []
                with open(os.path.split(args.model_path)[0] + '/../../' + 'control_gen/target_spans.json',
                          'r') as controlf:
                    for line in controlf:
                        control_label_lst.append(json.loads(line))
                # print(control_label_lst[:1])
                control_constraints = []
                for label_class_dict in control_label_lst:
                    spans = label_class_dict['spans']
                    spans = [(a + 1, b + 1, c) for (a, b, c) in spans]
                    assert len(spans) == 1
                    padded_charts = padded_chart_from_spans(tree_vocab, spans)
                    padded_charts = th.LongTensor(padded_charts).unsqueeze(0)
                    print(padded_charts.shape, 'built from spans. ')  # torch.Size([1, 64, 64])
                    label_ids = padded_charts
                    langevin_fn_selected = partial(langevin_fn_tree, 0.1, model_control, model3.cuda(),
                                                   label_ids.expand(args.batch_size, -1, -1),
                                                   0.1)
                    print((str(label_class_dict['spans'][0]),))
                    control_constraints.append((langevin_fn_selected, (str(label_class_dict['spans'][0]),)
                                                ))

                partial_seq = control_constraints
                # print(control_constraints)
                encoded_partial_seq = [encoded_partial_seq[0] for _ in range(len(partial_seq))]
                assert len(partial_seq) == len(encoded_partial_seq)
                print(f'RUNNING FOR {len(partial_seq)} constraints.', '*-' * 20)


            elif args.eval_task_ == 'control_length':
                control_label_lst = []
                with open(os.path.split(args.model_path)[0] + "/../../" + "control_gen/" + args.tgt_file,
                          'r') as controlf:
                    for line in controlf:
                        control_label_lst.append(json.loads(line))
                print(control_label_lst[:2])
                control_constraints = []
                for label_class_dict in control_label_lst:
                    encoded_partial_seq = [th.LongTensor([-1])]
                    # print(encoded_partial_seq)
                    assert len(encoded_partial_seq) == 1
                    right_length = args.image_size ** 2 - len(encoded_partial_seq[0])
                    # logger.info(f'args.image_size: {args.image_size}')  # 8
                    # right_length = args.tgt_len - len(encoded_partial_seq[0])
                    # assert args.tgt_len > len(encoded_partial_seq[0])
                    right_pad = th.empty(right_length).fill_(todo_pad_token).long()
                    # print(right_pad, right_length, len(encoded_partial_seq[0]))
                    # tensor([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    #         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    #         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    #         -1, -1, -1, -1, -1, -1, -1, -1, -1]) 63 1

                    encoded_partial_seq = [th.cat([seq, right_pad], dim=0) for seq in encoded_partial_seq]
                    # encoded_partial_seq[0][target_length - 1] = tokens2id['END']
                    # 尝试：连续两句同长
                    # encoded_partial_seq[0][target_length] = tokens2id['START']
                    # encoded_partial_seq[0][2*target_length - 1] = tokens2id['END']
                    words_ = label_class_dict['words_']
                    for i in range(min(len(words_), 64)):
                        if words_[i] in ["，", "。", "？", "！"]:
                            encoded_partial_seq[0][i] = tokens2id[words_[i]]

                    # print(encoded_partial_seq[0], todo_pad_token)
                    # tensor([ 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    #         -1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    #         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    #         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]) -1
                    # partial_mask = (encoded_partial_seq[0] == todo_pad_token).unsqueeze(0).expand(args.batch_size, -1)
                    # print(partial_mask[0])
                    # 10/0
                    # label = encoded_partial_seq[0]
                    # label_ids = th.tensor(label).unsqueeze(0)
                    # logger.info(f'label_ids: {label_ids}')
                    # tensor([[ 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    #          -1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    #          -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    #          -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

                    # label_ids = label_ids.masked_fill(label_ids == todo_pad_token, 3)
                    # logger.info(f' --> label_ids: {label_ids}')     # START位置是0，END位置是1，其余是3
                    # tensor([[0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3,
                    #          3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                    #          3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]])

                    partial_mask = (encoded_partial_seq[0] == todo_pad_token).unsqueeze(0).expand(args.batch_size, -1)
                    label_ids = th.tensor(encoded_partial_seq[0]).unsqueeze(0)
                    label_ids = label_ids.masked_fill(label_ids == todo_pad_token, 3)
                    tgt_embs = model3.cuda()(label_ids.cuda())
                    langevin_fn_selected = partial(langevin_fn_length, 0.01, diffusion, partial_mask, model,
                                                   tgt_embs.expand(args.batch_size, -1, -1), 0.1)
                    # logger.info(f'tgt_embs.expand(args.batch_size, -1, -1): {tgt_embs.expand(args.batch_size, -1, -1)}')
                    control_constraints.append((langevin_fn_selected, label_class_dict['words_']))

                partial_seq = control_constraints
                # print(control_constraints)
                encoded_partial_seq = [encoded_partial_seq[0] for _ in range(len(partial_seq))]
                assert len(partial_seq) == len(encoded_partial_seq)
                print(f'RUNNING FOR {len(partial_seq)} constraints.', '*-' * 20)


        elif args.eval_task_ == 'interpolate':
            print(encoded_partial_seq)
            assert len(encoded_partial_seq[0]) == len(encoded_partial_seq[1])
            assert len(encoded_partial_seq[0]) > 1
        # print(encoded_partial_seq[1], len(encoded_partial_seq[1]))
        # encoded_partial_seq[1]: tensor([0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3,
        #         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        #         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]) 64
    # else: text, using huggingface tokenizer.

    logger.log("sampling...")
    # logger.info(f"1st encoded_seq: {encoded_partial_seq[0]}")
    # logger.info(f"1st control_helper: {partial_seq[0]}")
    sample_dict = {}
    middle_dict = {}

    def decode_helper(args, step, key, sample):
        result_dict = {}
        set_seed(101)
        model, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                       os.path.split(args.model_path)[0])

        arr = sample
        word_lst = rounding_func(args.experiment, arr, model, tokenizer)
        result_dict[(step, tuple(key))] = word_lst
        return result_dict

    # model3 = get_weights(model_embs, args)
    # wandb.init(
    #     project=os.getenv("WANDB_PROJECT", "diffusion_sample"),
    #     name=f'infill_{args.eval_task_}_{args.infill_notes}',
    # )
    if True:
        i = 0
        for (encoded_seq, control_helper) in zip(encoded_partial_seq, partial_seq):
            i += 1
            # if i > 10:
            #     break
            all_images = []
            all_labels = []
            print(args.num_samples, encoded_seq.shape, 'encoded_seq.shape')  # 50 torch.Size([64]) encoded_seq.shape
            while len(all_images) * args.batch_size < args.num_samples:
                model_kwargs = {}
                # print(encoded_seq.shape)
                # logger.info(f"encoded_seq.shape: {encoded_seq.shape}")  # encoded_seq.shape: torch.Size([64])
                encoded_seq = encoded_seq.unsqueeze(0).expand(args.batch_size, -1)
                # print(model_embs.weight.device, encoded_seq.device)
                # logger.info(f"model_embs.weight.device: {model_embs.weight.device}")  # model_embs.weight.device: cuda:0
                # logger.info(f"encoded_seq.device: {encoded_seq.device}")  # encoded_seq.device: cpu
                partial_mask_temp = (encoded_seq == todo_pad_token).view(args.batch_size, -1)
                # encoded_seq[encoded_seq == todo_pad_token] = 0
                encoded_seq.masked_fill_(encoded_seq == todo_pad_token, 3)

                encoded_seq_hidden = model_embs(encoded_seq.cuda())
                seqlen = encoded_seq.size(1)
                if args.model_arch == '1d-unet':
                    encoded_seq_hidden = encoded_seq_hidden.permute(0, 2, 1)
                    partial_mask = partial_mask_temp.unsqueeze(1).expand(-1, args.in_channel, -1)
                    sample_shape = (args.batch_size, args.in_channel, seqlen)
                else:  # √
                    partial_mask = partial_mask_temp.unsqueeze(-1).expand(-1, -1, args.in_channel)
                    sample_shape = (args.batch_size, seqlen, args.in_channel,)
                # print(  # f"partial_mask: {partial_mask}\n"
                #     f"encoded_seq_hidden.shape: {encoded_seq_hidden.shape}")
                # partial_mask: tensor([[[False, False, False, ..., False, False, False],
                #                        [False, False, False, ..., False, False, False],
                #                        [False, False, False, ..., False, False, False],
                #                        ...,
                #                        [False, False, False, ..., False, False, False],
                #                        [False, False, False, ..., False, False, False],
                #                        [False, False, False, ..., False, False, False]]])
                # encoded_seq_hidden.shape: torch.Size([64, 64, 16])

                if args.eval_task_.startswith('control'):
                    langevin_fn_selected, label_class_attributes = control_helper
                    print('-*' * 200, label_class_attributes, '-*' * 200)
                    # loop_func_ = diffusion.p_sample_loop_langevin_progressive
                    logger.info(f"args.use_ddim: {args.use_ddim}")  # True
                    if args.use_ddim:
                        loop_func_ = diffusion.ddim_sample_loop_progressive
                    else:
                        loop_func_ = diffusion.p_sample_loop_progressilenve

                    step = 0
                    for sample in loop_func_(
                            model,
                            sample_shape,
                            denoised_fn=partial(denoised_fn_round, args, model3.cuda()),
                            # denoised_fn=partial(langevin_early, model_control, model3.cuda(),
                            #                     label_ids.expand(args.batch_size, -1), 0.1),
                            clip_denoised=args.clip_denoised,
                            model_kwargs=model_kwargs,
                            device=encoded_seq_hidden.device,
                            langevin_fn=langevin_fn_selected,
                            eta=args.eta,
                            # langevin_func=partial(langevin_func, model_control,
                            #                       label_ids.expand(args.batch_size, -1), 0.01),
                    ):
                        # if args.print_middle_sent:
                        #     rst_dict = decode_helper(args, step, label_class_attributes, copy.deepcopy(sample["sample"]))
                        #     with open(f"out_gen/middle_steps_{args.tgt_file.split('_')[-1].split('.')[0]}.json", "a", encoding='utf-8') as f:
                        #         print(rst_dict, file=f)
                        #     step += 1
                        # print("checkpoint in loop_func_")
                        final = sample["sample"]

                    logger.info(f"args.verbose: {args.verbose}")  # pipe
                    if args.verbose == 'yes':
                        with open(f'debug_lst_lgv_{args.infill_notes}.json', 'w') as f:
                            json.dump(debug_lst, f)
                        if args.eval_task_ == 'control_tree':
                            label_ids = label_ids.expand(args.batch_size, -1, -1).cuda()
                            tgt_embs = model3(label_ids[:, final.size(1):])
                        else:
                            label_ids = label_ids.expand(args.batch_size, -1).cuda()
                            tgt_embs = model3(label_ids[:, final.size(1):])

                        if args.eval_task_ == 'control_attributes':
                            label_ids2 = label_ids.clone()
                            label_ids2[:, :65] = -100
                            # print(label_ids2[:, 65:])
                            # print(final.shape, tgt_embs.shape)
                            input_embs = th.cat([final, tgt_embs], dim=1)
                            model_out = model_control(input_embs=input_embs,
                                                      labels=label_ids2)
                            print(model_out.loss, 'final end')
                            loss_fn = th.nn.CrossEntropyLoss(reduction='none')
                            shifted_logits = model_out.logits[:, :-1].contiguous()
                            shifted_labels = label_ids2[:, 1:].contiguous()
                            loss = loss_fn(shifted_logits.view(-1, shifted_logits.size(-1)),
                                           shifted_labels.view(-1)).reshape(shifted_labels.shape)
                            print(loss.sum(dim=-1).tolist())
                            word_lst = rounding_func(args.experiment, final, model3, tokenizer)
                            print(len(word_lst))
                            for ww, ll in zip(word_lst, loss.sum(dim=-1).tolist()):
                                print([ww], ll)
                        elif args.eval_task_ == 'control_pos':
                            model_out = model_control(input_embs=final,
                                                      pos_ids=label_ids)
                            print(model_out.loss, 'final end')
                            loss_fn = th.nn.CrossEntropyLoss(reduction='none')
                            shifted_logits = model_out.logits
                            shifted_labels = label_ids
                            loss = loss_fn(shifted_logits.view(-1, shifted_logits.size(-1)),
                                           shifted_labels.view(-1)).reshape(shifted_labels.shape)
                            print(loss)
                            print(loss.sum(dim=-1).tolist())
                            word_lst = rounding_func(args.experiment, final, model3, tokenizer)
                            print(len(word_lst))
                            for ww, ll in zip(word_lst, loss.sum(dim=-1).tolist()):
                                print([ww], ll)
                        elif args.eval_task_ == 'control_tree':
                            model_out = model_control(input_embs=final,
                                                      parse_chart=label_ids)
                            print(model_out.loss, 'final end')
                            loss_fn = th.nn.CrossEntropyLoss(reduction='none')
                            shifted_logits = model_out.logits
                            shifted_labels = label_ids
                            loss = loss_fn(shifted_logits.view(-1, shifted_logits.size(-1)),
                                           shifted_labels.view(-1)).reshape(shifted_labels.shape)
                            print(loss, loss.shape)
                            # print(loss.sum(dim=-1).tolist())
                            word_lst = rounding_func(args.experiment, final, model3, tokenizer)
                            print(len(word_lst))
                            for ww, ll in zip(word_lst, loss.sum(dim=-1).sum(dim=-1).tolist()):
                                # print([ww], ll)
                                logger.info(f"[ww]: {[ww]}, ll: {ll}")

                            print(parse_lst[0])
                        else:
                            label_ids2 = th.cat([label_ids[:, :final.size(1)], label_ids], dim=1)
                            label_ids2[:, :64 * 2 + 1] = -100
                            tt = th.LongTensor([0]).expand(final.size(0)).to(final.device)
                            prev_sample = diffusion.q_sample(final, tt)
                            input_embs = th.cat([final, prev_sample, tgt_embs], dim=1)
                            model_out = model_control(input_embs=input_embs,
                                                      labels=label_ids2)
                            print(model_out.loss, 'final end')
                            loss_fn = th.nn.CrossEntropyLoss(reduction='none')
                            shifted_logits = model_out.logits[:, :-1].contiguous()
                            shifted_labels = label_ids2[:, 1:].contiguous()
                            loss = loss_fn(shifted_logits.view(-1, shifted_logits.size(-1)),
                                           shifted_labels.view(-1)).reshape(shifted_labels.shape)
                            print(loss.sum(dim=-1).tolist())
                            word_lst = rounding_func(args.experiment, final, model3, tokenizer)
                            print(len(word_lst))
                            for ww, ll in zip(word_lst, loss.sum(dim=-1).tolist()):
                                print([ww], ll)




                else:
                    label_class_attributes = control_helper
                    loop_func_ = diffusion.p_sample_loop_progressive_infill

                    for sample in loop_func_(
                            model,
                            sample_shape,
                            encoded_seq_hidden,
                            partial_mask,
                            denoised_fn=partial(denoised_fn_round, args, model3.cuda()),
                            clip_denoised=args.clip_denoised,
                            model_kwargs=model_kwargs,
                            device=encoded_seq_hidden.device,
                            greedy=False,
                    ):
                        final = sample["sample"]

                sample = final
                # logger.info(f"sample: {sample}")
                # sample: tensor([[[-0.1312, 0.2085, -1.7784, ..., -1.2753, 0.8636, -0.2230],
                #                  [0.9875, 0.0148, -0.2951, ..., 0.7778, -1.7666, -0.1038],
                #                  [-0.0356, 0.1451, -0.1663, ..., 0.4945, -0.0103, -0.0515],
                #                 ...,
                #                  ...,
                #                  [0.1735, 1.9400, 1.2144, ..., -0.7579, 1.2890, -0.1517],
                #                  [0.5879, -1.2383, -0.2532, ..., 0.7688, -0.6292, 0.3778],
                #                  [-0.0356, 0.1451, -0.1663, ..., 0.4945, -0.0103, -0.0515]]],
                #                device='cuda:0')
                logger.info(f"sample.shape: {sample.shape}")  # torch.Size([64, 64, 16])

                logger.info(f"args.model_arch: {args.model_arch}")  # transformer
                if args.model_arch == '1d-unet':
                    print(sample.shape)
                    sample = sample.permute(0, 2, 1)
                    print(sample.shape)

                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
                logger.info(f"gathered_samples[0].shape: {gathered_samples[0].shape}")  # torch.Size([64, 64, 16])
                logger.info(f"len(all_images): {len(all_images)}")  # 1
                if args.class_cond:
                    gathered_labels = [
                        th.zeros_like(classes) for _ in range(dist.get_world_size())
                    ]
                    dist.all_gather(gathered_labels, classes)
                    all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
                logger.log(f"created {len(all_images)} * {args.batch_size} samples")  # 1 * 64

            arr = np.concatenate(all_images, axis=0)
            arr = arr[: args.num_samples]
            if args.verbose == 'pipe':
                if args.eval_task_ == 'control_line':
                    sample_dict[tuple([i] + label_class_attributes)] = arr
                else:
                    sample_dict[tuple(label_class_attributes)] = arr
                print(f'writing to sample_dict, for class {" ".join(label_class_attributes)}')
            # if args.print_middle_sent:
            #     out_path_pipe = f"out_gen/middle_steps_{args.tgt_file.split('_')[-1].split('.')[0]}.json"
            #     logger.info(f'out_path_pipe: {out_path_pipe}')
            #     fout = open(out_path_pipe, 'a', encoding='utf-8')
            #     from improved_diffusion.gaussian_diffusion import middle_sample
            #     for i in range(len(middle_sample)):
            #         sub_sample = middle_sample[i]
            #         print(f"sub_sample (part): {sub_sample[0][0][0]}")
            #         # gathered_samples = [th.zeros_like(sub_sample) for _ in range(dist.get_world_size())]
            #         # dist.all_gather(gathered_samples, sub_sample)  # gather not supported with NCCL
            #         #
            #         # arr = np.concatenate([sub_sample.cpu().numpy() for sub_sample in gathered_samples], axis=0)
            #         # arr = arr[:args.num_samples]
            #         # logger.info(f'[MIDDLE SENT] writing to middle_dict, {arr}')
            #         rst_dict = decode_helper(args, i, label_class_attributes, copy.deepcopy(sub_sample))
            #         print(rst_dict, file=fout)
            #     fout.close()
            #     logger.info(f'[MIDDLE SENT] written the decoded output to {out_path_pipe}')

    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'

    dist.barrier()
    logger.log("sampling complete")

    def decode_helper(args, sample_dict, diff_model=None):
        result_dict = {}
        if not diffusion.training_mode.startswith('e2e'):
            logger.log('decode by rounding. ')
            set_seed(101)
            model, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                           os.path.split(args.model_path)[0])

        for k, v in sample_dict.items():
            arr = v
            if diffusion.training_mode.startswith('e2e'):
                word_lst_e2e = []
                # print('decoding for e2e', )
                x_t = th.tensor(arr).cuda()
                # print(x_t.shape)
                if args.model_arch == 'conv-unet':
                    reshaped_x_t = x_t.view(x_t.size(0), -1, x_t.size(-1))
                else:
                    reshaped_x_t = x_t
                logits = diff_model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab
                # logger.info(f"logits.shape: {logits.shape}")  # logits.shape: torch.Size([50, 64, 821])
                cands = th.topk(logits, k=1, dim=-1)

                tokenizer = load_tokenizer(args.modality, args.experiment, os.path.split(args.model_path)[0])
                for seq in cands.indices:
                    tokens = " ".join([tokenizer[x[0].item()] for x in seq])
                    # logger.info(f"tokens: {tokens}")    # sentence
                    word_lst_e2e.append(tokens)
                # logger.info(f"last tokens: {tokens}")
                word_lst = word_lst_e2e
            else:
                word_lst = rounding_func(args.experiment, arr, model, tokenizer)
            result_dict[k] = word_lst
        return result_dict

    if args.verbose == 'pipe':
        logger.info(f'sampled for {len(sample_dict)} control tasks')  # 100
        # out_path_pipe = os.path.join(args.out_dir, f"{model_base_name}.infill_{args.eval_task_}_{args.infill_notes}.json")
        # out_path_pipe = f"/home/LAB/liucm/Diffusion_LM/improved_diffusion/out_gen/infill_{args.eval_task_}_{args.infill_notes}.json"
        out_path_pipe = f"out_gen/{args.eval_task_}/infill_{args.infill_notes}_{args.tgt_file.split('_')[-1].split('.')[0]}.json"
        logger.info(f'out_path_pipe: {out_path_pipe}')
        fout = open(out_path_pipe, 'w')
        result_dict = decode_helper(args, sample_dict, diff_model=model)
        for k, word_lst in result_dict.items():
            print({k: word_lst}, file=fout)
        print("", file=fout)
        fout.close()
        logger.info(f'written the decoded output to {out_path_pipe}')
        out_path2 = out_path_pipe
        if args.print_middle_sent:
            from improved_diffusion.gaussian_diffusion import middle_sample
            print(len(middle_sample), middle_sample[0], middle_sample[-1])
            idx = 0
            for i in range(len(middle_sample)):
                sub_sample = middle_sample[i]
                print(f"sub_sample (part): {sub_sample[0][0][0]}")
                gathered_samples = [th.zeros_like(sub_sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sub_sample)  # gather not supported with NCCL

                arr = np.concatenate([sub_sample.cpu().numpy() for sub_sample in gathered_samples], axis=0)
                arr = arr[:args.num_samples]
                middle_dict[tuple([idx] + label_class_attributes)] = copy.deepcopy(arr)
                idx = (idx + 1) % 200
                # logger.info(f'[MIDDLE SENT] writing to middle_dict, {arr}')
            out_path_pipe = f"out_gen/middle_steps_{args.tgt_file.split('_')[-1].split('.')[0]}.json"
            logger.info(f'out_path_pipe: {out_path_pipe}')
            fout = open(out_path_pipe, 'w')
            result_dict = decode_helper(args, middle_dict, diff_model=model)
            for k, word_lst in result_dict.items():
                print({k: word_lst}, file=fout)
            fout.close()
            logger.info(f'[MIDDLE SENT] written the decoded output to {out_path_pipe}')



    elif args.verbose == 'yes':

        if diffusion.training_mode.startswith('e2e'):
            word_lst_e2e = []
            print('decoding for e2e', )
            print(sample.shape)
            x_t = sample
            if args.model_arch == 'conv-unet':
                reshaped_x_t = x_t.view(x_t.size(0), -1, x_t.size(-1))
            else:
                reshaped_x_t = x_t
            logits = model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab
            cands = th.topk(logits, k=1, dim=-1)
            tokenizer = load_tokenizer(args.modality, args.experiment, os.path.split(args.model_path)[0])
            for seq in cands.indices:
                tokens = " ".join([tokenizer[x[0].item()] for x in seq])
                word_lst_e2e.append(tokens)
            word_lst = word_lst_e2e
        else:
            logger.log('decode by rounding. ')
            print('load_models')
            set_seed(101)
            print(os.path.split(args.model_path)[0])
            model, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                           os.path.split(args.model_path)[0])
            print('rounding')
            word_lst = rounding_func(args.experiment, arr, model, tokenizer)

        # out_path2 = os.path.join(args.out_dir, f"{model_base_name}.infill_{args.eval_task_}_{shape_str}_{args.infill_notes}.txt")
        out_path2 = f"/home/LAB/liucm/diffusion_lm/improved_diffusion/out_gen/infill_{args.eval_task_}_{shape_str}_{args.infill_notes}.txt"
        fout = open(out_path2, 'w')
        for (xx) in zip(word_lst):
            print(xx[0], file=fout)
        fout.close()
        print(f'written the decoded output to {out_path2}')

        ##############
        # out_path2 = os.path.join(args.out_dir, f"{model_base_name}.infill_{args.eval_task_}_{shape_str}_{args.infill_notes}.json")
        fout = open(out_path2, 'w')
        for (xx) in zip(word_lst):
            print(json.dumps(xx), file=fout)
        fout.close()
        print(f'written the decoded output to {out_path2}')

    args.out_path2 = out_path2
    return args


def create_argparser():
    defaults = dict(
        data_dir="", clip_denoised=False, use_ddim=False, eta=1.0, num_samples=50, batch_size=1, model_path="",
        out_dir="diffusion_lm/improved_diffusion/out_gen",
        emb_scale_factor=1.0, split='train', debug_path='', eval_task_='infill',
        partial_seq="", partial_seq_file="", verbose='yes', tgt_len=15, t_merge=200, interp_coef=0.5, infill_notes='',
        start_idx=0, end_idx=0, classifier_model_name='ci-tone_epochs=20', classifier_model_name_2='ci-vowel_epochs=20',
        print_middle_sent=False, change_num_steps=200, tgt_file="target_same_as_AAAI.json", coef=0.0001
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


#
# def create_argparser():
#     defaults = dict(
#         clip_denoised=True,
#         num_samples=50,#10000,
#         batch_size=16,
#         use_ddim=False,
#         model_path="",
#         model_arch='conv-unet',
#         verbose='yes',
#         out_dir="diffusion_lm/improved_diffusion/out_gen",
#         partial_seq=""
#     )
#     text_defaults = dict(modality='text',
#                          dataset_name='wikitext',
#                          dataset_config_name='wikitext-2-raw-v1',
#                          model_name_or_path='predictability/diff_models/compress_e=5_b=60_m=gpt2_wikitext-103-raw-v1_None',
#                          experiment='gpt2_pre_compress', model_arch='trans-unet',
#                          preprocessing_num_workers=1)
#     defaults.update(model_and_diffusion_defaults())
#     defaults.update(text_defaults)
#     # defaults.update(model_and_diffusion_defaults())
#     parser = argparse.ArgumentParser()
#     add_dict_to_argparser(parser, defaults)
#     return parser

def eval(args):
    if args.modality == 'e2e-tgt':
        model_name_path = "predictability/diff_models/e2e-tgt_e=15_b=20_m=gpt2_wikitext-103-raw-v1_101_None"

        COMMAND = f"python scripts/ppl_under_ar.py " \
                  f"--model_path {args.model_path} " \
                  f"--modality {args.modality}  --experiment random " \
                  f"--model_name_or_path {model_name_path} " \
                  f"--input_text {args.out_path2}  --mode eval"
        print(COMMAND)
        os.system(COMMAND)


if __name__ == "__main__":
    args = main()
    import numpy as np

    if args.verbose != 'pipe':
        eval(args)
