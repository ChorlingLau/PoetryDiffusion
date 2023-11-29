import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
import os
import torch
from collections import Counter, defaultdict


def helper_tokenize_encode(sentence_lst, vocab_dict, model, seqlen, data_args, padding_mode, ):
    result_train_lst = []
    group_lst = defaultdict(list)
    with torch.no_grad():
        for input_ids in sentence_lst:
            tokenized_ = [vocab_dict.get(x, vocab_dict['UNK']) for x in input_ids]
            input_ids = [0] + tokenized_ + [1]
            group_lst['word_ids'].append(input_ids)
        print(group_lst['word_ids'][:2])

        if padding_mode == 'block':
            print('padding mode is block')
            concatenated_examples = {k: sum(group_lst[k], []) for k in group_lst.keys()}
            total_length = len(concatenated_examples[list(group_lst.keys())[0]])
            block_size = seqlen
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            group_lst = {
                k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
        elif padding_mode == 'pad':
            print('padding mode is pad')
            max_length = seqlen
            group_lst['word_ids'] = _collate_batch_helper(group_lst['word_ids'], vocab_dict['PAD'], max_length)

        for input_ids in group_lst['word_ids']:
            if data_args.experiment.startswith('random'):
                hidden_state = model(torch.tensor(input_ids))
            elif data_args.experiment == 'gpt2_pre_compress':
                input_ids2 = torch.tensor(input_ids).to(model.device)
                input_embs = model.transformer.wte(input_ids2)  # input_embs
                hidden_state = model.down_proj(input_embs)
                hidden_state = hidden_state * data_args.emb_scale_factor
            elif data_args.experiment == 'glove':
                hidden_state = model(torch.tensor(input_ids))
            result_train_lst.append({'input_ids': input_ids, 'hidden_states': hidden_state.cpu().tolist()})

    return result_train_lst


def get_corpus_rocstory(data_args, model, image_size, padding_mode='block',
                        split='train', load_vocab=None):
    import csv, torch, json
    from spacy.lang.en import English
    from spacy.lang.zh import Chinese

    if data_args.experiment_mode == 'lm':
        if data_args.modality == 'e2e-tgt' and data_args.notes == 'sonnet399':
            print('loading dataset from sonnet dataset')
            sentence_lst = []
            nlp = English()
            tokenizer = nlp.tokenizer
            if split == 'train':
                print('loading form the TRAIN set')
                path = f'{data_args.e2e_train}/poem_train.json'
            elif split == 'valid':
                print('loading form the VALID set')
                path = f'{data_args.e2e_train}/poem_valid.json'
            elif split == 'test':
                print('loading form the TEST set')
                path = f'{data_args.e2e_train}/poem_valid.json'
            elif split == 'debug':
                print('loading form the DEBUG set')
                path = data_args.debug_path
                import json
                with open(path, 'r') as ff:
                    for line in ff:
                        sentence_lst.append(json.loads(line)[0].split(' '))
                sentence_lst = sentence_lst + sentence_lst
            if split in ['train', 'valid', 'test']:
                with open(path, 'r') as ff:
                    t = json.load(ff)
                    for row in t:
                        word_lst = row['text']
                        word_lst = [x.text for x in tokenizer(word_lst)]
                        sentence_lst.append(word_lst)
            print(sentence_lst[:2])
        elif data_args.modality == 'e2e-tgt' and data_args.notes == 'sonnet3355':
            print('loading dataset from sonnet dataset')
            sentence_lst = []
            nlp = English()
            tokenizer = nlp.tokenizer
            if split == 'train':
                print('loading form the TRAIN set')
                path = f'{data_args.e2e_train}/sonnet_train.txt'
            elif split == 'valid':
                print('loading form the VALID set')
                path = f'{data_args.e2e_train}/sonnet_valid.txt'
            elif split == 'test':
                print('loading form the TEST set')
                path = f'{data_args.e2e_train}/sonnet_test.txt'
            elif split == 'debug':
                print('loading form the DEBUG set')
                path = data_args.debug_path
                import json
                with open(path, 'r') as ff:
                    for line in ff:
                        sentence_lst.append(json.loads(line)[0].split(' '))
                sentence_lst = sentence_lst + sentence_lst
            if split in ['train', 'valid', 'test']:
                with open(path, 'r') as ff:
                    for row in ff:
                        word_lst = row[:-1]
                        word_lst = [x.text for x in tokenizer(word_lst)]
                        sentence_lst.append(word_lst)
            print(sentence_lst[:2])
        elif data_args.modality == 'e2e-tgt' and data_args.notes == 'ci':
            print('loading dataset from ci dataset')
            sentence_lst = []
            nlp = Chinese()
            tokenizer = nlp.tokenizer
            if split == 'train':
                print('loading form the TRAIN set')
                path = f'{data_args.e2e_train}/ci_train.txt'
            elif split == 'valid':
                print('loading form the VALID set')
                path = f'{data_args.e2e_train}/ci_valid.txt'
            elif split == 'test':
                print('loading form the TEST set')
                path = f'{data_args.e2e_train}/ci_valid.txt'
            elif split == 'debug':
                print('loading form the DEBUG set')
                path = data_args.debug_path
                import json
                with open(path, 'r') as ff:
                    for line in ff:
                        sentence_lst.append(json.loads(line)[0].split(' '))
                sentence_lst = sentence_lst + sentence_lst
            if split in ['train', 'valid', 'test']:
                with open(path, 'r') as ff:
                    for row in ff:
                        word_lst = row.split('|')[1][:-1]
                        word_lst = [x.text for x in tokenizer(word_lst)]
                        sentence_lst.append(word_lst)
            print(sentence_lst[:2])
        elif data_args.modality == 'e2e-tgt':
            print('loading dataset from simple e2e dataset')
            sentence_lst = []
            nlp = English()
            tokenizer = nlp.tokenizer
            if split == 'train':
                print('loading form the TRAIN set')
                path = f'{data_args.e2e_train}/src1_train.txt'
            elif split == 'valid':
                print('loading form the VALID set')
                path = f'{data_args.e2e_train}/src1_valid.txt'
            elif split == 'test':
                print('loading form the TEST set')
                path = f'{data_args.e2e_train}/src1_test.txt'
            elif split == 'debug':
                print('loading form the DEBUG set')
                path = data_args.debug_path
                import json
                with open(path, 'r') as ff:
                    for line in ff:
                        sentence_lst.append(json.loads(line)[0].split(' '))
                sentence_lst = sentence_lst + sentence_lst
            if split in ['train', 'valid', 'test']:
                with open(path, 'r') as ff:
                    for row in ff:
                        word_lst = row.split('||')[1]
                        word_lst = [x.text for x in tokenizer(word_lst)]
                        sentence_lst.append(word_lst)
            print(sentence_lst[:2])

        # get tokenizer.
        if load_vocab is None:
            counter = Counter()
            for input_ids in sentence_lst:
                counter.update(input_ids)

    if data_args.experiment_mode == 'conditional_gen':
        if data_args.modality == 'e2e':
            print('loading dataset from simple e2e dataset')
            sentence_lst = []
            nlp = English()
            tokenizer = nlp.tokenizer
            if split == 'train':
                path = f'{data_args.e2e_train}/src1_train.txt'
                with open(path, 'r') as ff:
                    for row in ff:
                        src_lst, word_lst = row.split('||')
                        word_lst = [x.text for x in tokenizer(word_lst)]
                        src_lst = [x.text for x in tokenizer(src_lst)]
                        sentence_lst.append((src_lst, word_lst))
            elif split == 'valid':
                path = f'{data_args.e2e_train}/src1_valid.txt'
                sentence_lst = read_e2e_files(path, data_args, tokenizer)
            print(sentence_lst[:2])
        # get tokenizer.
        if load_vocab is None:
            counter = Counter()
            for (src_ids, input_ids) in sentence_lst:
                counter.update(input_ids)
                counter.update(src_ids)

    if load_vocab is None:
        vocab_dict = {'START': 0, 'END': 1, 'UNK': 2, 'PAD': 3}
        for k, v in counter.items():
            if v > 10:
                vocab_dict[k] = len(vocab_dict)
        print(len(counter), len(vocab_dict))

        path_save_vocab = f'{data_args.checkpoint_path}/vocab.json'
        print(f'save the vocab to {path_save_vocab}')
        with open(path_save_vocab, 'w') as f:
            json.dump(vocab_dict, f)
    else:
        vocab_dict = load_vocab
        path_save_vocab = f'{data_args.checkpoint_path}/vocab.json'
        if not os.path.exists(path_save_vocab):
            print(f'save the vocab to {path_save_vocab}')
            if isinstance(vocab_dict, dict):
                with open(path_save_vocab, 'w') as f:
                    json.dump(vocab_dict, f)
                assert vocab_dict['START'] == 0
            elif isinstance(vocab_dict, PreTrainedTokenizerFast):
                vocab_dict.save_pretrained(data_args.checkpoint_path)
            else:
                assert False, "invalid type of vocab_dict"

    if model is None and data_args.experiment == 'random':
        model = torch.nn.Embedding(len(vocab_dict), data_args.in_channel)
        print('initializing the random embeddings', model)
        torch.nn.init.normal_(model.weight)
        path_save = f'{data_args.checkpoint_path}/random_emb.torch'
        print(f'save the random encoder to {data_args.checkpoint_path}/random_emb.torch')
        torch.save(model.state_dict(), path_save)

    path_save = f'{data_args.checkpoint_path}/random_emb.torch'
    if not os.path.exists(path_save) and data_args.experiment == 'random':
        torch.save(model.state_dict(), path_save)

    if data_args.experiment_mode == 'lm':
        result_train_lst = helper_tokenize_encode(
            sentence_lst, vocab_dict, model, image_size ** 2, data_args, padding_mode)
    return {'train': result_train_lst}, model


def write_e2e_corr(prompt_lst, file_dict, corr_path):
    print(len(prompt_lst))
    with open(corr_path, 'w') as f:
        for x in prompt_lst:
            for line in file_dict[x]:
                print(" ".join(line), file=f)
            print('', file=f)


def write_e2e_src(prompt_lst, corr_path):
    with open(corr_path, 'w') as f:
        for x in prompt_lst:
            print(" ".join(x), file=f)
    return


def read_e2e_files(path, args, tokenizer):
    file_dict = {}
    with open(path, 'r') as f:
        for line in f:
            src_lst, word_lst = line.strip().split('||')
            tgt = tuple([x.text for x in tokenizer(word_lst)])
            src = tuple([x.text for x in tokenizer(src_lst)])
            if src not in file_dict:
                file_dict[src] = []
            file_dict[src].append(tgt)
    temp = '1'
    prompt_text_dict = file_dict
    prompt_text_lst = list(prompt_text_dict.keys())
    gold_dir = os.path.join(args.out_dir, '{}_{}_{}'.format(temp, args.split, 'gold'))
    print("gold dir", gold_dir)
    write_e2e_corr(prompt_text_lst, prompt_text_dict, gold_dir)
    src_dir = os.path.join(args.out_dir, '{}_{}_{}'.format(temp, args.split, 'src'))
    write_e2e_src(prompt_text_lst, src_dir)
    final_lst = [(xx, prompt_text_dict[xx][0]) for xx in prompt_text_lst]
    return final_lst


class TextDataset(Dataset):
    def __init__(self, text_datasets, resolution, data_args, model_arch='conv-unet',
                 classes=None, shard=0, num_shards=1, eigen_transform=None,
                 mapping_func=None, model_emb=None):
        super().__init__()
        self.resolution = resolution
        self.text_datasets = text_datasets
        self.length = len(self.text_datasets['train'])
        self.model_arch = model_arch
        self.data_args = data_args
        print(self.resolution)
        self.eigen_transform = eigen_transform
        self.mapping_func = mapping_func
        self.model_emb = model_emb
        # self.local_images = image_paths[shard:][::num_shards]
        # self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        if self.model_arch == 'conv-unet':
            arr = np.array(self.text_datasets['train'][idx]['hidden_states'],
                           dtype=np.float32).reshape(self.resolution, self.resolution, -1)
            # print(self.eigen_transform.shape)
            if self.eigen_transform is not None:
                old_shape = arr.shape
                arr = arr.reshape(1, -1) - self.eigen_transform['mean']
                arr = arr @ self.eigen_transform['map']
                arr = arr.reshape(old_shape)
            if hasattr(self.data_args, 'noise_level') and self.data_args.noise_level > 0:
                arr = arr + self.data_args.noise_level * np.random.randn(*arr.shape).astype(arr.dtype)

            out_dict = {}
            out_dict['input_ids'] = np.array(self.text_datasets['train'][idx]['input_ids'])
            # if self.local_classes is not None:
            #     out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
            # print(out_dict.keys())
            return np.transpose(arr, [2, 0, 1]), out_dict
        elif self.model_arch == '1d-unet':
            arr = np.array(self.text_datasets['train'][idx]['hidden_states'],
                           dtype=np.float32)  # seqlen, dim
            if self.eigen_transform is not None:
                old_shape = arr.shape
                arr = arr.reshape(1, -1) - self.eigen_transform['mean']
                arr = arr @ self.eigen_transform['map']
                arr = arr.reshape(old_shape)
            if hasattr(self.data_args, 'noise_level') and self.data_args.noise_level > 0:
                arr = arr + self.data_args.noise_level * np.random.randn(*arr.shape).astype(arr.dtype)
            arr = np.transpose(arr, [1, 0])
            out_dict = {}
            out_dict['input_ids'] = np.array(self.text_datasets['train'][idx]['input_ids'])
            # out_dict['mapping_func'] = self.mapping_func
            # if self.local_classes is not None:
            #     out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
            # print(arr.shape)
            return arr, out_dict
        else:
            arr = np.array(self.text_datasets['train'][idx]['hidden_states'],
                           dtype=np.float32)
            if self.eigen_transform is not None:
                old_shape = arr.shape
                # arr = arr.reshape(1, -1) @ self.eigen_transform
                arr = arr.reshape(1, -1) - self.eigen_transform['mean']
                arr = arr @ self.eigen_transform['map']
                arr = arr.reshape(old_shape)

            if hasattr(self.data_args, 'noise_level') and self.data_args.noise_level > 0:
                # print(arr.dtype)
                # print(self.data_args.noise_level, 'using the noise level.')
                arr = arr + self.data_args.noise_level * np.random.randn(*arr.shape).astype(arr.dtype)
                # print(arr.dtype)

            out_dict = {}
            out_dict['input_ids'] = np.array(self.text_datasets['train'][idx]['input_ids'])
            # out_dict['mapping_func'] = self.mapping_func
            if self.data_args.experiment_mode == 'conditional_gen':
                out_dict['src_ids'] = np.array(self.text_datasets['train'][idx]['src_ids'])
                out_dict['src_mask'] = np.array(self.text_datasets['train'][idx]['src_mask'])
            # if self.local_classes is not None:
            #     out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
            return arr, out_dict
        # print(arr.dtype)
        # arr = arr.float()
        # print(arr.shape)


class TextDataset_NoCache(Dataset):
    def __init__(self, text_datasets, resolution, data_args, model_arch='conv-unet',
                 classes=None, shard=0, num_shards=1, eigen_transform=None,
                 mapping_func=None, model_emb=None):
        super().__init__()
        self.resolution = resolution
        self.text_datasets = text_datasets
        self.length = len(self.text_datasets['train'])
        self.model_arch = model_arch
        self.data_args = data_args
        print(self.resolution)
        self.eigen_transform = eigen_transform
        self.mapping_func = mapping_func
        self.model_emb = model_emb
        # self.local_images = image_paths[shard:][::num_shards]
        # self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        with torch.no_grad():
            input_ids = self.text_datasets['train'][idx]['input_ids']
            model = self.model_emb
            if self.data_args.experiment.startswith('random'):
                hidden_state = model(torch.tensor(input_ids))
            elif self.data_args.experiment == 'gpt2_pre_compress':
                input_ids2 = torch.tensor(input_ids).to(model.device)
                input_embs = model.transformer.wte(input_ids2)  # input_embs
                hidden_state = model.down_proj(input_embs)
                hidden_state = hidden_state * data_args.emb_scale_factor

            if self.model_arch == 'conv-unet':
                arr = np.array(hidden_state,
                               dtype=np.float32).reshape(self.resolution, self.resolution, -1)
                # print(self.eigen_transform.shape)
                if self.eigen_transform is not None:
                    old_shape = arr.shape
                    arr = arr.reshape(1, -1) - self.eigen_transform['mean']
                    arr = arr @ self.eigen_transform['map']
                    arr = arr.reshape(old_shape)
                if hasattr(self.data_args, 'noise_level') and self.data_args.noise_level > 0:
                    arr = arr + self.data_args.noise_level * np.random.randn(*arr.shape).astype(arr.dtype)

                out_dict = {}
                out_dict['input_ids'] = np.array(self.text_datasets['train'][idx]['input_ids'])
                # if self.local_classes is not None:
                #     out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
                # print(out_dict.keys())
                return np.transpose(arr, [2, 0, 1]), out_dict
            elif self.model_arch == '1d-unet':
                arr = np.array(hidden_state,
                               dtype=np.float32)  # seqlen, dim
                if self.eigen_transform is not None:
                    old_shape = arr.shape
                    arr = arr.reshape(1, -1) - self.eigen_transform['mean']
                    arr = arr @ self.eigen_transform['map']
                    arr = arr.reshape(old_shape)
                if hasattr(self.data_args, 'noise_level') and self.data_args.noise_level > 0:
                    arr = arr + self.data_args.noise_level * np.random.randn(*arr.shape).astype(arr.dtype)
                arr = np.transpose(arr, [1, 0])
                out_dict = {}
                out_dict['input_ids'] = np.array(self.text_datasets['train'][idx]['input_ids'])
                # out_dict['mapping_func'] = self.mapping_func
                # if self.local_classes is not None:
                #     out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
                # print(arr.shape)
                return arr, out_dict
            else:
                arr = np.array(hidden_state,
                               dtype=np.float32)
                if self.eigen_transform is not None:
                    old_shape = arr.shape
                    # arr = arr.reshape(1, -1) @ self.eigen_transform
                    arr = arr.reshape(1, -1) - self.eigen_transform['mean']
                    arr = arr @ self.eigen_transform['map']
                    arr = arr.reshape(old_shape)

                if hasattr(self.data_args, 'noise_level') and self.data_args.noise_level > 0:
                    # print(arr.dtype)
                    # print(self.data_args.noise_level, 'using the noise level.')
                    arr = arr + self.data_args.noise_level * np.random.randn(*arr.shape).astype(arr.dtype)
                    # print(arr.dtype)

                out_dict = {}
                out_dict['input_ids'] = np.array(self.text_datasets['train'][idx]['input_ids'])
                # out_dict['mapping_func'] = self.mapping_func
                if self.data_args.experiment_mode == 'conditional_gen':
                    out_dict['src_ids'] = np.array(self.text_datasets['train'][idx]['src_ids'])
                    out_dict['src_mask'] = np.array(self.text_datasets['train'][idx]['src_mask'])
                # if self.local_classes is not None:
                #     out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
                return arr, out_dict


def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False):
    result = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    mask_ = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result


def _torch_collate_batch(examples, pad_token_id, max_length):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    import numpy as np
    import torch

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    # length_of_first = examples[0].size(0)
    # Check if padding is necessary.
    # are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    # if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
    #     return torch.stack(examples, dim=0)
    # Creating the full tensor and filling it with our data.
    # max_length = max(x.size(0) for x in examples)
    # if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
    #     max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], pad_token_id)
    for i, example in enumerate(examples):
        if True:
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0]:] = example
    return result
