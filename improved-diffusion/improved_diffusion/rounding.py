import torch
from transformers import AutoTokenizer
import yaml


def load_models(modality, mode, model_name_or_path, emb_dim, file, extra_args=None):

    if mode in ['random', 'random1', 'random_up_proj', 'glove']:
        import json
        if modality == 'book' or (extra_args is not None and extra_args.use_bert_tokenizer == 'yes'):
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            if 'e2e' in file and modality == 'book':
                emb_dim = 1
        else:
            path_save_tokenizer = '{}/vocab.json'.format(file)
            print(f'loading from {path_save_tokenizer}')
            with open(path_save_tokenizer, 'r') as f:
                vocab = json.load(f)
            print(len(vocab))
            tokenizer = {v: k for k, v in vocab.items()}
        model = torch.nn.Embedding(len(tokenizer), emb_dim)
        path_save = '{}/random_emb.torch'.format(file)
        model.load_state_dict(torch.load(path_save))

    return model, tokenizer


def load_tokenizer(modality, mode, model_name_or_path):
    if mode in ['random', 'random_up_proj', 'glove']:
        import json
        path_save_tokenizer = '{}/vocab.json'.format(model_name_or_path)
        with open(path_save_tokenizer, 'r') as f:
            vocab = json.load(f)
        tokenizer = {v: k for k, v in vocab.items()}
    return tokenizer


def rounding_func(mode, text_emb_lst, model, tokenizer, emb_scale_factor=1.0):
    decoded_out_lst = []
    if mode in ['random', 'random_up_proj', 'glove']:
        down_proj_emb = model.weight  # input_embs
        down_proj_emb2 = None

        def get_knn(down_proj_emb, text_emb, dist='cos'):

            if dist == 'cos':
                adjacency = down_proj_emb @ text_emb.transpose(1, 0).to(down_proj_emb.device)
            elif dist == 'l2':
                adjacency = down_proj_emb.unsqueeze(1).expand(-1, text_emb.size(0), -1) - text_emb.unsqueeze(0).expand(
                    down_proj_emb.size(0), -1, -1)
                adjacency = -torch.norm(adjacency, dim=-1)
            topk_out = torch.topk(adjacency, k=6, dim=0)
            return topk_out.values, topk_out.indices

        dist = 'l2'
        # print(npzfile['arr_0'].shape)
        for text_emb in text_emb_lst:
            import torch
            text_emb = torch.tensor(text_emb)
            # print(text_emb.shape)
            if len(text_emb.shape) > 2:
                text_emb = text_emb.view(-1, text_emb.size(-1))
            else:
                text_emb = text_emb
            val, indices = get_knn((down_proj_emb2 if dist == 'cos' else down_proj_emb),
                                   text_emb.to(down_proj_emb.device), dist=dist)
            # generated_lst.append(tuple(indices[0].tolist()))

            # print(indices[0].tolist())
            # for i in range(64):
            #     print([tokenizer[x.item()] for x in indices[:,i]])
            decoded_out = " ".join([tokenizer[i] for i in indices[0].tolist()])
            decoded_out_lst.append(decoded_out)

    return decoded_out_lst

