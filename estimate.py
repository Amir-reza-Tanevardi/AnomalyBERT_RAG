import os, json
import numpy as np
import torch
import argparse
import faiss  # for nearest‐neighbor retrieval

import utils.config as config


import os
import numpy as np
import faiss

def build_or_load_index(
    train_data,
    model,
    post_activation,
    window_size,
    window_sliding,
    device,
    dataset_name,
    cache_dir='cache',
    K=10
):
    """
    Builds (or loads) a FAISS index, the per-window scores, AND the raw embeddings.
    Returns: (index, history_scores, history_embeddings)
    """
    import os, numpy as np, torch, faiss
    os.makedirs(cache_dir, exist_ok=True)
    idx_path        = os.path.join(cache_dir, f"{dataset_name}.index")
    scores_path     = os.path.join(cache_dir, f"{dataset_name}_scores.npy")
    emb_path        = os.path.join(cache_dir, f"{dataset_name}_embeddings.npy")

    # If all three exist, load and return
    if os.path.exists(idx_path) and os.path.exists(scores_path) and os.path.exists(emb_path):
        index              = faiss.read_index(idx_path)
        history_scores     = np.load(scores_path)
        history_embeddings = np.load(emb_path)
        return index, history_scores, history_embeddings

    # Otherwise build afresh
    train_latents = []
    train_scores  = []
    model.to(device)
    print(len(train_data) - window_size + 1)
    with torch.no_grad():
        for i in range(0, len(train_data) - window_size + 1, window_sliding):
            print(i)
            x_win    = torch.Tensor(train_data[i : i + window_size]) \
                             .unsqueeze(0).to(device)
            h        = model.encode(x_win)               # (1, latent_dim)
            y_logits = model(x_win).squeeze(-1)           # (1, window_size)
            y_score  = post_activation(y_logits)         # (1, window_size)
            scalar   = y_score.mean().item()             # one number

            train_latents.append(h.view(-1).cpu().numpy())
            train_scores.append(scalar)

    # Stack into arrays
    history_embeddings = np.vstack(train_latents).astype('float32')
    history_scores     = np.array(train_scores, dtype='float32')

    # Build FAISS index (cosine-sim via IP on normalized vectors)
    d     = history_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(history_embeddings)
    index.add(history_embeddings)

    # Cache everything
    faiss.write_index(index, idx_path)
    np.save(scores_path, history_scores)
    np.save(emb_path, history_embeddings)

    return index, history_scores, history_embeddings



def estimate(test_data, train_data, model, post_activation, out_dim, batch_size, window_sliding, divisions, check_count,dataset, device):

    """
    Estimate both:
      - output_values: per‐timestamp raw anomaly score (sigmoid of model logits)
      - ra_scores: retrieval‐augmented anomaly score per timestamp
    """
    window_size = model.max_seq_len * model.patch_size
    assert window_size % window_sliding == 0

    # Construct FAISS index and retrieve training latents/scores
    index, history_scores , history_embeddings = build_or_load_index(
        train_data=train_data,
        model=model,
        post_activation=post_activation,
        window_size=model.max_seq_len * model.patch_size,
        window_sliding=window_sliding,
        device=device,
        dataset_name=dataset,    # e.g. 'MSL' or 'SWaT'
        cache_dir='/content/AnomalyBERT_RAG/index',       # where to store .index and _scores.npy
        K=10                     # number of neighbors
      )

    print("Done makng the index.")

    n_column = out_dim
    n_batch = batch_size
    batch_sliding = n_batch * window_size
    _batch_sliding = n_batch * window_sliding

    # prepare accumulators
    output_values = torch.zeros(len(test_data), n_column, device=device)
    rag_values     = torch.zeros(len(test_data) , device=device)
    count = 0
    checked_index = np.inf if check_count is None else check_count

    # Record output values.
    # Record output values.
    # Record output values.
    for division in divisions:
        data_len = division[1] - division[0]
        last_window = data_len - window_size + 1
        _test_data = test_data[division[0]:division[1]]
        _output_values = torch.zeros(data_len, n_column, device=device)
        _latent_values = torch.zeros(data_len, 512, device=device)  # added
        n_overlap = torch.zeros(data_len, device=device)

        with torch.no_grad():
            _first = -batch_sliding
            for first in range(0, last_window - batch_sliding + 1, batch_sliding):
                for i in range(first, first + window_size, window_sliding):
                    x = torch.Tensor(_test_data[i:i + batch_sliding].copy()).reshape(n_batch, window_size, -1).to(device)

                    # Eval and record errors
                    y = post_activation(model(x))
                    _output_values[i:i + batch_sliding] += y.view(-1, n_column)

                    # Also collect latent representations
                    h = model.encode(x)  # (batch, latent_dim)
                    _latent_values[i:i + batch_sliding] += h.view(-1, 512)  # added

                    n_overlap[i:i + batch_sliding] += 1
                    count += n_batch
                    if count > checked_index:
                        print(count, 'windows are computed.')
                        checked_index += check_count

            _first += batch_sliding
            for first, last in zip(range(_first, last_window, _batch_sliding),
                                  list(range(_first + _batch_sliding, last_window, _batch_sliding)) + [last_window]):
                x = []
                for i in list(range(first, last - 1, window_sliding)) + [last - 1]:
                    x.append(torch.Tensor(_test_data[i:i + window_size].copy()))
                x = torch.stack(x).to(device)

                y = post_activation(model(x))
                h = model.encode(x)  # added

                for i, j in enumerate(list(range(first, last - 1, window_sliding)) + [last - 1]):
                    _output_values[j:j + window_size] += y[i]
                    _latent_values[j:j + window_size] += h[i].unsqueeze(0).repeat(window_size, 1)  # added
                    n_overlap[j:j + window_size] += 1

                count += n_batch
                if count > checked_index:
                    print(count, 'windows are computed.')
                    checked_index += check_count

            _output_values = _output_values / n_overlap.unsqueeze(-1)
            _latent_values = _latent_values / n_overlap.unsqueeze(-1)  # added

            # Record values
            output_values[division[0]:division[1]] = _output_values

            # === Compute RAG values ===
            lat_np = _latent_values.cpu().numpy().astype('float32')
            faiss.normalize_L2(lat_np)
            D, I = index.search(lat_np, k=100)  # Retrieve k-NN IDs

            nbr_scores = history_scores[I]  # shape (data_len, k)
            local_mean = torch.from_numpy(nbr_scores.mean(axis=1)).to(device)
            local_std = torch.from_numpy(nbr_scores.std(axis=1)).to(device)
            raw_ts = _output_values.squeeze(-1)  # shape (data_len,)
            rag_values[division[0]:division[1]] = (raw_ts - 10*local_mean)


    return output_values, rag_values


def main(options):
    # Load test data.
    test_data = np.load(config.TEST_DATASET[options.dataset]).copy().astype(np.float32)
    
    # Ignore the specific columns.
    if options.dataset in config.IGNORED_COLUMNS.keys():
        ignored_column = np.array(config.IGNORED_COLUMNS[options.dataset])
        remaining_column = [col for col in range(len(test_data[0])) if col not in ignored_column]
        test_data = test_data[:, remaining_column]
    
    # Load model.
    device = torch.device('cuda:{}'.format(options.gpu_id))
    model = torch.load(options.model, map_location=device)
    if options.state_dict != None:
        model.load_state_dict(torch.load(options.state_dict, map_location='cpu'))
    model.eval()
    
    # Data division
    data_division = config.DEFAULT_DIVISION[options.dataset] if options.data_division == None else options.data_division 
    if data_division == 'total':
        divisions = [[0, len(test_data)]]
    else:
        with open(config.DATA_DIVISION[options.dataset][data_division], 'r') as f:
            divisions = json.load(f)
        if isinstance(divisions, dict):
            divisions = divisions.values()
            
    n_column = len(test_data[0]) if options.reconstruction_output else 1
    post_activation = torch.nn.Identity().to(device) if options.reconstruction_output\
                      else torch.nn.Sigmoid().to(device)
            
    # Estimate scores.
    output_values = estimate(test_data, model, post_activation, n_column, options.batch_size,
                             options.window_sliding, divisions, options.check_count, device)
    
    # Save results.
    output_values = output_values.cpu().numpy()
    outfile = options.state_dict[:-3] + '_results.npy' if options.outfile == None else options.outfile
    np.save(outfile, output_values)
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--dataset", default='SMAP', type=str, help='SMAP/MSL/SMD/SWaT/WADI')
    
    parser.add_argument("--model", required=True, type=str, help='model file (.pt) to estimate')
    parser.add_argument("--state_dict", default=None, type=str, help='state dict file (.pt) to estimate')
    parser.add_argument("--outfile", default=None, type=str, help='output file name (.npy) to save anomaly scores')
    
    parser.add_argument("--data_division", default=None, type=str, help='data division; None(defualt)/channel/class/total')
    parser.add_argument("--check_count", default=5000, type=int, help='check count of window computing')
    
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--window_sliding", default=16, type=int, help='sliding steps of windows; window size should be divisible by this value')
    parser.add_argument('--reconstruction_output', default=False, action='store_true', help='option for reconstruction model (deprecated)')
    
    options = parser.parse_args()
    main(options)
