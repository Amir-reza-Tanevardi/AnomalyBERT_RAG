import os, json
import numpy as np
import torch
import argparse
import faiss  # for nearest‐neighbor retrieval

import utils.config as config


import os
import numpy as np
import faiss

def build_or_load_index(train_data, model, post_activation,
                        window_size, window_sliding, device,
                        dataset_name, cache_dir='cache', K=10):
    """
    Builds a FAISS index + history_scores if not cached, else loads them.
    Returns (index, history_scores).
    """
    os.makedirs(cache_dir, exist_ok=True)
    idx_path   = os.path.join(cache_dir, f"{dataset_name}.index")
    scores_path= os.path.join(cache_dir, f"{dataset_name}_scores.npy")

    if os.path.exists(idx_path) and os.path.exists(scores_path):
        # Load from disk
        index = faiss.read_index(idx_path)
        history_scores = np.load(scores_path)
        return index, history_scores

    # Otherwise build fresh
    train_latents = []
    train_scores  = []
    print(f"Here: {len(train_data) - window_size + 1}")
    with torch.no_grad():
        for i in range(0, len(train_data) - window_size + 1, window_sliding):
            print(i)
            x_win      = torch.Tensor(train_data[i:i + window_size]) \
                              .unsqueeze(0).to(device)
            h          = model.encode(x_win)          # (1, latent_dim)
            y_logits   = model(x_win).squeeze(-1)     # (1, window_size)
            y_score    = post_activation(y_logits)    # (1, window_size)
            scalar_score = y_score.mean().item()      # one number per window

            train_latents.append(h.view(-1).cpu().numpy())
            train_scores.append(scalar_score)

    history_embeddings = np.vstack(train_latents).astype('float32')
    history_scores     = np.array(train_scores, dtype='float32')

    # Build FAISS index (cosine)
    d = history_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(history_embeddings)
    index.add(history_embeddings)

    # Cache to disk
    faiss.write_index(index, idx_path)
    np.save(scores_path, history_scores)

    return index, history_scores


def estimate(test_data, train_data, model, post_activation, out_dim, batch_size, window_sliding, divisions, check_count, device):

    """
    Estimate both:
      - output_values: per‐timestamp raw anomaly score (sigmoid of model logits)
      - ra_scores: retrieval‐augmented anomaly score per timestamp
    """
    window_size = model.max_seq_len * model.patch_size
    assert window_size % window_sliding == 0

    # Construct FAISS index and retrieve training latents/scores
    index, history_scores = build_or_load_index(
        train_data=train_data,
        model=model,
        post_activation=post_activation,
        window_size=model.max_seq_len * model.patch_size,
        window_sliding=window_sliding,
        device=device,
        dataset_name="MSL",    # e.g. 'MSL' or 'SWaT'
        cache_dir='/content/AnomalyBERT/index',       # where to store .index and _scores.npy
        K=10                     # number of neighbors
      )

    print("Done makng the index.")

    n_column = out_dim
    n_batch = batch_size
    batch_sliding = n_batch * window_size
    _batch_sliding = n_batch * window_sliding

    # prepare accumulators
    output_values = torch.zeros(len(test_data), n_column, device=device)
    ra_values     = torch.zeros(len(test_data),       device=device)
    count = 0
    checked_index = np.inf if check_count is None else check_count

    for division in divisions:
        start, end = division
        data_len = end - start
        last_window = data_len - window_size + 1
        _test = test_data[start:end]
        _out = torch.zeros(data_len, n_column, device=device)
        _latent = torch.zeros(data_len, 512, device=device)
        _n_overlap = torch.zeros(data_len, device=device)

        with torch.no_grad():
            # sliding over full batches
            for first in range(0, last_window - batch_sliding + 1, batch_sliding):
                for i in range(first, first + window_size, window_sliding):
                    # get batch of windows
                    x = torch.Tensor(_test[i:i + batch_sliding].copy()) \
                             .reshape(n_batch, window_size, -1).to(device)

                    print(x.shape)
                    # get latent representations and raw anomaly scores
                    #h = model.encode(x)                     # (batch, latent_dim)
                    y_logits = model(x).squeeze(-1)         # (batch, window_size)
                    y_scores = post_activation(y_logits)    # sigmoid if BCE

                    # flatten to timestamp‐wise
                    h_flat = h.view(-1,512)
                    y_flat = y_scores.view(-1, 1)

                    _latent[i:i + batch_sliding]    += h_flat
                    _out[i:i + batch_sliding]       += y_flat
                    _n_overlap[i:i + batch_sliding] += 1

                    count += n_batch
                    if count > checked_index:
                        print(count, 'windows processed.')
                        checked_index += check_count

            # sliding over remainder
            tail_start = ((last_window - batch_sliding) // _batch_sliding + 1) * _batch_sliding
            for first in range(tail_start, last_window, _batch_sliding):
                last = min(first + _batch_sliding, last_window)
                xs = []
                positions = list(range(first, last - 1, window_sliding)) + [last - 1]
                for i in positions:
                    xs.append(torch.Tensor(_test[i:i + window_size].copy()))
                x = torch.stack(xs).to(device)

                h = model.encode(x)
                y_logits = model(x).squeeze(-1)
                y_scores = post_activation(y_logits)

                for idx, i in enumerate(positions):
                    h_i = h[idx]
                    y_i = y_scores[idx]  # (window_size,)

                    _latent[i:i + window_size]    += h_i.unsqueeze(0).repeat(window_size, 1)
                    _out[i:i + window_size]       += y_i.unsqueeze(1)
                    _n_overlap[i:i + window_size] += 1

                count += n_batch
                if count > checked_index:
                    print(count, 'windows processed.')
                    checked_index += check_count

            # average over overlaps
            _latent = _latent / _n_overlap.unsqueeze(-1)
            _out    = _out    / _n_overlap.unsqueeze(-1)

            # store raw per‐timestamp scores
            output_values[start:end] = _out

            # build RAG scores per timestamp
            lat_np = _latent.cpu().numpy().astype('float32')
            faiss.normalize_L2(lat_np)
            D, I = index.search(lat_np, k=K_NEIGHBORS)  # retrieve neighbor IDs

            # lookup neighbor raw scores and compute local mean
            nbr_scores = history_scores[I]               # shape (data_len, K_NEIGHBORS)
            local_mean = torch.from_numpy(nbr_scores.mean(axis=1)).to(device)

            # raw timestamp‐wise score is _out.squeeze(-1)
            raw_ts = _out.squeeze(-1)                    # shape (data_len,)
            ra_values[start:end] = raw_ts / (local_mean + 1e-8)

    return output_values, ra_values


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
