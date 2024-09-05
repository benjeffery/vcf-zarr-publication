import numpy as np
import pandas as pd
import tensorstore as ts

import numba

@numba.jit(nopython=True)
def process_chunk(chunk, variant_chunk_len, sample_chunk_size, ref_counts, het_counts, hom_alt_counts):
    for i in range(variant_chunk_len):
        chunk_ref_count = 0
        chunk_het_count = 0
        chunk_hom_alt_count = 0
        
        for j in range(sample_chunk_size):
            a = chunk[i, j, 0]
            b = chunk[i, j, 1]
            
            if a == 0:
                chunk_ref_count += 1
            if b == 0:
                chunk_ref_count += 1
            if a != b:
                chunk_het_count += 1
            if a == b and a > 0:
                chunk_hom_alt_count += 1
        
        ref_counts[i] += chunk_ref_count
        het_counts[i] += chunk_het_count
        hom_alt_counts[i] += chunk_hom_alt_count

def ts_afdist(path):
    path = str(path)
    store = ts.open({
            "driver": "zarr",
            "kvstore": {
                "driver": "file",
                "path": path,
            },
    }, write=False).result()
    variant_count = store.shape[0]
    sample_count = store.shape[1]
    chunk_shape = store.chunk_layout.read_chunk.shape
    variant_chunk_size = chunk_shape[0]
    sample_chunk_size = chunk_shape[1]
    bin_counts = np.zeros((11,), dtype=np.int64)

    ref_counts = np.zeros((variant_chunk_size,), dtype=np.int64)
    het_counts = np.zeros((variant_chunk_size,), dtype=np.int64)
    hom_alt_counts = np.zeros((variant_chunk_size,), dtype=np.int64)

    for variant_chunk_start in range(0, variant_count, variant_chunk_size):
        variant_chunk_end = min(variant_count, variant_chunk_start + variant_chunk_size)
        variant_chunk_len = variant_chunk_end - variant_chunk_start
        ref_counts[:] = 0
        het_counts[:] = 0
        hom_alt_counts[:] = 0

        for sample_chunk_start in range(0, sample_count, sample_chunk_size):
            sample_chunk_end = min(sample_count, sample_chunk_start + sample_chunk_size)
            chunk = store[variant_chunk_start:variant_chunk_end, sample_chunk_start:sample_chunk_end].read().result()
            process_chunk(chunk, variant_chunk_len, sample_chunk_end - sample_chunk_start, ref_counts, het_counts, hom_alt_counts)

        alt_count = 2 * sample_count - ref_counts[:variant_chunk_len]
        alt_freq = alt_count / (2 * sample_count)
        het_ref_freq = 2 * alt_freq * (1 - alt_freq)
        hom_alt_freq = alt_freq * alt_freq

        bins = np.linspace(0, 1.0, len(bin_counts))
        bins[-1] += 0.0125
        a = np.bincount(np.digitize(het_ref_freq, bins), weights=het_counts[:variant_chunk_len], minlength=len(bins)).astype(np.int64)
        b = np.bincount(np.digitize(hom_alt_freq, bins), weights=hom_alt_counts[:variant_chunk_len], minlength=len(bins)).astype(np.int64)
        np.add(bin_counts, a, out=bin_counts)
        np.add(bin_counts, b, out=bin_counts)

    return pd.DataFrame({"start": bins[:-1], "stop": bins[1:], "prob dist": bin_counts[1:]})
