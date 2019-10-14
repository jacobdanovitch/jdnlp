import warnings
import multiprocessing as mp
from tqdm.auto import tqdm


def tqdm_iparallel(fn, vals, processes):
    warnings.warn("tqdm_iparallel is slower than tqdm_parallel. Please use the latter for large datasets.")
    with mp.Pool(processes=processes) as pool, tqdm(total=len(vals)) if isinstance(vals, list) else tqdm() as pbar:
        for x in pool.imap(fn, vals):
            pbar.update()
            yield x


def tqdm_parallel(fn, vals, processes):
    with mp.Pool(processes=processes) as pool:
        return pool.map(fn, tqdm(vals))