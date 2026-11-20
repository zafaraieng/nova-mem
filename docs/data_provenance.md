# Dataset provenance and handling

NOVA-Mem is designed to use the cleaned LongMemEval benchmark. The benchmark
files are **not redistributed** in this repository. Obtain them from the
upstream dataset project and review its current terms before use:

- Dataset: <https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned>
- Upstream project and official code: <https://github.com/xiaowu0162/LongMemEval>

Run `python scripts/prepare_data.py` from the repository root to retrieve the
configured public benchmark inputs and perform the checksum validation
implemented by the script. Keep downloaded files, derived splits, caches,
indexes, and experiment outputs local. The root `.gitignore` excludes the
root-level data and run-output directories; do not add restricted or
non-redistributable data to Git.

The release contains checksum sidecars in `data-templates/` and hash values in
the experiment configuration. These are integrity references, not copies of
the benchmark data. This publication audit did not download the benchmark
files, so it does not independently attest to the current contents of any
local dataset installation.

No raw dataset, private conversation collection, annotation mapping, or
unblinding key is included in this repository.
