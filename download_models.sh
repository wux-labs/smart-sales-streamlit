#!/bin/bash

HF_ENDPOINT=https://hf-mirror.com huggingface-cli download --resume-download --local-dir-use-symlinks False GanymedeNil/text2vec-large-chinese --local-dir models/GanymedeNil/text2vec-large-chinese
