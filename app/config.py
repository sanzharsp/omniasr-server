"""
Configuration constants for Omnilingual-ASR server.
"""

import os


# Available models:
# - omniASR_CTC_{300M,1B,3B,7B}_v2: Fast parallel CTC generation
# - omniASR_LLM_{300M,1B,3B,7B}_v2: Language-conditioned autoregressive
# - omniASR_LLM_Unlimited_{300M,1B,3B,7B}_v2: Unlimited audio length
MODEL_NAME = os.getenv("MODEL_NAME", "omniASR_CTC_300M_v2")

# Optional custom download URLs — useful for airgap / self-hosted S3 environments.
# When set, these override the default fairseq2 registry URLs for the model
# and tokenizer assets respectively.
MODEL_CHECKPOINT_URL = os.getenv("MODEL_CHECKPOINT_URL")
MODEL_TOKENIZER_URL = os.getenv("MODEL_TOKENIZER_URL")
