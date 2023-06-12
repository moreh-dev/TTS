import os
from dataclasses import dataclass, field
from typing import Dict

from TTS.tts.configs.shared_configs import BaseTTSConfig
from TTS.tts.layers.bark.model import GPTConfig
from TTS.tts.layers.bark.model_fine import FineGPTConfig
from TTS.utils.generic_utils import get_user_data_dir


@dataclass
class BarkConfig(BaseTTSConfig):
    num_chars: int = 0
    semantic_config: GPTConfig = GPTConfig()
    fine_config: FineGPTConfig = FineGPTConfig()
    coarse_config: GPTConfig = GPTConfig()
    CONTEXT_WINDOW_SIZE: int = 1024
    SEMANTIC_RATE_HZ: float = 49.9
    SEMANTIC_VOCAB_SIZE: int = 10_000
    CODEBOOK_SIZE: int = 1024
    N_COARSE_CODEBOOKS: int = 2
    N_FINE_CODEBOOKS: int = 8
    COARSE_RATE_HZ: int = 75
    SAMPLE_RATE: int = 24_000
    USE_SMALLER_MODELS: bool = False

    TEXT_ENCODING_OFFSET: int = 10_048
    SEMANTIC_PAD_TOKEN: int = 10_000
    TEXT_PAD_TOKEN: int = 129_595
    SEMANTIC_INFER_TOKEN: int = 129_599
    COARSE_SEMANTIC_PAD_TOKEN: int = 12_048
    COARSE_INFER_TOKEN: int = 12_050

    REMOTE_BASE_URL = "https://dl.suno-models.io/bark/models/v0/"
    REMOTE_MODEL_PATHS: Dict = None
    LOCAL_MODEL_PATHS: Dict = None
    SMALL_REMOTE_MODEL_PATHS: Dict = None
    CACHE_DIR: str = str(get_user_data_dir("tts/suno/bark_v0"))

    def __post_init__(self):
        self.REMOTE_MODEL_PATHS = {
            "text": {
                "path": os.path.join(self.REMOTE_BASE_URL, "text_2.pt"),
                "checksum": "54afa89d65e318d4f5f80e8e8799026a",
            },
            "coarse": {
                "path": os.path.join(self.REMOTE_BASE_URL, "coarse_2.pt"),
                "checksum": "8a98094e5e3a255a5c9c0ab7efe8fd28",
            },
            "fine": {
                "path": os.path.join(self.REMOTE_BASE_URL, "fine_2.pt"),
                "checksum": "59d184ed44e3650774a2f0503a48a97b",
            },
        }
        self.LOCAL_MODEL_PATHS = {
            "text": os.path.join(self.CACHE_DIR, "text_2.pt"),
            "coarse": os.path.join(self.CACHE_DIR, "coarse_2.pt"),
            "fine": os.path.join(self.CACHE_DIR, "fine_2.pt"),
            "hubert_tokenizer": os.path.join(self.CACHE_DIR, "tokenizer.pth"),
            "hubert": os.path.join(self.CACHE_DIR, "hubert.pt"),
        }
        self.SMALL_REMOTE_MODEL_PATHS = {
            "text": {"path": os.path.join(self.REMOTE_BASE_URL, "text.pt")},
            "coarse": {"path": os.path.join(self.REMOTE_BASE_URL, "coarse.pt")},
            "fine": {"path": os.path.join(self.REMOTE_BASE_URL, "fine.pt")},
        }
        self.sample_rate = self.SAMPLE_RATE
