"""
build_local_model.py
--------------------
Создаёт МАЛЕНЬКУЮ sentence-transformer модель ЛОКАЛЬНО, без скачивания
с HuggingFace. Нужно ТОЛЬКО потому, что в текущей sandbox-среде нет
доступа к huggingface.co. На локальной машине пользователя этот скрипт
НЕ НУЖЕН — там просто скачается paraphrase-multilingual-MiniLM-L12-v2.

Эта мини-модель используется ИСКЛЮЧИТЕЛЬНО для демонстрации того,
что pipeline (обучение + метрики) работает end-to-end.
"""

import os
import json
import tempfile
from pathlib import Path

import torch
from transformers import (
    BertConfig, BertModel, PreTrainedTokenizerFast
)
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Transformer, Pooling


def build_tiny_multilingual_model(
    output_dir: str,
    vocab_texts: list[str],
    hidden_size: int = 128,
    num_layers: int = 2,
    num_heads: int = 4,
    vocab_size: int = 5000,
) -> str:
    """
    Собирает локальную мини-модель:
      - WordLevel tokenizer, обученный на vocab_texts
      - BERT-подобный transformer с нуля (со случайной инициализацией)
      - mean pooling
    Сохраняет в формате SentenceTransformer.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Тренируем простой WordLevel tokenizer ----
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )
    tokenizer.train_from_iterator(vocab_texts, trainer=trainer)

    tok_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tok_path))

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tok_path),
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )

    # ---- 2. Создаём mini-BERT с нуля ----
    config = BertConfig(
        vocab_size=hf_tokenizer.vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=hidden_size * 4,
        max_position_embeddings=512,
        pad_token_id=hf_tokenizer.pad_token_id,
    )
    bert = BertModel(config)

    # ---- 3. Сохраняем transformer-блок в формате, понятном sentence-transformers ----
    tf_dir = output_dir / "0_Transformer"
    tf_dir.mkdir(exist_ok=True)
    bert.save_pretrained(tf_dir)
    hf_tokenizer.save_pretrained(tf_dir)

    # sentence_bert_config.json для Transformer-модуля
    with open(tf_dir / "sentence_bert_config.json", "w") as f:
        json.dump({
            "max_seq_length": 128,
            "do_lower_case": False,
        }, f)

    # ---- 4. Pooling-модуль ----
    pool_dir = output_dir / "1_Pooling"
    pool_dir.mkdir(exist_ok=True)
    with open(pool_dir / "config.json", "w") as f:
        json.dump({
            "word_embedding_dimension": hidden_size,
            "pooling_mode_cls_token": False,
            "pooling_mode_mean_tokens": True,
            "pooling_mode_max_tokens": False,
            "pooling_mode_mean_sqrt_len_tokens": False,
            "pooling_mode_weightedmean_tokens": False,
            "pooling_mode_lasttoken": False,
            "include_prompt": True,
        }, f)

    # ---- 5. modules.json ----
    with open(output_dir / "modules.json", "w") as f:
        json.dump([
            {"idx": 0, "name": "0", "path": "0_Transformer",
             "type": "sentence_transformers.models.Transformer"},
            {"idx": 1, "name": "1", "path": "1_Pooling",
             "type": "sentence_transformers.models.Pooling"},
        ], f)

    # ---- 6. config_sentence_transformers.json ----
    with open(output_dir / "config_sentence_transformers.json", "w") as f:
        json.dump({
            "__version__": {
                "sentence_transformers": "2.7.0",
                "transformers": "4.40.0",
                "pytorch": str(torch.__version__),
            }
        }, f)

    return str(output_dir)


if __name__ == "__main__":
    # Простой smoke test
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.run_demo import build_synthetic_dataset

    ds = build_synthetic_dataset()
    texts = (
        ds.occupations["title"].tolist() +
        ds.occupations["description"].tolist() +
        ds.skills["title"].tolist() +
        ds.skills["description"].tolist()
    )
    out = build_tiny_multilingual_model("/tmp/tiny_model_test", texts)
    print(f"Built model at {out}")
    m = SentenceTransformer(out)
    print(f"Loaded! Embedding shape:", m.encode(["test"]).shape)
