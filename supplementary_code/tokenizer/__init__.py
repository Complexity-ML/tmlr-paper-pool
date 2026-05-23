"""Tokenizer loader used by local training scripts."""

from __future__ import annotations


class Tokenizer:
    """Thin wrapper around a local Hugging Face tokenizer directory."""

    @classmethod
    def load(cls, path: str):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
