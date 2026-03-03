"""Tests for DNA modality constants and bit-packing."""
import pytest


def test_modality_constants_exist():
    from project.config import (
        MODALITY_NEUTRAL, MODALITY_VISUAL,
        MODALITY_AUDIO_LEFT, MODALITY_AUDIO_RIGHT,
        MODALITY_SHIFT, MODALITY_MASK,
    )
    assert MODALITY_NEUTRAL == 0
    assert MODALITY_VISUAL == 1
    assert MODALITY_AUDIO_LEFT == 2
    assert MODALITY_AUDIO_RIGHT == 3
    assert MODALITY_SHIFT == 0
    assert MODALITY_MASK == 0b111


def test_modality_mask_extracts_bits():
    from project.config import MODALITY_VISUAL, MODALITY_SHIFT, MODALITY_MASK
    # Simulate packing: VISUAL=1 in bits 2-0
    packed_state = MODALITY_VISUAL << MODALITY_SHIFT
    extracted = (packed_state >> MODALITY_SHIFT) & MODALITY_MASK
    assert extracted == MODALITY_VISUAL


def test_modality_does_not_overlap_dna():
    """Bits 2-0 must not overlap DNA range bits 57-18."""
    from project.config import MODALITY_SHIFT, MODALITY_MASK, BINARY_DNA_BASE_SHIFT
    modality_top_bit = MODALITY_SHIFT + MODALITY_MASK.bit_length() - 1
    assert modality_top_bit < BINARY_DNA_BASE_SHIFT  # bits 2-0 are below bit 18
