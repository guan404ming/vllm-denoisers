from types import SimpleNamespace

import pytest
import torch
from PIL import Image

pytest.importorskip("vllm")

from denoisers.models.llada2.config import LLaDA2MoeConfig
from denoisers.models.llada2.uni import UniModelState, UniProcessor
from denoisers.models.nemotron_diffusion.vlm import NemotronVLMProcessingInfo, NemotronVLMProcessor
from denoisers.policies.decode import registry


def test_llada_uni_keeps_text_checkpoints_on_the_text_model():
    architecture = ["LLaDA2MoeModelLM"]
    assert LLaDA2MoeConfig(architectures=architecture).architectures == architecture
    uni = LLaDA2MoeConfig(architectures=architecture, image_token_offset=157184)
    assert uni.architectures == ["LLaDA2UniForImageUnderstanding"]


def test_nemotron_image_rows_keep_text_delimiters():
    config = SimpleNamespace(vision_config=SimpleNamespace(patch_size=14), spatial_merge_size=2)
    info = SimpleNamespace(get_hf_config=lambda: config)
    info.image_encoder = lambda: NemotronVLMProcessingInfo.image_encoder(info)
    processor = SimpleNamespace(info=info)
    data = NemotronVLMProcessor._call_hf_processor(
        processor, {"images": [Image.new("RGB", (80, 35))]}, {}
    )
    pixels = data["pixel_values"][0]
    assert pixels.shape == (3, 56, 84)
    kwargs = {"image": [{"pixel_values": SimpleNamespace(data=pixels)}]}
    (update,) = NemotronVLMProcessor._get_prompt_updates(processor, None, {}, kwargs)
    result = update.replacement(0)
    assert result.full == [18, 19, 19, 19, 20, 19, 19, 19, 21]
    assert result.is_embed(result.full).sum().item() == 6


def test_uni_image_headers_match_the_patch_grid():
    data = UniProcessor._call_hf_processor(None, {"images": [Image.new("RGB", (192, 128))]}, {})
    grid = data["image_grid_thw"][0]
    assert grid.tolist() == [1, 8, 12]
    assert data["pixel_values"].shape == (96, 768)
    vocabulary = {
        "<|image|>": [1],
        "<|image|><|reserved_token_8|><|reserved_token_12|><boi>": [1, 2, 3, 4],
        "<|/image|>": [5],
    }
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: vocabulary[text])
    info = SimpleNamespace(
        get_tokenizer=lambda: tokenizer,
        get_hf_config=lambda: SimpleNamespace(image_token_offset=157184),
    )
    kwargs = {"image": [{"image_grid_thw": SimpleNamespace(data=grid)}]}
    (update,) = UniProcessor._get_prompt_updates(SimpleNamespace(info=info), None, {}, kwargs)
    result = update.replacement(0)
    assert result.full == [1, 2, 3, 4] + [157184] * 96 + [5]
    assert result.is_embed(result.full).sum().item() == 96


def test_uni_keeps_image_codes_when_the_prompt_tail_is_decoded_again():
    state = SimpleNamespace(
        device="cpu",
        supports_mm_inputs=True,
        execute_mm_encoder=lambda inputs: None,
        gather_mm_embeddings=lambda batch: (
            [torch.tensor([[157190], [157202]])],
            torch.tensor([False, True, True, False]),
        ),
    )
    batch = SimpleNamespace(
        num_reqs=2,
        num_tokens=4,
        input_ids=torch.tensor([7, 157184, 157184, 8], dtype=torch.int32),
        idx_mapping=torch.tensor([1, 0]),
        query_start_loc=torch.tensor([0, 3, 4]),
        positions=torch.tensor([5, 6, 7, 0]),
    )
    saved = torch.zeros(2, 8, dtype=torch.int32)
    requests = SimpleNamespace(all_token_ids=SimpleNamespace(gpu=saved))
    UniModelState.prepare_inputs_embeds(state, {}, batch, requests)
    assert batch.input_ids.tolist() == [7, 157190, 157202, 8]
    assert saved[1, 6:].tolist() == [157190, 157202]
    assert saved[0].count_nonzero() == 0


def test_nemotron_vlm_opens_each_block_fully_masked():
    config = SimpleNamespace(mask_token_id=100, confidence_threshold=0.9)
    policy = registry.resolve("nemotron_vlm").from_config(config, None)
    canvas = torch.full((1, 32), 100)
    assert torch.equal(policy.init_canvas(canvas.clone(), first=True), canvas)
    assert policy.next_seed(torch.zeros(1, 32, 8)) is None
