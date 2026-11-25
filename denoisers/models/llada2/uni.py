from PIL import Image, ImageOps
from torch import nn
from transformers import GlmImageConfig, GlmImageImageProcessor
from transformers.models.glm_image.modeling_glm_image import GlmImageVisionModel, GlmImageVQVAE
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    PromptReplacement,
    PromptUpdateDetails,
)

from denoisers.models.common.image import ImageDummyInputs, ImageProcessingInfo
from denoisers.models.common.mixin import BlockDiffusionModel
from denoisers.models.llada2.model import LLaDA2MoeModelLM

IMAGE_START = "<|image|>"


def resize_image(image):
    width, height = image.size
    h, w = smart_resize(height, width, factor=32, min_pixels=128**2, max_pixels=800**2)
    scale = max(w / width, h / height)
    image = image.convert("RGB").resize(
        (round(width * scale), round(height * scale)), Image.Resampling.BICUBIC
    )
    return ImageOps.crop(
        image,
        (
            (image.width % 32) // 2,
            (image.height % 32) // 2,
            (image.width % 32 + 1) // 2,
            (image.height % 32 + 1) // 2,
        ),
    )


class UniProcessingInfo(ImageProcessingInfo):
    image_placeholder = IMAGE_START
    max_image_size = 800
    max_images = 1


class UniProcessor(BaseMultiModalProcessor[UniProcessingInfo]):
    def _call_hf_processor(self, hf_data, hf_kwargs):
        images = [resize_image(image) for image in hf_data["images"]]
        processor = GlmImageImageProcessor(
            do_resize=False,
            patch_size=16,
            temporal_patch_size=1,
            merge_size=1,
            image_mean=[0.5] * 3,
            image_std=[0.5] * 3,
        )
        return processor(images=images, return_tensors="pt")

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        sizes = hf_inputs["image_grid_thw"].prod(-1)
        return {
            "pixel_values": MultiModalFieldConfig.flat_from_sizes("image", sizes),
            "image_grid_thw": MultiModalFieldConfig.batched("image"),
        }

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        tokenizer = self.info.get_tokenizer()
        start = tokenizer.encode(IMAGE_START, add_special_tokens=False)
        pad = self.info.get_hf_config().image_token_offset

        def replacement(index):
            _, height, width = out_mm_kwargs["image"][index]["image_grid_thw"].data.tolist()
            header = f"{IMAGE_START}<|reserved_token_{height}|><|reserved_token_{width}|><boi>"
            tokens = tokenizer.encode(header, add_special_tokens=False) + [pad] * (height * width)
            tokens += tokenizer.encode("<|/image|>", add_special_tokens=False)
            return PromptUpdateDetails.select_token_id(tokens, pad)

        return [PromptReplacement(modality="image", target=start, replacement=replacement)]


@MULTIMODAL_REGISTRY.register_processor(
    UniProcessor, info=UniProcessingInfo, dummy_inputs=ImageDummyInputs
)
class LLaDA2UniForImageUnderstanding(BlockDiffusionModel, nn.Module, SupportsMultiModal):
    image_encoder_output = "token_ids"
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "image_tokenizer.model.visual.": "visual.",
            "image_tokenizer.model.vqmodel.": "vqmodel.",
            "model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality, i):
        return IMAGE_START

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        model = vllm_config.model_config
        self.config = model.hf_config
        with self._mark_language_model(vllm_config):
            self.language_model = LLaDA2MoeModelLM(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
            )
        vision = GlmImageConfig.from_pretrained(
            model.model, subfolder="image_tokenizer", revision=model.revision
        )
        with self._mark_tower_model(vllm_config, "image"):
            self.visual = GlmImageVisionModel(vision.vision_config)
            self.vqmodel = GlmImageVQVAE(vision.vq_config)
        self.secondary_weights = [
            DefaultModelLoader.Source(
                model.model, model.revision, subfolder="image_tokenizer", prefix="image_tokenizer."
            )
        ]

    def input_embedding(self):
        return self.language_model.input_embedding()

    def embed_multimodal(self, pixel_values=None, image_grid_thw=None, **kwargs):
        if pixel_values is None:
            return []
        hidden = self.visual(
            pixel_values.to(self.visual.dtype), grid_thw=image_grid_thw
        ).last_hidden_state
        images = []
        for _, height, width in image_grid_thw.tolist():
            count = height * width
            grid = hidden[:count].T.reshape(1, -1, height, width).contiguous()
            codes = self.vqmodel.encode(grid).image_tokens
            images.append((codes + self.config.image_token_offset).view(-1, 1))
            hidden = hidden[count:]
        return images

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        return self.language_model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states):
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights):
        return AutoWeightsLoader(self).load_weights(weights, mapper=self.hf_to_vllm_mapper)
