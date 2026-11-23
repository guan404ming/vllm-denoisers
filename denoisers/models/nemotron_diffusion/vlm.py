import torch
from mistral_common.protocol.instruct.chunk import ImageChunk
from mistral_common.tokens.tokenizers.image import ImageConfig, ImageEncoder, SpecialImageIDs
from transformers import BatchFeature
from vllm.model_executor.models import mistral3
from vllm.model_executor.models.utils import WeightsMapper
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    PromptReplacement,
    PromptUpdateDetails,
)

from denoisers.models.image import ImageDummyInputs, ImageProcessingInfo
from denoisers.models.mixin import BlockDiffusionModel

IMAGE_START = "<|image_start|>"
IMAGE_IDS = SpecialImageIDs(img=19, img_break=20, img_end=21)
MAX_IMAGE_SIZE = 1400


class NemotronVLMProcessingInfo(ImageProcessingInfo):
    image_placeholder = IMAGE_START
    max_image_size = MAX_IMAGE_SIZE

    def image_encoder(self):
        config = self.get_hf_config()
        return ImageEncoder(
            ImageConfig(config.vision_config.patch_size, MAX_IMAGE_SIZE, config.spatial_merge_size),
            IMAGE_IDS,
        )


class NemotronVLMProcessor(BaseMultiModalProcessor[NemotronVLMProcessingInfo]):
    def _call_hf_processor(self, hf_data, hf_kwargs):
        encoder = self.info.image_encoder()
        images = [
            torch.from_numpy(encoder(ImageChunk(image=image)).image) for image in hf_data["images"]
        ]
        return BatchFeature({"pixel_values": images})

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        return {"pixel_values": MultiModalFieldConfig.batched("image")}

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        config = self.info.get_hf_config()
        patch = config.vision_config.patch_size * config.spatial_merge_size

        def replacement(index):
            pixels = out_mm_kwargs["image"][index]["pixel_values"].data
            height, width = pixels.shape[-2:]
            row = [IMAGE_IDS.img] * (width // patch) + [IMAGE_IDS.img_break]
            tokens = [18] + row * (height // patch)
            tokens[-1] = IMAGE_IDS.img_end
            return PromptUpdateDetails.select_token_id(tokens, IMAGE_IDS.img)

        return [PromptReplacement(modality="image", target=[18], replacement=replacement)]


@MULTIMODAL_REGISTRY.register_processor(
    NemotronVLMProcessor, info=NemotronVLMProcessingInfo, dummy_inputs=ImageDummyInputs
)
class NemotronLabsDiffusionVLMModel(BlockDiffusionModel, mistral3.Mistral3ForConditionalGeneration):
    supports_mm_device_do_normalize = False
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "encoder.vision_tower.": "vision_tower.",
            "encoder.multi_modal_projector.": "multi_modal_projector.",
            "encoder.": "language_model.model.",
            "diffusion_head.": "language_model.lm_head.",
        }
    )

    @property
    def lm_head(self):
        return self.language_model.lm_head

    def input_embedding(self):
        return self.language_model.model.embed_tokens
