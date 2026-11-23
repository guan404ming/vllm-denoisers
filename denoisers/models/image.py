from vllm.model_executor.models.pixtral import PixtralDummyInputsBuilder
from vllm.multimodal.parse import ImageSize
from vllm.multimodal.processing import BaseProcessingInfo


class ImageProcessingInfo(BaseProcessingInfo):
    image_placeholder: str
    max_image_size: int
    max_images: int | None = None

    def get_supported_mm_limits(self):
        return {"image": self.max_images}

    def get_supported_mm_processor_kwargs(self):
        return {}

    def get_image_size_with_most_features(self):
        return ImageSize(width=self.max_image_size, height=self.max_image_size)


class ImageDummyInputs(PixtralDummyInputsBuilder):
    def get_dummy_text(self, mm_counts):
        return self.info.image_placeholder * mm_counts.get("image", 0)
