class DecodeConfig:
    def verify(self, vllm_config) -> None:
        if getattr(self, "denoise_steps", 0) is None:
            self.denoise_steps = self.canvas_length
