import shutil
from pathlib import Path
from uuid import uuid4


class Generator:
    MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"

    def __init__(
        self,
        model_id=MODEL_ID,
        output_dir=None,
        device=None,
        guidance_scale=2.5,
        pipeline=None,
    ):
        self.model_id = model_id
        self.output_dir = (
            Path(output_dir) if output_dir else Path(__file__).parent / "outputs"
        )
        self.device = device
        self.guidance_scale = guidance_scale
        self.pipeline = pipeline

    def _load_pipeline(self):
        if self.pipeline is not None:
            return self.pipeline

        try:
            import torch
            from diffusers import FluxKontextPipeline
        except ImportError as exc:
            raise RuntimeError(
                "FLUX.1 Kontext generation requires torch and diffusers to be installed."
            ) from exc

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(
            "[Generator] Loading {} on {}...".format(self.model_id, self.device),
            flush=True,
        )
        if str(self.device).startswith("cuda"):
            dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
        else:
            dtype = torch.float32

        self.pipeline = FluxKontextPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
        )
        self.pipeline.to(self.device)
        print("[Generator] FLUX pipeline ready.", flush=True)
        return self.pipeline

    @staticmethod
    def _load_reference(reference_image):
        if reference_image is None:
            return None
        if not isinstance(reference_image, (str, Path)):
            return reference_image

        try:
            from diffusers.utils import load_image
        except ImportError as exc:
            raise RuntimeError(
                "Loading a reference image requires diffusers to be installed."
            ) from exc
        return load_image(str(reference_image))

    def generate(self, prompt, reference_image=None, output_name=None):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        pipeline = self._load_pipeline()
        kwargs = {
            "prompt": prompt,
            "guidance_scale": self.guidance_scale,
        }
        reference = self._load_reference(reference_image)
        if reference is not None:
            kwargs["image"] = reference

        print(
            "[Generator] Generating image (reference image: {}).".format(
                "yes" if reference is not None else "no"
            ),
            flush=True,
        )

        try:
            import torch
            with torch.inference_mode():
                result = pipeline(**kwargs)
        except ImportError:
            result = pipeline(**kwargs)

        if not result.images:
            raise RuntimeError("FLUX.1 Kontext returned no generated image")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_name = output_name or "{}.png".format(uuid4().hex)
        image_path = self.output_dir / Path(image_name).name
        result.images[0].save(image_path)
        print("[Generator] Saved image: {}".format(image_path), flush=True)
        return str(image_path)

    def publish_scene(self, image_path, turn, filename=None):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        scene_path = self.output_dir / (filename or "scene_{}.png".format(turn))
        if Path(image_path).resolve() != scene_path.resolve():
            shutil.copy2(image_path, scene_path)
        print("[Generator] Published scene frame: {}".format(scene_path), flush=True)
        return str(scene_path)
