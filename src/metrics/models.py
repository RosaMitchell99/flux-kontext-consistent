from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image


def _unit_vector(vector):
    return functional.normalize(vector.float(), p=2, dim=-1)


class VisionMetrics:
    CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
    DINO_MODEL_ID = "facebook/dinov2-base"

    def __init__(
        self,
        device=None,
        clip_model_id=CLIP_MODEL_ID,
        dino_model_id=DINO_MODEL_ID,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.clip_model_id = clip_model_id
        self.dino_model_id = dino_model_id
        self.clip_model = None
        self.clip_processor = None
        self.dino_model = None
        self.dino_processor = None

    @staticmethod
    def open_image(image):
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return Image.open(Path(image)).convert("RGB")

    def _load_clip(self):
        if self.clip_model is not None:
            return
        from transformers import CLIPModel, CLIPProcessor

        print(
            "[Metrics] Loading CLIP {} on {}.".format(self.clip_model_id, self.device),
            flush=True,
        )
        self.clip_model = (
            CLIPModel.from_pretrained(self.clip_model_id).to(self.device).eval()
        )
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_id)

    def clip_image_embedding(self, image):
        self._load_clip()
        inputs = self.clip_processor(
            images=self.open_image(image), return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            vision_outputs = self.clip_model.vision_model(
                pixel_values=inputs["pixel_values"]
            )
            embedding = self.clip_model.visual_projection(vision_outputs.pooler_output)
        return _unit_vector(embedding)

    def clip_text_embedding(self, text):
        self._load_clip()
        inputs = self.clip_processor(
            text=[text], return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.inference_mode():
            text_outputs = self.clip_model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
            embedding = self.clip_model.text_projection(text_outputs.pooler_output)
        return _unit_vector(embedding)

    @staticmethod
    def cosine(first, second):
        return float((first.reshape(-1) * second.reshape(-1)).sum().item())

    def clip_i(self, image_a, image_b):
        return self.cosine(
            self.clip_image_embedding(image_a),
            self.clip_image_embedding(image_b),
        )

    def clip_t(self, image, text):
        return self.cosine(
            self.clip_image_embedding(image), self.clip_text_embedding(text)
        )

    def _load_dino(self):
        if self.dino_model is not None:
            return
        from transformers import AutoImageProcessor, AutoModel

        print(
            "[Metrics] Loading DINOv2 {} on {}.".format(
                self.dino_model_id, self.device
            ),
            flush=True,
        )
        self.dino_processor = AutoImageProcessor.from_pretrained(self.dino_model_id)
        self.dino_model = (
            AutoModel.from_pretrained(self.dino_model_id).to(self.device).eval()
        )

    def dino_embedding(self, image):
        self._load_dino()
        inputs = self.dino_processor(
            images=self.open_image(image), return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            outputs = self.dino_model(**inputs)
            embedding = getattr(outputs, "pooler_output", None)
            if embedding is None:
                embedding = outputs.last_hidden_state[:, 0]
        return _unit_vector(embedding)

    def dino_similarity(self, image_a, image_b):
        return self.cosine(self.dino_embedding(image_a), self.dino_embedding(image_b))


class SAM3Segmenter:

    MODEL_ID = "facebook/sam3"

    def __init__(self, model_id=MODEL_ID, device=None, threshold=0.5):
        self.model_id = model_id
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.threshold = threshold
        self.model = None
        self.processor = None

    def _load(self):
        if self.model is not None:
            return
        from transformers import Sam3Model, Sam3Processor

        print(
            "[Metrics] Loading SAM3 {} on {}.".format(self.model_id, self.device),
            flush=True,
        )
        self.model = Sam3Model.from_pretrained(self.model_id).to(self.device).eval()
        self.processor = Sam3Processor.from_pretrained(self.model_id)

    def segment(self, image, text_prompt):
        if not text_prompt or not text_prompt.strip():
            raise ValueError("SAM3 needs a short text prompt naming the edit target.")
        self._load()
        image = VisionMetrics.open_image(image)
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(
            self.device
        )
        with torch.inference_mode():
            outputs = self.model(**inputs)
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.threshold,
            mask_threshold=0.5,
            target_sizes=inputs["original_sizes"].tolist(),
        )[0]
        masks = results.get("masks")
        if masks is None or len(masks) == 0:
            return None, None

        masks = torch.as_tensor(masks).detach().cpu()
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        scores = results.get("scores")
        if scores is None or len(scores) != len(masks):
            selected = 0
            confidence = None
        else:
            scores = torch.as_tensor(scores).detach().cpu()
            selected = int(scores.argmax().item())
            confidence = float(scores[selected].item())
        mask = masks[selected].numpy().astype(bool)
        if mask.shape != (image.height, image.width):
            mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
            mask = (
                np.asarray(mask_image.resize(image.size, Image.Resampling.NEAREST)) > 0
            )
        if not mask.any():
            return None, confidence
        return mask, confidence


def masked_region_crop(image, mask, include_mask=True):
    image = VisionMetrics.open_image(image)
    mask = np.asarray(mask, dtype=bool)
    if not include_mask:
        mask = ~mask
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    crop = image.crop((x0, y0, x1, y1))
    crop_mask = Image.fromarray(mask[y0:y1, x0:x1].astype(np.uint8) * 255)
    neutral = Image.new("RGB", crop.size, (127, 127, 127))
    neutral.paste(crop, mask=crop_mask)
    return neutral


def masked_full_image(image, mask, include_mask=True):
    image = VisionMetrics.open_image(image)
    mask = np.asarray(mask, dtype=bool)
    if not include_mask:
        mask = ~mask
    neutral = Image.new("RGB", image.size, (127, 127, 127))
    neutral.paste(image, mask=Image.fromarray(mask.astype(np.uint8) * 255))
    return neutral
