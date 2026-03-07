"""
CHAIR: Caption Hallucination Assessment with Image Relevance.

CHAIR_s: Fraction of captions containing ≥1 hallucinated object.
CHAIR_i: Fraction of hallucinated objects over all mentioned objects.
Lower is better.

Fix over previous version:
  - Multi-word COCO objects (e.g. "traffic light", "hot dog") now matched correctly
    via bigram extraction before single-word fallback.
  - Expanded synonym table covering common paraphrases.
  - Annotation-based GT loading (uses COCO instances_val2014.json).
"""

import json
import os
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Set
from collections import defaultdict
from tqdm import tqdm


# ── COCO 80-class object list ─────────────────────────────────────────────────
# Multi-word objects are listed first so bigram matching takes priority.
COCO_OBJECTS: List[str] = [
    # Multi-word (checked as bigrams)
    "traffic light", "fire hydrant", "stop sign", "parking meter",
    "sports ball", "baseball bat", "baseball glove", "tennis racket",
    "wine glass", "hot dog", "potted plant", "dining table",
    "cell phone", "teddy bear", "hair drier",
    # Single-word
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "kite", "skateboard", "surfboard", "bottle", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "pizza", "donut", "cake", "chair", "couch",
    "bed", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "toothbrush",
]

MULTI_WORD_OBJECTS: Set[str] = {o for o in COCO_OBJECTS if " " in o}
SINGLE_WORD_OBJECTS: Set[str] = {o for o in COCO_OBJECTS if " " not in o}

# ── Synonym table ─────────────────────────────────────────────────────────────
COCO_SYNONYMS: Dict[str, str] = {
    # person
    "people": "person", "man": "person", "woman": "person", "men": "person",
    "women": "person", "child": "person", "children": "person", "kid": "person",
    "kids": "person", "boy": "person", "girl": "person", "human": "person",
    "guy": "person", "lady": "person", "gentleman": "person",
    # bicycle
    "bike": "bicycle", "bikes": "bicycle", "bicycles": "bicycle",
    # car
    "car": "car", "cars": "car", "vehicle": "car", "vehicles": "car",
    "automobile": "car", "sedan": "car", "suv": "car",
    # motorcycle
    "motorbike": "motorcycle", "scooter": "motorcycle",
    # airplane
    "plane": "airplane", "planes": "airplane", "aircraft": "airplane", "jet": "airplane",
    # bus
    "buses": "bus",
    # train
    "trains": "train",
    # truck
    "trucks": "truck",
    # boat
    "boats": "boat", "ship": "boat", "ships": "boat",
    # bench
    "benches": "bench",
    # bird
    "birds": "bird", "pigeon": "bird", "pigeons": "bird", "duck": "bird",
    "ducks": "bird", "seagull": "bird",
    # cat
    "cats": "cat", "kitten": "cat", "kitty": "cat",
    # dog
    "dogs": "dog", "puppy": "dog", "puppies": "dog",
    # horse
    "horses": "horse",
    # sheep
    "lamb": "sheep",
    # cow
    "cows": "cow", "cattle": "cow",
    # elephant
    "elephants": "elephant",
    # bear
    "bears": "bear",
    # zebra
    "zebras": "zebra",
    # giraffe
    "giraffes": "giraffe",
    # backpack
    "backpacks": "backpack", "rucksack": "backpack", "bag": "backpack",
    # umbrella
    "umbrellas": "umbrella",
    # handbag
    "purse": "handbag", "pocketbook": "handbag",
    # suitcase
    "luggage": "suitcase", "suitcases": "suitcase",
    # frisbee
    "frisbees": "frisbee",
    # skis
    "ski": "skis",
    # snowboard
    "snowboards": "snowboard",
    # kite
    "kites": "kite",
    # skateboard
    "skateboards": "skateboard",
    # surfboard
    "surfboards": "surfboard",
    # bottle
    "bottles": "bottle",
    # wine glass
    "wineglass": "wine glass", "goblet": "wine glass",
    # cup
    "cups": "cup", "mug": "cup", "mugs": "cup", "glass": "cup",
    # fork
    "forks": "fork",
    # knife
    "knives": "knife",
    # spoon
    "spoons": "spoon",
    # bowl
    "bowls": "bowl",
    # banana
    "bananas": "banana",
    # apple
    "apples": "apple",
    # sandwich
    "sandwiches": "sandwich", "burger": "sandwich",
    # orange
    "oranges": "orange",
    # broccoli
    # carrot
    "carrots": "carrot",
    # hot dog
    "hotdog": "hot dog", "sausage": "hot dog",
    # pizza
    "pizzas": "pizza",
    # donut
    "donuts": "donut", "doughnut": "donut", "doughnuts": "donut",
    # cake
    "cakes": "cake",
    # chair
    "chairs": "chair", "stool": "chair",
    # couch
    "sofa": "couch", "sofas": "couch", "couches": "couch",
    # potted plant
    "plant": "potted plant", "plants": "potted plant", "flower": "potted plant",
    "flowers": "potted plant", "vase": "potted plant",
    # bed
    "beds": "bed",
    # dining table
    "table": "dining table", "tables": "dining table", "desk": "dining table",
    # toilet
    "toilets": "toilet",
    # tv
    "television": "tv", "monitor": "tv", "screen": "tv",
    # laptop
    "laptops": "laptop", "computer": "laptop", "notebook": "laptop",
    # mouse (computer)
    "mice": "mouse",
    # remote
    "remotes": "remote", "remote control": "remote",
    # keyboard
    "keyboards": "keyboard",
    # cell phone
    "phone": "cell phone", "phones": "cell phone", "smartphone": "cell phone",
    "mobile": "cell phone",
    # microwave
    "microwaves": "microwave",
    # oven
    "ovens": "oven", "stove": "oven",
    # sink
    "sinks": "sink",
    # refrigerator
    "fridge": "refrigerator", "fridges": "refrigerator",
    # book
    "books": "book",
    # clock
    "clocks": "clock", "watch": "clock",
    # scissors
    # teddy bear
    "teddy": "teddy bear", "stuffed animal": "teddy bear",
    # toothbrush
    "toothbrushes": "toothbrush",
    # fire hydrant
    "hydrant": "fire hydrant",
    # stop sign
    "stopsign": "stop sign",
    # traffic light
    "stoplight": "traffic light", "streetlight": "traffic light",
    # sports ball
    "ball": "sports ball", "balls": "sports ball", "soccer ball": "sports ball",
    "football": "sports ball", "basketball": "sports ball",
    # baseball bat
    # baseball glove
    # tennis racket
    "racket": "tennis racket", "racquet": "tennis racket",
    # hair drier
    "hairdryer": "hair drier", "hair dryer": "hair drier",
}


def _normalize(candidate: str) -> Optional[str]:
    """Map a word or phrase to a canonical COCO object name."""
    c = candidate.lower().strip()
    if c in COCO_OBJECTS:
        return c
    if c in COCO_SYNONYMS:
        return COCO_SYNONYMS[c]
    # Plural → singular simple heuristic for single-word objects
    for obj in SINGLE_WORD_OBJECTS:
        if c == obj + "s" or c == obj + "es":
            return obj
    return None


def extract_objects(caption: str) -> Set[str]:
    """
    Extract COCO object names mentioned in a caption.

    Strategy:
      1. Tokenise into words (lower-case).
      2. Check all consecutive bigrams against multi-word COCO objects first.
      3. Check individual words and synonym table.
    """
    words = re.findall(r"[a-zA-Z]+", caption.lower())
    found: Set[str] = set()

    # Bigrams (covers "traffic light", "hot dog", "cell phone", etc.)
    for i in range(len(words) - 1):
        bigram = words[i] + " " + words[i + 1]
        n = _normalize(bigram)
        if n:
            found.add(n)

    # Unigrams
    for w in words:
        n = _normalize(w)
        if n:
            found.add(n)

    return found


def compute_chair(
    captions: List[str],
    gt_objects: List[Set[str]],
) -> Dict[str, float]:
    """
    Compute CHAIR_s and CHAIR_i.

    Args:
        captions:    Generated captions (one per image).
        gt_objects:  Set of GT COCO object names per image.
    """
    total_mentioned = 0
    total_hallucinated = 0
    n_with_halluc = 0
    n_valid = 0       # captions that mention ≥1 COCO object

    for cap, gt in zip(captions, gt_objects):
        mentioned = extract_objects(cap)
        if not mentioned:
            continue
        n_valid += 1
        hallucinated = mentioned - gt
        total_mentioned += len(mentioned)
        total_hallucinated += len(hallucinated)
        if hallucinated:
            n_with_halluc += 1

    chair_s = n_with_halluc / n_valid if n_valid else 0.0
    chair_i = total_hallucinated / total_mentioned if total_mentioned else 0.0
    return {"chair_s": chair_s, "chair_i": chair_i, "n_evaluated": n_valid}


def load_coco_annotations(annot_path: str) -> Dict[int, Set[str]]:
    """Load COCO instances JSON → {image_id: set of object names}."""
    with open(annot_path, "r", encoding="utf-8") as f:
        ann = json.load(f)
    id_to_cat = {c["id"]: c["name"].lower() for c in ann["categories"]}
    img_to_objs: Dict[int, Set[str]] = defaultdict(set)
    for a in ann.get("annotations", []):
        cat_name = id_to_cat.get(a["category_id"], "")
        if cat_name:
            img_to_objs[a["image_id"]].add(cat_name)
    return dict(img_to_objs)


class CHAIREvaluator:
    """CHAIR benchmark evaluator."""

    def __init__(
        self,
        data_path: str = "data/mscoco",
        annot_path: Optional[str] = None,
        split: str = "val2014",
        num_samples: int = 500,
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.num_samples = num_samples
        self.annot_path = annot_path or str(
            self.data_path / "annotations" / "instances_val2014.json"
        )
        self.img_root = self.data_path / split
        self._gt: Optional[Dict[int, Set[str]]] = None

    def _load_gt(self) -> Dict[int, Set[str]]:
        if self._gt is None:
            if os.path.exists(self.annot_path):
                print(f"[CHAIR] Loading annotations from {self.annot_path}")
                self._gt = load_coco_annotations(self.annot_path)
            else:
                print(f"[CHAIR] WARNING: Annotation file not found: {self.annot_path}")
                self._gt = {}
        return self._gt

    def evaluate(
        self,
        model,
        processor,
        decode_fn: Callable,
        model_type: str = "llava",
        image_ids: Optional[List[int]] = None,
        prompt: str = "Describe this image in detail.",
        **decode_kwargs,
    ) -> Dict[str, float]:
        """
        Generate captions for COCO images and compute CHAIR_s / CHAIR_i.

        Args:
            model, processor: VLM and processor.
            decode_fn:        Callable(model, input_ids, ...) → generated_ids tensor.
            model_type:       "llava" | "qwen2_vl".
            image_ids:        Specific COCO image IDs to evaluate; if None, uses first
                              num_samples from GT.
            prompt:           Caption instruction sent to the model.
        """
        try:
            from src.models.model_loader import prepare_inputs
        except ImportError:
            from models.model_loader import prepare_inputs

        gt = self._load_gt()
        if not gt:
            return {"chair_s": 0.0, "chair_i": 0.0, "n_evaluated": 0}

        if image_ids is None:
            image_ids = list(gt.keys())[: self.num_samples]

        captions: List[str] = []
        gt_sets: List[Set[str]] = []

        for img_id in tqdm(image_ids, desc="CHAIR caption generation"):
            img_file = f"COCO_{self.split}_{img_id:012d}.jpg"
            img_path = self.img_root / img_file
            if not img_path.exists():
                continue
            try:
                inputs = prepare_inputs(processor, str(img_path), prompt, model_type)
                inputs = {
                    k: v.to(model.device) if hasattr(v, "to") else v
                    for k, v in inputs.items()
                }
                gen_ids = decode_fn(
                    model,
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"),
                    max_new_tokens=128,
                    **decode_kwargs,
                )
                prompt_len = inputs["input_ids"].shape[1]
                out_text = processor.decode(
                    gen_ids[0][prompt_len:], skip_special_tokens=True
                )
            except Exception as e:
                print(f"[CHAIR] Error on image {img_id}: {e}")
                out_text = ""
            captions.append(out_text)
            gt_sets.append(gt.get(img_id, set()))

        return compute_chair(captions, gt_sets)
