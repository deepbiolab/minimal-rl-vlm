"""
Modified by https://github.com/GAIR-NLP/MAYE/tree/master/maye/datasets
For more details, visit: https://arxiv.org/abs/2504.02587
"""


import torch
from torch.utils.data import Dataset, DataLoader
from itertools import chain
from typing import List, Dict, Any, Tuple

from .utils import open_jsonl

class MathGenerationDataset(Dataset):
    """Dataset for math generation tasks"""
    
    def __init__(
        self,
        dataset_path,
        use_chat_template=True,
    ):
        self.data = open_jsonl(dataset_path)
        self.use_chat_template = use_chat_template
        print(f"Loaded {len(self.data)} examples from {dataset_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract image path and prompt
        image_path = item.get("image", "")
        prompt = item.get("question", "")
        
        # Create a sample with the image and prompt
        sample = {
            "prompt": prompt,
            "images": [image_path],  # List for compatibility with multi-image inputs
            "answer": item.get("answer", ""),
            "id": item.get("id", str(idx)),
        }
        
        return sample

def collate_vision_inputs(samples: List[Dict], processor):
    """Collate images for vision model input"""
    images = list(chain.from_iterable(sample["images"] for sample in samples))
    
    # Handle Qwen processor specifically
    if "Qwen" in processor.__class__.__name__:
        try:
            from qwen_vl_utils import fetch_image
            images = [fetch_image({"image": image}) for image in images]
        except ImportError:
            # Fallback to direct loading if qwen_vl_utils is not available
            images = [image for image in images]
        
        vision_inputs = processor.image_processor(images=images, return_tensors="pt")
    else:
        raise ValueError(f"Unsupported processor: {processor.__class__.__name__}")
    
    return vision_inputs

def collate_generation_vllm(
    samples: List[Dict],
    processor,
    use_chat_template=True,
) -> Tuple[List[Dict[str, Any]], List[Dict]]:
    """Collate function for generation with vLLM"""
    images = list(chain.from_iterable(sample["images"] for sample in samples))

    prompts = []
    for sample in samples:
        prompt = sample["prompt"]
        if use_chat_template:
            prompt = processor.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
        prompts.append(prompt)

    # Handle Qwen processor
    if "Qwen" in processor.__class__.__name__:
        try:
            from qwen_vl_utils import fetch_image
            images = [fetch_image({"image": image}) for image in images]
        except ImportError:
            # Fallback
            pass

    # Create inputs for vLLM
    inputs = [
        {
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        }
        for prompt, image in zip(prompts, images)
    ]

    return inputs, samples

def collate_rlhf_vllm(
    samples: List[Dict],
    processor,
    padding_strategy="longest",
    use_chat_template=True,
) -> Tuple[List, List[Dict[str, Any]], List[Dict]]:
    """Collate function for RLHF with vLLM"""
    # Set padding side to left for causal language modeling
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    if hasattr(processor, "padding_side"):
        processor.padding_side = "left"

    # Extract images and prompts
    images = list(chain.from_iterable(sample["images"] for sample in samples))
    assert len(images) == len(samples), "Each sample must have exactly one image"
    
    prompts = []
    for sample in samples:
        prompt = sample["prompt"]
        if use_chat_template:
            prompt = processor.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
        prompts.append(prompt)

    # Tokenization parameters
    text_kwargs = {
        "padding": padding_strategy,
        "return_tensors": "pt",
    }
    
    # Handle Qwen processor
    if "Qwen" in processor.__class__.__name__:
        try:
            from qwen_vl_utils import fetch_image
            images = [fetch_image({"image": image}) for image in images]
        except ImportError:
            # Fallback
            pass

    # Create vLLM inputs
    vllm_inputs = [
        {
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        }
        for prompt, image in zip(prompts, images)
    ]

    # Process inputs with the processor
    batch_encoding = processor(images=images, text=prompts, **text_kwargs)
    vision_inputs = [
        processor.image_processor(images=img, return_tensors="pt") for img in images
    ]

    # Combine text and vision inputs
    encodings = [{} for _ in range(len(samples))]
    for key, value in batch_encoding.items():
        if key not in vision_inputs[0]:
            splits = torch.split(value, 1, dim=0)
            for encoding, split in zip(encodings, splits):
                encoding[key] = split
    
    for encoding, vision_input in zip(encodings, vision_inputs):
        encoding.update(vision_input)

    return encodings, vllm_inputs, samples

def create_dataloader(dataset, batch_size, processor, collate_fn=None, shuffle=True):
    """Create a dataloader with the specified dataset and collate function"""
    if collate_fn is None:
        collate_fn = lambda samples: collate_rlhf_vllm(samples, processor)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )