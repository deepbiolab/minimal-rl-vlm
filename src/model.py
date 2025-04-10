"""
Modified by https://github.com/GAIR-NLP/MAYE/tree/master/maye/training
For more details, visit: https://arxiv.org/abs/2504.02587
"""


import torch
from transformers import AutoConfig, Qwen2VLForConditionalGeneration, AutoProcessor
from vllm import LLM, SamplingParams


class VLLMWrapper:
    """
    Wrapper for vLLM to handle generation
    """
    def __init__(
        self,
        model_path,
        gpu_memory_utilization=0.8,
        dtype="bfloat16",
    ):
        print(f"Initializing vLLM with model: {model_path}")
        
        device = torch.device("cuda:0")

        # Convert string dtype to torch dtype
        if dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        # Initialize vLLM
        self.llm = LLM(
            model=model_path,
            device=device,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=torch_dtype,
        )
        
        print("vLLM initialized successfully")
        
    def generate(
        self,
        inputs,
        sampling_params=None,
    ):
        """Generate completions using vLLM"""
        if sampling_params is None:
            sampling_params = SamplingParams(
                max_tokens=512,
                temperature=1.0,
                top_p=1.0,
            )
            
        # Extract prompts and images from inputs
        prompts = [inp["prompt"] for inp in inputs]
        multi_modal_data = [inp.get("multi_modal_data", {}) for inp in inputs]
        
        # Generate completions
        outputs = self.llm.generate(
            prompts=prompts,
            sampling_params=sampling_params,
            multi_modal_data=multi_modal_data,
        )
        
        return outputs

def setup_model(
    model_path,
    dtype="bfloat16",
    train_vit=False,
    train_connector=True,
    train_llm=True,
):
    """
    Set up the policy model and reference model
    """
    device = torch.device("cuda:0")
    
    # Convert string dtype to torch dtype
    if dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    # Load model configuration
    config = AutoConfig.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
    )
    config.use_cache = False
    
    # Load model
    print(f"Loading model from {model_path}")
    policy_model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch_dtype,
    )
    
    # Load reference model (with same weights but no gradient updates)
    ref_policy_model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch_dtype,
    )
    
    # Set up training mode
    policy_model.train()
    ref_policy_model.eval()
    
    # Freeze reference model parameters
    for param in ref_policy_model.parameters():
        param.requires_grad = False
    
    # Handle Qwen model specifically
    llm_backend = policy_model.model
    vit_backend = policy_model.visual
    connector = policy_model.visual.merger
    
    # Set training flags for different components
    if not train_llm:
        for param in llm_backend.parameters():
            param.requires_grad = False
            
    if not train_vit:
        for param in vit_backend.parameters():
            param.requires_grad = False
            
    if train_connector:
        for param in connector.parameters():
            param.requires_grad = True
    
    # Move models to device
    policy_model = policy_model.to(device)
    ref_policy_model = ref_policy_model.to(device)
    
    return policy_model, ref_policy_model

def load_processor(model_path):
    """Load the model processor/tokenizer"""
    return AutoProcessor.from_pretrained(model_path)

def get_trainable_parameters(model):
    """Get trainable parameters and their count"""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in trainable_params)
    
    print(f"Total trainable parameters: {total_params:,}")
    
    return trainable_params

def sync_weights_to_vllm(policy_model, vllm_wrapper):
    """
    Sync weights from policy model to vLLM
    
    This function extracts the weights from the policy model and loads them into the vLLM model.
    """
    print("Syncing weights from policy model to vLLM")
    
    # Extract state dict from policy model
    cpu_state_dict = {}
    sharded_sd = policy_model.state_dict()
    
    # Process each parameter
    for param_name, param in sharded_sd.items():
        # Handle CPU offloaded parameters
        if hasattr(param, "is_cpu") and param.is_cpu:
            # Move back to device if offloaded to CPU
            device = next(policy_model.parameters()).device
            param = param.to(device)
        
        # Handle distributed tensor (DTensor)
        if hasattr(param, "_local_tensor"):
            # If using distributed tensors, get the full tensor
            param = param.full_tensor()
        
        # Move parameter to CPU for vLLM
        cpu_state_dict[param_name] = param.cpu()
    
    # Access the underlying model in vLLM
    try:
        # Navigate through vLLM's object hierarchy to get the model
        llm_model = vllm_wrapper.llm.llm_engine.model_executor.driver_worker.model_runner.model
        
        # Load weights into vLLM model
        print(f"Loading {len(cpu_state_dict)} parameters into vLLM model")
        llm_model.load_weights(cpu_state_dict.items())
        print("Successfully synced weights to vLLM")
    except AttributeError as e:
        print(f"Error accessing vLLM model: {e}")
        print("vLLM model structure might have changed. Please check the vLLM version and update the code accordingly.")
    except Exception as e:
        print(f"Error syncing weights to vLLM: {e}")
        
    # Clear CPU memory
    del cpu_state_dict
    import gc
    gc.collect()
    torch.cuda.empty_cache()
