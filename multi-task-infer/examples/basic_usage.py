#!/usr/bin/env python3
"""
Basic usage example for ExternalExecutor with StorageCheckpointEngine.

This example demonstrates:
1. Creating ActorPoolManager with CacheManagerActor
2. Acquiring actors and creating ExternalExecutor
3. Loading model weights from persistent storage (NFS/Mooncake)
4. Compilation cache sharing via CacheManagerActor

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│              Storage (NFS/Mooncake Store)                        │
│  - Model checkpoints (safetensors)                               │
│  - Compilation caches                                            │
└─────────────────────────────────────────────────────────────────┘
         │                              │
         ↓                              ↓
┌─────────────────────┐    ┌─────────────────────────────────────┐
│  StorageCheckpoint  │    │        CacheManagerActor            │
│     Engine          │    │  - Compilation cache sharing         │
│  - Load model       │    │  - pull/push API                     │
│    weights          │    │  - Compile lock coordination         │
└─────────────────────┘    └─────────────────────────────────────┘
         │                              │
         ↓                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              ExternalExecutor Workers                            │
│  - Pre-started Ray Actors bound to GPU/NPU devices              │
│  - Load model via StorageCheckpointEngine                       │
│  - Share compilation cache via CacheManagerActor                │
└─────────────────────────────────────────────────────────────────┘
"""

import ray
from vllm import LLM, SamplingParams
from vllm.config import ModelConfig

from vllm_external_executor import (
    ActorPoolManager,
    ExternalExecutor,
    StorageCheckpointEngine,
)


def example_basic_usage():
    """Basic example: Load model from local files."""
    # Initialize Ray
    ray.init()
    
    # 1. Create ActorPoolManager with CacheManagerActor
    pool = ActorPoolManager()
    pool.pre_start(
        num_actors=8,
        devices_per_node=[0, 1, 2, 3, 4, 5, 6, 7],
        warmup_distributed=True,
        # Optional: NFS directory for large compilation caches
        shared_cache_dir="/shared/vllm_compile_cache",
        enable_cache_compression=True,
    )
    
    # 2. Acquire actors for a vLLM instance
    actors = pool.acquire(tp_size=4, pp_size=2)
    
    # 3. Create vLLM config
    model_config = ModelConfig(
        model="facebook/opt-125m",
        task="generate",
        tokenizer="facebook/opt-125m",
    )
    
    # 4. Create ExternalExecutor with cache_manager
    executor = ExternalExecutor(
        model_config=model_config,
        external_actors=actors,
        cache_manager=pool.cache_manager,
    )
    
    # 5. Create LLM instance
    llm = LLM(
        model="facebook/opt-125m",
        executor_class=ExternalExecutor,
        external_actors=actors,
        cache_manager=pool.cache_manager,
    )
    
    # 6. Run inference
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
    ]
    
    outputs = llm.generate(prompts, sampling_params)
    
    for output in outputs:
        prompt = output.prompt
        generated = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated: {generated!r}")
    
    # 7. Release actors back to pool
    pool.release(actors)
    
    # 8. Check cache stats
    cache_stats = ray.get(pool.cache_manager.get_stats.remote())
    print(f"Cache stats: {cache_stats}")
    
    # Cleanup
    pool.shutdown()
    ray.shutdown()


def example_load_from_storage():
    """Example: Load model weights from persistent storage (NFS)."""
    ray.init()
    
    # 1. Create ActorPoolManager
    pool = ActorPoolManager()
    pool.pre_start(
        num_actors=4,
        devices_per_node=[0, 1, 2, 3],
        warmup_distributed=True,
    )
    
    # 2. Acquire actors
    actors = pool.acquire(tp_size=4, pp_size=1)
    
    # 3. Load model from storage using StorageCheckpointEngine
    # This is compatible with verl's CheckpointEngineWithCache interface
    storage_config = {
        "base_path": "/shared/checkpoints",  # NFS mount point
    }
    
    # Create StorageCheckpointEngine
    engine = StorageCheckpointEngine(
        backend="nfs",
        config=storage_config,
        device="cuda",
    )
    engine.set_checkpoint("opt-125m/step_1000")
    
    # 4. Load weights from storage to each worker
    for actor in actors:
        ray.get(actor.load_model_from_storage.remote(
            checkpoint_path="opt-125m/step_1000",
            storage_backend="nfs",
            storage_config=storage_config,
        ))
    
    # 5. Create vLLM instance with pre-loaded weights
    llm = LLM(
        model="facebook/opt-125m",
        executor_class=ExternalExecutor,
        external_actors=actors,
    )
    
    # 6. Run inference
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    prompts = ["Hello, my name is"]
    outputs = llm.generate(prompts, sampling_params)
    
    for output in outputs:
        print(f"Generated: {output.outputs[0].text}")
    
    # Cleanup
    pool.release(actors)
    pool.shutdown()
    ray.shutdown()


def example_save_checkpoint_to_storage():
    """Example: Save model checkpoint to persistent storage."""
    # Create StorageCheckpointEngine
    engine = StorageCheckpointEngine(
        backend="nfs",
        config={"base_path": "/shared/checkpoints"},
    )
    engine.set_checkpoint("my-model/step_100")
    
    # Simulate saving weights
    import torch
    
    def weight_generator():
        """Generate model weights."""
        yield "model.layers.0.weight", torch.randn(1024, 1024)
        yield "model.layers.0.bias", torch.randn(1024)
        yield "model.layers.1.weight", torch.randn(1024, 1024)
        yield "model.layers.1.bias", torch.randn(1024)
    
    # Save to storage
    import asyncio
    asyncio.run(engine.send_weights(
        weight_generator(),
        global_steps=100,
    ))
    
    print("Checkpoint saved to storage")
    
    # Verify checkpoint exists
    assert engine.exists("my-model/step_100")
    print("Checkpoint verified")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example_name = sys.argv[1]
        if example_name == "basic":
            example_basic_usage()
        elif example_name == "storage":
            example_load_from_storage()
        elif example_name == "save":
            example_save_checkpoint_to_storage()
        else:
            print(f"Unknown example: {example_name}")
            print("Available: basic, storage, save")
    else:
        print("Usage: python basic_usage.py [basic|storage|save]")
        print("  basic   - Basic usage with CacheManagerActor")
        print("  storage - Load model from persistent storage")
        print("  save    - Save checkpoint to storage")
