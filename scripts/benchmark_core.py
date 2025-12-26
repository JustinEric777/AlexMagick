import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from core.models.engine import ModelEngine
from core.models.adapters import resolve_model_config

def benchmark_llm(arch="OpenVino", model_name="DeepSeek", model_path=""):
    print(f"Benchmarking LLM: {model_name} ({arch})")
    
    # Resolve config
    backend, impl = resolve_model_config("llm", arch, model_name)
    print(f"Resolved: backend={backend}, impl={impl}")

    # Initialize Engine
    start_load = time.time()
    try:
        engine = ModelEngine(
            model_type="llm",
            model_name_or_path=model_path,
            device="CPU",
            backend=backend,
            impl=impl
        )
        print(f"Model Loaded in {time.time() - start_load:.2f}s")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Inference
    prompt = "Hello, who are you?"
    print(f"Prompt: {prompt}")
    
    start_gen = time.time()
    try:
        # LLM usually returns a generator or iterator
        response_gen = engine.generate([{"role": "user", "content": prompt}], max_tokens=50)
        
        full_response = ""
        first_token_time = None
        
        for chunk in response_gen:
            # Chunk format varies, usually (text, ...)
            if isinstance(chunk, tuple):
                text = chunk[0]
            elif isinstance(chunk, str):
                text = chunk
            else:
                text = str(chunk)
            
            if not first_token_time:
                first_token_time = time.time()
            
            # Simple accumulation (depends on how yield works, some yield accumulated text, some yield tokens)
            # Assuming accumulated text for simplicity based on previous code analysis
            full_response = text 
            
        end_gen = time.time()
        
        print(f"Response: {full_response[:100]}...")
        print(f"Total Generation Time: {end_gen - start_gen:.2f}s")
        if first_token_time:
            print(f"TTFT (Time To First Token): {first_token_time - start_gen:.2f}s")
            
    except Exception as e:
        print(f"Inference failed: {e}")
    finally:
        engine.close()

if __name__ == "__main__":
    # Example usage (requires actual model path)
    # benchmark_llm(model_path="/path/to/model")
    print("Please edit the script to provide a valid model_path for benchmarking.")
