import torch
from unsloth import FastLanguageModel
from vergil.core.env import VERGILEnv
from vergil.core.pomdp import POMDPWrapper
from scripts.train_grpo_colab import state_to_prompt

def test_vergil_agent():
    print("\n📦 Loading VERGIL SOTA Adapter...")
    model_name = "thekrishdshah/vergil-qwen-grpo"
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = 2048,
        load_in_4bit = True,
        dtype = None,
    )
    FastLanguageModel.for_inference(model)

    print("\n🌍 Initializing Test Environment...")
    env = VERGILEnv(seed=123) # Different seed for fresh testing
    pomdp = POMDPWrapper(env)
    
    # Manually trigger a "High Pressure" scenario
    state, _, _ = pomdp.reset() 
    prompt = state_to_prompt(state, env)

    print("\n💬 Sending State to Agent...")
    print("-" * 30)
    print(f"INPUT CAPACITY: {getattr(state, 'available_hours_next_48h', 0)}h")
    print("-" * 30)

    inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
    
    # Generate with the new reasoning capabilities
    outputs = model.generate(
        **inputs, 
        max_new_tokens = 512,
        temperature = 0.1,
        do_sample = False
    )
    
    response = tokenizer.batch_decode(outputs, skip_special_tokens = True)[0]
    
    # Extract only the completion (skip the prompt)
    completion = response[len(prompt):]
    
    print("\n🧠 AGENT REASONING & DECISION:")
    print("=" * 50)
    print(completion)
    print("=" * 50)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ Error: CUDA-capable GPU required for inference test.")
    else:
        test_vergil_agent()
