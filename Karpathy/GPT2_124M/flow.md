1. Assemble architecture of gpt2 to load pretrain weights 
    1. inference with actual openai weights
    2. inference with random weight 

2. Do pre-training on mini-shakespeare
    1. add loss calculation in class GPT and verify uniform random init 
    2. loss.backward and update steps (overfit on a single batch) 
        - loss at init = 10.87 (~ln(1/50257))
        - loss at step 50 on cpu = 0.003 (overfit done)
    3. data loader for fresh batch sampling