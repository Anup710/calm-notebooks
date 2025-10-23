<b><span style="color:red;">NOTE:</span></b> Performance gains are relative to previous step and not absolute. 

1. Assemble architecture of gpt2 to load pretrain weights 

    1. inference with actual openai weights
    2. inference with random weight 

2. Do pre-training on mini-shakespeare

    1. add loss calculation in class GPT and verify uniform random init 
    2. loss.backward and update steps (overfit on a single batch) 
        - loss at init = 10.87 (~ln(1/50257))
        - loss at step 50 on cpu = 0.003 (overfit done) _doable on cpu_
    3. data loader for fresh batch sampling (batch_size: 4,32)
        - loss after 50 iters ~ 6.5 (11-> 6.5 gained by reducing probability of non-english language vocab mainly) _doable on cpu_
        - in 50 iters, not even a single epoch is done btw
    4. Fixing the bug - weights of lm_head and token_embedding (wte) 
        - reduces params from 163.04M -> 124.44M
        - loss after 50 iters ~ 7 
    5. Fix initialization: adhere to actual openai gpt2 (infer from code, not mentioned in paper explicitly)
        - scale residual pathways by $\frac{1}{\sqrt(N)}$, best to be understood [here](https://youtu.be/l8pRSuU81PU?si=pBykX2rcwagxq4io&t=4432). 

3. Precision, scaling batch_size + block_size, tokens/sec and tracking time per step 

    1. Crank up to (B = 16,T= 1024). Take a baseline reading on time required to run each step. implmenent `torch.cuda.synchronize()`. Handle OOM errors by choosing a bigger GPU or a smaller batch size for baseline. 
    2. On changing precision: [read this pytorch documentation page](https://docs.pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices) for latest apis. Just adding `torch.backends.fp32_precision = "tf32"` doesnt make the expected jump in tokens/sec - since tensors are still float32 even if operations are tf32. 
    3. BF16 and [torch.autocast()](https://docs.pytorch.org/docs/stable/amp.html) of forward pass and loss ONLY. __~30% gain__
    4. Run `torch.compile()` on the model: [read more](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) __~ 60% gain__
    5. Flash attention using online softmax calculation, never creating the `att` tensor explicitly __~25% gain__
    6. justification for vocab size, batch size, good/ ugly numbers (powers of 2 are better)
        - Adding a few spurious rows by changing vocab_size: 50257 $\rightarrow$ 50304, doesn't break anything because through the optimization, the newtwork learns to drive those probabilities to 0, just like all probs which don't occur in mini-shakespeare
        - kernels like nice nos for quicker batch processing on gpus/ hbm storage
        - __~5% gain__
