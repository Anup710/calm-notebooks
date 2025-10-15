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