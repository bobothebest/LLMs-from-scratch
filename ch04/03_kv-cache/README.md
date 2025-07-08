# Bonus Material: KV Cache



**This folder implements the addition of a KV cache to the GPT model.** 

&nbsp;
## Overview

In short, a KV cache stores intermediate key (K) and value (V) computations for reuse during inference, which results in a substantial speed-up when generating responses. The downside is that it adds some complexity to the code, increases memory usage, and can't be used during training. However, the inference speed-ups are often well worth the trade-offs in code complexity and memory when deploying LLMs.

&nbsp;
## How it works

Imagine the LLM is generating some text. Concretely, suppose the LLM is given the following prompt: "Time flies".

The figure below shows an excerpt of the underlying attention score computation using a modified graphic from Chapter 3 with the key and value vectors highlighted:

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/kv-cache-attn-1.png?3" width=800>

Now, as we learned in Chapters 2 and 4, LLMs generate one word (or token) at a time. Suppose the LLM generated the word "fast" so that the prompt for the next round becomes "Time flies fast". This is illustrated in the next figure below:

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/kv-cache-attn-2.png?3" width=800>

As we can see, based on comparing the previous 2 figures, the keys, and value vectors for the first two tokens are exactly the same, and it would be wasteful to recompute them in each next-token text generation round.

So, the idea of the KV cache is to implement a caching mechanism that stores the previously generated key and value vectors for reuse, which helps us to avoid unnecessary recomputations.

&nbsp;

## KV cache implementation

There are many ways to implement a KV cache, with the main idea being that we only compute the key and value tensors for the newly generated tokens in each generation step.

I opted for a simple one that emphasizes code readability. I think it's easiest to just scroll through the code changes to see how it's implemented.

There are two files in this folder:

1. [`gpt_ch04.py`](gpt_ch04.py): Self-contained code taken from Chapter 3 and 4 to implement the LLM and run the simple text generation function
2. [`gpt_with_kv_cache.py`](gpt_with_kv_cache.py): The same as above, but with the necessary changes made to implement the KV cache. 

You can either 

a. Open the [`gpt_with_kv_cache.py`](gpt_with_kv_cache.py) file and look out for the `# NEW` sections that mark the new changes:

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/new-sections.png?3" width=800>

b. Check out the two code files via a file diff tool of your choice to compare the changes:

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/file-diff.png?3" width=800>

To summarize the implementation details, here's a short walkthrough.

&nbsp;

### 1. Registering the cache buffers

Inside the `MultiHeadAttention` constructor we add two non-persistent buffers, `cache_k` and `cache_v`, which will hold concatenated keys and values across steps:

```python
self.register_buffer("cache_k", None, persistent=False)
self.register_buffer("cache_v", None, persistent=False)
```

&nbsp;

### 2. Forward pass with `use_cache` flag

Next, we extend the `forward` method of the `MultiHeadAttention` class to accept `use_cache` argument. After projecting the new chunk of tokens into `keys_new`, `values_new` and `queries`, we either initialize the kv cache or append to our cache:

```python
def forward(self, x, use_cache=False):
    b, num_tokens, d_in = x.shape

    keys_new = self.W_key(x)  # Shape: (b, num_tokens, d_out)
    values_new = self.W_value(x)
    queries = self.W_query(x)
    #...

    if use_cache:
        if self.cache_k is None:
            self.cache_k, self.cache_v = keys_new, values_new
        else:
            self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
            self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
        keys, values = self.cache_k, self.cache_v
    else:
        keys, values = keys_new, values_new
        
    # ...
    
    num_tokens_Q = queries.shape[-2]
    num_tokens_K = keys.shape[-2]
    if use_cache:
        mask_bool = self.mask.bool()[
            self.ptr_current_pos:self.ptr_current_pos + num_tokens_Q, :num_tokens_K
        ]
        self.ptr_current_pos += num_tokens_Q
    else:
        mask_bool = self.mask.bool()[:num_tokens_Q, :num_tokens_K]
```

&nbsp;


### 3. Clearing the cache

When generating texts, between independent sequences (for instance to text generation calls) we must reset both buffers, so we also add a cache resetting method the to the `MultiHeadAttention` class:

```python
def reset_cache(self):
    self.cache_k, self.cache_v = None, None
    self.ptr_current_pos = 0
```

&nbsp;

### 4. Propagating `use_cache` in the full model

With the changes to the `MultiHeadAttention` class in place, we now modify the  `GPTModel` class. First, we add a position tracking for the token indices to the instructor:

```python
self.current_pos = 0
```

Then, we replace the one-liner block call with an explicit loop, passing `use_cache` through each transformer block:

```python
def forward(self, in_idx, use_cache=False):
    # ...
 
    if use_cache:
        pos_ids = torch.arange(
            self.current_pos, self.current_pos + seq_len,            
            device=in_idx.device, dtype=torch.long
        )
        self.current_pos += seq_len
    else:
        pos_ids = torch.arange(
            0, seq_len, device=in_idx.device, dtype=torch.long
        )
    
    pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
    x = tok_embeds + pos_embeds
    # ...
    for blk in self.trf_blocks:
        x = blk(x, use_cache=use_cache)
```

The above change then also requires a small modification to the `TransformerBlock` class to accept the `use_cache` argument:
```python
    def forward(self, x, use_cache=False):
        # ...
        self.att(x, use_cache=use_cache)
```

Lastly, we add a model-level reset to `GPTModel` to clear all block caches at once for our convenience:

```python
def reset_kv_cache(self):
    for blk in self.trf_blocks:
        blk.att.reset_cache()
    self.current_pos = 0
```

&nbsp;

### 5. Using the cache in generation

With the changes to the `GPTModel`, `TransformerBlock`, and `MultiHeadAttention`, finally, here's how we use the KV cache in a simple text generation function:

```python
def generate_text_simple_cached(model, idx, max_new_tokens, 
                                context_size=None, use_cache=True):
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():
        if use_cache:
            # Init cache with full prompt
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for _ in range(max_new_tokens):
                # a) pick the token with the highest log-probability (greedy sampling)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # b) append it to the running sequence
                idx = torch.cat([idx, next_idx], dim=1)
                # c) feed model only the new token
                logits = model(next_idx, use_cache=True)
        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx
```

Note that we only feed the model the new token in c) via `logits = model(next_idx, use_cache=True)`. Without caching, we feed the model the whole input `logits = model(idx[:, -ctx_len:], use_cache=False)` as it has no stored keys and values to reuse.

&nbsp;

## Simple performance comparison

After covering the KV cache on a conceptual level, the big question is how well it actually performs in practice on a small example. To give the implementation a try, we can run the two aforementioned code files as Python scripts, which will run the small 124 M parameter LLM to generate 200 new tokens (given a 4-token prompt "Hello, I am" to start with):

```bash
pip install -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt

python gpt_ch04.py

python gpt_with_kv_cache.py
```

On a Mac Mini with M4 chip (CPU), the results are as follows:

|                        | Tokens/sec |
| ---------------------- | ---------- |
| `gpt_ch04.py`          | 27         |
| `gpt_with_kv_cache.py` | 144        |

So, as we can see, we already get a ~5x speed-up with a small 124 M parameter model and a short 200-token sequence length. (Note that this implementation is optimized for code readability and not optimized for CUDA or MPS runtime speed, which would require pre-allocating tensors instead of reinstating and concatenating them.)

**Note:** The model generates "gibberish" in both cases, i.e., text that looks like this: 

> Output text: Hello, I am Featureiman Byeswickattribute argue logger Normandy Compton analogous bore ITVEGIN ministriesysics Kle functional recountrictionchangingVirgin embarrassedgl ...

This is because we haven't trained the model, yet. The next chapter trains the model, and you can use the KV-cache on the trained model (however, the KV cache is only meant to be used during inference) to generate coherent text. Here, we are using the untrained model to keep the code simple(r).

What's more important, though, is that both the `gpt_ch04.py` and `gpt_with_kv_cache.py` implementations produce exactly the same text. This tells us that the KV cache is implemented correctly -- it is easy to make indexing mistakes that can lead to divergent results.


&nbsp;

## KV cache advantages and disadvantages 

As sequence length increases, the benefits and downsides of a KV cache become more pronounced in the following ways:

- [Good] **Computational efficiency increases**: Without caching, the attention at step *t* must compare the new query with *t* previous keys, so the cumulative work scales quadratically, O(n²). With a cache, each key and value is computed once and then reused, reducing the total per-step complexity to linear, O(n).

- [Bad] **Memory usage increases linearly**: Each new token appends to the KV cache. For long sequences and larger LLMs, the cumulative KV cache grows larger, which can consume a significant or even prohibitive amount of (GPU) memory. As a workaround, we can truncate the KV cache, but this adds even more complexity (but again, it may well be worth it when deploying LLMs.)

## KV缓存的优点和缺点

随着序列长度的增加，KV缓存的好处和弊端会在以下方面变得更加明显：

- [好处] **计算效率提升**：在没有缓存的情况下，第*t*步的注意力机制必须将新查询与之前的*t*个键进行比较，因此累积工作量呈二次方增长，O(n²)。有了缓存，每个键和值只需计算一次然后重复使用，将每步的总复杂度降低到线性，O(n)。

- [坏处] **内存使用呈线性增长**：每个新标记都会追加到KV缓存中。对于长序列和大型LLM，累积的KV缓存会变得越来越大，这可能会消耗大量甚至是令人望而却步的（GPU）内存。作为解决方法，我们可以截断KV缓存，但这会增加更多复杂性（不过，在部署LLM时这样做可能仍然是值得的）。



&nbsp;
## Optimizing the KV Cache Implementation

While my conceptual implementation of a KV cache above helps with clarity and is mainly geared towards code readability and educational purposes, deploying it in real-world scenarios (especially with larger models and longer sequence lengths) requires more careful optimization.

&nbsp;
### Common pitfalls when scaling the cache

- **Memory fragmentation and repeated allocations**: Continuously concatenating tensors via `torch.cat` as shown earlier, leads to performance bottlenecks due to frequent memory allocation and reallocation.

- **Linear growth in memory usage**: Without proper handling, the KV cache size becomes impractical for very long sequences.

### 缓存扩展时的常见陷阱

- **内存碎片化和重复分配**：如前所示，通过`torch.cat`连续拼接张量会导致性能瓶颈，因为需要频繁的内存分配和重新分配。

- **内存使用的线性增长**：如果没有适当处理，KV缓存的大小对于非常长的序列来说会变得不切实际。

&nbsp;
#### Tip 1: Pre-allocate Memory

Rather than concatenating tensors repeatedly, we could pre-allocate a sufficiently large tensor based on the expected maximum sequence length. This ensures consistent memory use and reduces overhead. In pseudo-code, this may look like as follows:

#### 技巧1：预分配内存

与其重复拼接张量，我们可以基于预期的最大序列长度预先分配一个足够大的张量。这确保了一致的内存使用并减少了开销。在伪代码中，这可能看起来如下所示：

```python
# Example pre-allocation for keys and values
max_seq_len = 1024  # maximum expected sequence length
cache_k = torch.zeros((batch_size, num_heads, max_seq_len, head_dim), device=device)
cache_v = torch.zeros((batch_size, num_heads, max_seq_len, head_dim), device=device)
```

During inference, we can then simply write into slices of these pre-allocated tensors.

&nbsp;
#### Tip 2: Truncate Cache via Sliding Window

To avoid blowing up our GPU memory, we can implement a sliding window approach with dynamic truncation. Via the sliding window, we maintain only the last `window_size` tokens in the cache:

#### 技巧2：通过滑动窗口截断缓存

为了避免GPU内存爆炸，我们可以实现一个带有动态截断的滑动窗口方法。通过滑动窗口，我们只在缓存中保留最后`window_size`个 token：

```python
# Sliding window cache implementation
window_size = 512
cache_k = cache_k[:, :, -window_size:, :]
cache_v = cache_v[:, :, -window_size:, :]
```

&nbsp;
#### Optimizations in practice

You can find these optimizations in the [`gpt_with_kv_cache_optimized.py`](gpt_with_kv_cache_optimized.py) file. 


On a Mac Mini with an M4 chip (CPU), with a 200-token generation and a window size equal to the context length (to guarantee same results) below, the code runtimes compare as follows:

|                                  | Tokens/sec |
| -------------------------------- | ---------- |
| `gpt_ch04.py`                    | 27         |
| `gpt_with_kv_cache.py`           | 144        |
| `gpt_with_kv_cache_optimized.py` | 166        |

Unfortunately, the speed advantages disappear on CUDA devices as this is a tiny model, and the device transfer and communication outweigh the benefits of a KV cache for this small model. 


&nbsp;
## Additional Resources

1. [Qwen3 from-scratch KV cache benchmarks](../../ch05/11_qwen3#pro-tip-2-speed-up-inference-with-compilation)
2. [Llama 3 from-scratch KV cache benchmarks](../../ch05/07_gpt_to_llama/README.md#pro-tip-3-speed-up-inference-with-compilation)
3. [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) -- A more detailed write-up of this README
