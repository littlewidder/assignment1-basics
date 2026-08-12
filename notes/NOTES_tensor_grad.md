# Tensor Data & Grad — Quick Reference

A compact reference on how tensor **data** and **gradients** behave across common
PyTorch operations. Generic (not assignment-specific).

## 1. Two separate things live on a tensor

- **data** — the forward values (a buffer in memory, identified by `.data_ptr()`).
- **grad** — `.grad`, a *separate* tensor filled in by `backward()`. Independent of
  the data buffer.

Sharing one does **not** imply sharing the other.

## 2. Views vs copies (the data buffer)

- **View** = new tensor *object*, new shape/strides, **same data buffer** (no copy, cheap).
  - `unsqueeze`, `squeeze`, `reshape`/`view`, `transpose`, `.T`, `.mT`, basic slicing
    `x[:, 0]`, `x[..., None]`, `expand`.
  - Editing a view's contents in place also changes the base (shared memory).
- **Copy** = independent data buffer.
  - `.clone()`, `x + 0`, most arithmetic (`a * b`), `.to(dtype)` when dtype changes,
    `.contiguous()` on a non-contiguous tensor.
- A shape op never mutates the *original's* shape — it returns a new object. `x` is untouched.

## 3. In-place vs out-of-place

- **Trailing underscore = in-place**, mutates the tensor itself: `add_`, `copy_`, `mul_`,
  `unsqueeze_`, `t_`, `zero_`, `fill_`.
- **No underscore = returns a new tensor**, leaves the input alone: `add`, `mul`, `unsqueeze`, …
- In-place ops on tensors that require grad can corrupt autograd → prefer out-of-place
  unless you know it's safe.

## 4. requires_grad, leaf, non-leaf

- **Leaf** = a tensor you created directly (e.g. `torch.randn(.., requires_grad=True)`, or an
  `nn.Parameter`). `is_leaf == True`.
- **Non-leaf** = output of an operation on a grad-tracking tensor. Has a `grad_fn`
  (e.g. `<UnsqueezeBackward0>`); `is_leaf == False`.
- After `backward()`:
  - **only leaves** accumulate `.grad`.
  - non-leaves have `.grad == None` (warning if accessed). Use `.retain_grad()` to keep it —
    but that's a **separate** grad tensor, not shared with the leaf.
- Gradient **accumulates** (adds) across multiple `backward()` calls → call
  `optimizer.zero_grad()` / `x.grad = None` between steps.

## 5. How grad flows through a view

- A view op (like `unsqueeze`) is a **differentiable node** in the graph.
- Backward routes the gradient *through* it (unsqueeze's backward ≈ squeeze) down to the leaf,
  reshaped to the leaf's shape.
- So a view shares **data** but its gradient is **routed, not shared**.

## 6. Escaping / controlling autograd

| Tool | Effect |
|---|---|
| `with torch.no_grad():` | ops inside don't build the graph (inference, weight init, manual updates) |
| `x.detach()` | new tensor, **shares data**, but cut off from the graph (no grad flows back) |
| `x.requires_grad_(True/False)` | toggle tracking on a leaf in place |
| `.data` | raw data view, bypasses autograd — **legacy**, avoid; use `no_grad`/`detach` |

## 7. Quick checks

```python
x.data_ptr() == y.data_ptr()   # same data buffer? (view vs copy)
x.is_leaf                       # leaf or op-output?
x.requires_grad                 # tracked by autograd?
x.grad_fn                       # None for leaves; the backward node otherwise
x.grad                          # populated only after backward, only on leaves
```

## Mental model

> **data** = the numbers (can be shared via views).
> **grad** = a separate bucket that `backward()` fills, only on leaves, routed through the op
> graph — never shared just because data is.

## 8. Broadcasting (element-wise ops: `*` `/` `+` `-` `**`, comparisons)

The rule for combining two shapes:

1. **Right-align** the two shapes (line up by the **last** dim).
2. **Pad the shorter shape with `1`s on the left** (missing leading dims count as size 1).
3. Each aligned dim pair is compatible if **equal** OR **one of them is `1`**.
4. Output size in that dim = **max** of the two; the size-`1` side is stretched (values reused).
5. Any pair that is neither equal nor has a `1` → **error**.

```
(batch, seq, d) * (d,)      # (d,) -> (1,1,d) -> OK -> (batch, seq, d)
(2, 1, 3)       * (5, 3)    # -> (2, 5, 3)
(3, 1)          * (1, 4)    # both stretch -> (3, 4)   (outer-product style)
(3, 4)          * (2, 4)    # 3 vs 2, no 1 -> ERROR
```

Notes:
- Broadcasting **never copies data** — the size-`1` dim is stretched via stride-0 views (cheap).
- **`@` (matmul) is different**: last two dims must satisfy `(...,m,n)@(...,n,p)` and the contracted
  `n` must match **exactly** (a `1` does NOT stretch there). Only the *leading* batch dims of `@`
  follow the broadcasting rule above.
- Keep `keepdim=True` on reductions (`mean`/`sum`) when you want the result to broadcast back
  against the original (e.g. RMSNorm: `(...,1)` vs `(...,d)`).

## 9. Reshaping: `reshape` vs `view` vs `unsqueeze`/`squeeze`

### Contiguity

A tensor is **contiguous** when its elements sit in memory in standard row-major order for its
shape. Ops that share data but rearrange strides make the *result* non-contiguous:
`transpose`, `permute`, strided slicing, `expand`. (They return a NEW tensor; the source is
unchanged and keeps its contiguity.)

### `reshape` vs `view`

| | `view` | `reshape` |
|---|---|---|
| Shares memory (no copy)? | **Always** | If layout allows, else copies |
| If a view isn't possible | **Errors** | Makes a contiguous **copy** |
| Fails on non-contiguous layout | Sometimes (shape-dependent) | Never (falls back to copy) |

- Both require the element count (`numel`) to match the target shape — mismatch errors for both.
- `reshape(x) ≈ x.view(...) if possible else x.contiguous().view(...)`.
- A non-contiguous tensor can STILL be `view`ed for stride-compatible shapes (identity, adding a
  unit dim, splitting a contiguous sub-dim); it fails only when the target would span across
  non-contiguous memory (error message literally says "Use .reshape(...) instead").

### When to use which

```
Add/remove a size-1 dim?          -> unsqueeze / squeeze  (or [..., None])
Change shape, don't care re copy? -> reshape   (safe default)
Need guaranteed no-copy / aliasing, ok to fail? -> view  (add .contiguous() if it errors)
Complex/named reshaping (split heads, merge pairs)? -> einops.rearrange
```

- **`reshape`** = safe general default: always succeeds (given matching numel), copies only if it
  must.
- **`view`** = deliberate no-copy / shared-memory tool: use when you rely on aliasing (writes
  propagate to the source) or want it to fail loudly rather than silently copy. Fix a failure
  with `.contiguous().view(...)` or just `.reshape(...)`.
- **`unsqueeze`/`squeeze`** = clean, layout-agnostic way to add/remove size-1 dims. Advantages
  over `reshape`: relative (no need to know the other dims) and self-limiting (can't accidentally
  merge/scramble real axes).

### Common gotcha

After `transpose`/`permute`, a following `.view(...)` may fail (non-contiguous). Insert
`.contiguous()` before `.view(...)`, or use `.reshape(...)`. This is why multi-head attention
code often has `.contiguous()` right before merging the head axis back into `d_model`.

## 10. Reductions & max/argmax

- **Global** (over all elements): `x.max()` -> scalar tensor. `.item()` for a Python number.
- **Along a dim**: `x.max(dim=d)` returns a NAMED TUPLE `(values, indices)` — unpack it
  (`vals, idx = x.max(dim=-1)`), index `[0]`/`.values`, or use `x.argmax(dim=d)` for just the
  position.
- **`keepdim=True`** keeps the reduced axis as size 1 (e.g. `(...,1)`) so the result broadcasts
  back against the original — same trick as `mean`/`sum`. Needed for softmax stability:
  `x - x.max(dim=-1, keepdim=True).values`.
- Gotcha: `torch.max(a, b)` (two tensors) is ELEMENT-WISE max (like `np.maximum`), different from
  the reduction `x.max(dim=...)`.

## 11. Transpose / axis reorder

| op | effect | use for |
|---|---|---|
| `x.T` | reverse ALL dims (2-D only; deprecated + warns on >2-D) | plain 2-D matrices |
| `x.mT` | swap LAST TWO dims (batch-safe) | batched matrices |
| `x.transpose(-2,-1)` | swap last two (explicit) | same as `.mT` |
| `x.transpose(i,j)` | swap dims i and j | arbitrary two axes |
| `x.permute(...)` | arbitrary reordering | full control |

- `(2,3,4).T` -> `(4,3,2)` (reverses all); `(2,3,4).mT` -> `(2,4,3)` (swaps last two).
- Attention scores: `Q @ K.mT` (NOT `K.T`) so only the `d_k` axes contract.
- `Linear`'s `x @ self.weight.T` is fine because `weight` is exactly 2-D.

## 12. Masking (additive, for softmax/attention)

Add a mask to scores BEFORE softmax. Blocked positions get **-inf** (NOT +inf):
`exp(-inf)=0` -> weight 0; `+inf` -> NaN.

```python
# build an additive matrix (0 keep, -inf block)
add_mask = torch.where(mask, 0.0, float("-inf"));  masked = scores + add_mask
# or fill directly (idiomatic); ~mask because mask=True means "keep"
masked = scores.masked_fill(~mask, float("-inf"))
```

- `torch.where(cond, a, b)` = element-wise pick (a where cond True else b); `mask` broadcasts.
- `masked_fill(mask, value)` is a Tensor method; writes `value` where mask is True; mask broadcasts.
- Gradient through masking (in- or out-of-place) is identical: filled positions get 0 grad
  (constant), unfilled pass through; after softmax the masked weights ~0 contribute nothing.
- Full-row masks (all -inf) -> softmax NaN (0/0). Causal masks keep the diagonal so they're safe.

## 13. In-place ops: the leaf rule & memory

- **Cannot in-place-edit a LEAF that requires grad** (a Parameter, or `requires_grad=True` leaf):
  raises "a leaf Variable that requires grad is being used in an in-place operation." The leaf is
  the differentiation target; mutating it would corrupt backward. Do intentional edits under
  `with torch.no_grad():` (that's how optimizer steps modify weights).
- **Version-counter error**: an in-place op that overwrites a value some other op needs for its
  backward raises at `.backward()` ("...modified by an inplace operation").
- **Memory**: out-of-place (`scores = scores.masked_fill(...)`) allocates a NEW buffer — a
  transient ~2x peak of THAT tensor (old freed once dereferenced), not permanent doubling.
  In-place (`masked_fill_`) reuses the buffer.
  - **Inference** (`torch.no_grad()`/`inference_mode()`): no graph -> in-place is safe and saves
    the peak. Worth it for the quadratic `(seq,seq)` attention scores on long sequences.
  - **Training**: prefer out-of-place — identical gradients, avoids version-counter footguns.
  - Bigger memory wins come from fused kernels (FlashAttention) that never materialize full scores.

## 14. matmul semantics & fuse-vs-batch

- **`@` / `matmul` operate on the LAST TWO dims**; all leading dims are BATCH (broadcast).
  So to matmul over axes A and B independently per "slice", arrange the tensor as
  `(..., slice, A, B)` — matmul axes trailing, independent axes leading.
- **`.mT` is a PROPERTY (no args)** — always swaps the last two dims. To swap arbitrary dims use
  `.transpose(i, j)` (a method that takes args). `x.mT(-3,-2)` is a bug (`'Tensor' object is not
  callable`).

### Fuse (concat) vs batch — when to use which

Litmus test: **is the dimension I want to merge the one being contracted (summed)?**

- **Fuse into ONE big GEMM** when you concat along a NON-contracted dim and the contracted dim is
  shared. E.g. QKV projection: same input `x` (shared contraction `d_in`), different weights
  stacked along output -> `cat([Wq,Wk,Wv]) ; qkv = x @ W.T ; q,k,v = qkv.chunk(3,-1)`. Faster:
  one kernel, fewer launches, `x` read once.
- **Batch (independent dim -> leading axis)** when each slice has its OWN operands and you contract
  a per-slice dim. E.g. attention scores `q_h @ k_hᵀ` contract `d_head` per head -> head must be a
  batch axis. Concatenating heads here would SUM across heads (wrong).
- **Never a Python `for` loop** over the slices: serial, many tiny kernel launches, no parallelism.
  Batched matmul dispatches ONE strided-batched GEMM doing all slices in parallel.

## 15. Splitting / merging dims (the multi-head pattern)

`reshape`/`view` don't accept `...`; use one of these to split the LAST dim into sub-dims:

```python
q.unflatten(-1, (n_head, d_head))        # cleanest: split dim -1 -> (..., n_head, d_head)
q.reshape(*q.shape[:-1], n_head, d_head) # unpack leading dims with *; or use -1 for one
rearrange(q, '... s (h e) -> ... s h e', h=n_head)   # einops (allows ...)
```

- Merge sub-dims back: `x.flatten(-2)` (merge last two) or `x.reshape(*x.shape[:-2], d_m)`.
- **Multi-head attention shape dance** (why the transpose is unavoidable):
  ```
  x                (..., seq, d_m)
  project @ W.mT   (..., seq, d_m)
  unflatten(-1)    (..., seq, n_head, d_head)   # head lands AFTER seq
  transpose(-3,-2) (..., n_head, seq, d_head)   # move head to batch; (seq,d_head) trailing
  attention        (..., n_head, seq, d_head)
  transpose(-3,-2) (..., seq, n_head, d_head)   # move head back
  flatten(-2)      (..., seq, d_m)              # concat heads
  @ Wo.mT          (..., seq, d_m)
  ```
  The projection lays out head as a non-contracted dim (after seq), but the score matmul needs
  head as a batch dim -> a transpose (or an equivalent `einsum`/`rearrange`) is unavoidable. It's
  a cheap stride view; any downstream `.contiguous()` copy is minor.

### flatten vs unflatten (asymmetric)

| | operates on | arg | direction |
|---|---|---|---|
| `flatten(start_dim, end_dim=-1)` | a RANGE of dims | start/end indices | merge many -> 1 |
| `unflatten(dim, sizes)` | ONE dim | the new size tuple | split 1 -> many |

- `flatten` merges the contiguous range `start_dim..end_dim` (default `end_dim=-1` = last dim);
  dims outside the range are untouched.
  - `(5,3,2,1).flatten(-2)` -> `(5,3,2)`   (last 2)
  - `(5,3,2,1).flatten(-3)` -> `(5,6)`      (last 3; leading 5 kept)
  - `(5,3,2,1).flatten(1,2)` -> `(5,6,1)`   (merge dims 1..2 only)
  - `(5,3,2,1).flatten()` -> `(30,)`        (no args = everything -> 1D)
  - So `flatten(-3)` is NOT always `(6,)` — it depends on rank (only when the tensor is exactly 3-D).
- `unflatten` splits ANY single dim (int, may be negative) into `sizes`; product of `sizes` must
  equal that dim's size; one `-1` may be inferred; other dims untouched.
  - `(6,12,8).unflatten(-1,(2,4))` -> `(6,12,2,4)`   (split last)
  - `(6,12,8).unflatten(1,(3,4))`  -> `(6,3,4,8)`    (split middle)
  - `(6,12,8).unflatten(0,(2,3))`  -> `(2,3,12,8)`   (split first)
  - `(6,12,8).unflatten(1,(3,-1))` -> `(6,3,4,8)`    (-1 inferred)
  - wrong product -> error (e.g. `unflatten(-1,(3,3))` on dim size 8).
- They invert each other: `x.unflatten(dim, sizes).flatten(dim, dim+len(sizes)-1) == x` (shape).
- Asymmetry reason: merging needs a *range* (which consecutive dims to fuse); splitting needs a
  *single dim + target shape* (only one dim, but you must say into what). `unflatten` can't take a
  range.
