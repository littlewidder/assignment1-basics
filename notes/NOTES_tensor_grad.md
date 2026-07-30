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
