# PyTorch Module & Parameter Containers

## The core rule

`nn.Module` only tracks sub-objects it can *see*. Assigning a bare Python `list`
or `dict` of modules/tensors as an attribute **hides** its contents — those
parameters/buffers won't appear in `state_dict()`, `.parameters()`, or move with
`.to(device)`, and the optimizer won't train them.

```python
self.layers = [Block() for _ in range(n)]   # BAD: invisible to nn.Module
```

The wrappers below make the contents visible (registered). Pick by: is it a
sub-module or a raw tensor? is it trained? ordered or name-keyed?

## For sub-modules (things with a `forward`)

### `nn.ModuleList` — ordered, no forward of its own
Index/iterate like a list; **you** write the flow. Needed for residuals, passing
extra args (e.g. RoPE positions), or conditionals.
```python
self.layers = nn.ModuleList([Block() for _ in range(n)])
for layer in self.layers:
    x = layer(x)
```

### `nn.ModuleDict` — name-keyed modules
```python
self.heads = nn.ModuleDict({"cls": Linear(d, 2), "reg": Linear(d, 1)})
out = self.heads["cls"](x)
```

### `nn.Sequential` — ordered AND has a forward
Chains its modules automatically. Use only for a straight pipe with no branching.
```python
self.mlp = nn.Sequential(Linear(d, h), nn.ReLU(), Linear(h, d))
out = self.mlp(x)   # runs all three in order
```

**ModuleList vs Sequential:** ModuleList = you control the flow (custom logic
allowed); Sequential = it runs the modules for you (no custom logic).

## For raw tensors (no `forward`)

### `nn.ParameterList` / `nn.ParameterDict` — learnable tensors (have grad)
```python
self.scales = nn.ParameterList([nn.Parameter(torch.ones(d)) for _ in range(n)])
```

### `register_buffer(name, tensor)` — state but NOT trained (no grad)
For RoPE `cos_t`/`sin_t`, BatchNorm running stats, causal masks. Moves with
`.to(device)`; appears in `state_dict` unless `persistent=False`. No "BufferList" —
for a variable number, register in a loop with generated names:
```python
self.register_buffer("cos_t", cos, persistent=False)
self.register_buffer(f"mask_{i}", m)   # variable count
```

## Building a ModuleList — construct, don't index-assign

`self.layers[i] = mod` does **not** grow a list; it overwrites an element that
must already exist. Create the container first:

```python
# A. comprehension (most common)
self.layers = nn.ModuleList([Block() for _ in range(n)])

# B. empty then append
self.layers = nn.ModuleList()
for _ in range(n):
    self.layers.append(Block())     # .append()/.extend() are fine
```

Pitfalls that bit me:
- Forgetting `super().__init__()` as the **first line** of `__init__` → assigning
  any submodule raises "cannot assign module before Module.__init__() call".
- `self.layers[i] = ...` before `self.layers` exists → AttributeError.
- `self.layers.len()` — not a method. Use `len(self.layers)` or `for x in self.layers`.

## Decision table

| You have…                         | trained? | ordered/keyed | use               |
|-----------------------------------|----------|---------------|-------------------|
| sub-modules, custom forward       | —        | ordered       | `ModuleList`      |
| sub-modules, custom forward       | —        | keyed         | `ModuleDict`      |
| sub-modules, plain pipe           | —        | ordered       | `Sequential`      |
| raw tensors                       | yes (grad)| ordered      | `ParameterList`   |
| raw tensors                       | yes (grad)| keyed        | `ParameterDict`   |
| raw tensors                       | no (grad) | —            | `register_buffer` |
