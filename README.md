# AdaptiveABC

#### Installation

To install the package under Julia 0.7+ using `Pkg`, simply enter its interactive mode by pressing `]` in the Julia REPL and run
```
add https://github.com/istvankleijn/AdaptiveABC.jl
```
After this, your Julia install will track AdaptiveABC's master branch. Installing updates works the same way as for registered packages, i.e. `update` in the Pkg interactive mode.
See also https://docs.julialang.org/en/v1.0/stdlib/Pkg/ for more details.

#### Most important changes in this fork

Although the file structure has changed from https://github.com/aifbowman/ABC-SMC, the original `APMC()` function still houses the ABC-APMC algorithm and is called using the same arguments. However, it now returns a `APMCResult` type, with rather different field names from before. See https://github.com/istvankleijn/AdaptiveABC.jl/blob/master/src/types.jl for the type definitions.

Furthermore, the package includes a wrapper `modelselection()` function. Currently, this takes an `APMCInput` type that differs primarily from the input arguments to `APMC()` in requiring separate data-generating functions (simulators) and distance functions (metrics). In future updates, I aim to make the `modelselection()` interface more general (e.g. allowing for rejection-ABC only, or continuing a previous result) and add a `parameterinference()` wrapper.
