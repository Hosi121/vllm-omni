# Omni stage contracts

Versioned, dependency-free data contracts for local Omni stage adapters.
The optional `wire` extra provides the NumPy copied-host tensor transport.
Importing the contracts never imports torch, vLLM or vLLM-Omni.

Build from this directory with `uv build`; both the wheel and source archive
contain the canonical package. The parent Omni distribution bundles the same
source for compatibility. Install matching releases when using both distributions.
