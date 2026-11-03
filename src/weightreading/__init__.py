"""weightreading: a frontier LLM reads an RL policy's raw weights and explains what it computes.

The LLM never runs the network. It gets the architecture as equations, a tensor table and a
sandboxed Python executor with the weights preloaded as ``W``; see ``README.md`` in this package.
"""

SCHEMA_VERSION = 1
