"""Legacy LLM-Xavier research library of the independent ``workspace/`` project.

Subpackages: ``datasets`` (raw readers), ``preprocessing`` (prompt construction), ``experiment``
(planning, guarded execution, analysis), ``tools`` (CLI utilities), ``lag`` (OpenAI Batch runners of
experiments 14-17 and a frozen copy of the LAG episode loader). Layout lives in ``paths``. Nothing
here imports the core ``mtd_rl_xavier`` package, and the core never imports this one.
"""
