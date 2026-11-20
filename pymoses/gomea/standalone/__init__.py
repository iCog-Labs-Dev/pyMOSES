"""NOT A PORT: the IT linkage study (BACKLOG 13.9).

A stand-alone GOMEA harness with no MOSES loop (``docs/gomea/IT-LINKAGE-STUDY-PLAN.md``,
sections 4 to 6). It runs ``pymoses.gomea.engine.GomeaEngine`` on two fixed-length genomes:

* ``genome_template``: GP-GOMEA's tree template (a full binary tree, height 4, 31 positions,
  the functions ``+ - * aq``, no constants, linear scaling a + b f).
* ``genome_it``: the IT form (8 term slots, 2 factors per term, an input box and a power box
  per factor, 32 positions, per-term least squares).

Each genome runs under the five FOS modes of ``pymoses.gomea.fos.LinkageModel``. Start a run
with ``python -m pymoses.gomea.standalone`` (``run.py``). Nothing in MOSES imports this package,
so no MOSES output changes.
"""
