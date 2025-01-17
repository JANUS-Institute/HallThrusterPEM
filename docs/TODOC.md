Theory:
- Maybe some high-level summaries of the details in the journal paper (i.e. the 1d model, the UQ framework, etc.). Summarize the ideas behind this package, but leave painful details in the journal paper.

Data:
- More details on how we load and use experimental data (i.e. standard classes and comparing with the models)
- High-level summaries and citations of the sources of the data
- How to add your own experimental data

Thruster model:
- thruster_config, simulation_config, and postprocess_config and how they correspond to HallThruster.jl Thruster, Config, and post-processing steps
- Format of thruster device.yml configuration (essentially just match hallthruster.jl Thruster for now)

Tutorial:
- Do this in a jupyter notebook (walk through setup, loading data and models, amisc, etc. -- should take ~1 hour). For someone who is trying out the package themselves, this gives them a quick on-ramp to using it.

Examples:
- Show basic usage of the package -- targeted towards someone interested in the package but is making sure it is useful to them. Should emphasize the modularity and plug-and-play. Someone with basic familiarity of the package could also use this to jump-start their own projects.

How-to Guides:
- These are specific step-by-step instructions of how to do certain things. Focused on the "how" rather than the "why" or "what". Someone reading this is just interested in reaching an end goal and needs help on the logistics of getting there. They are certainly already somewhat familiar with the package and has likely already done the tutorial and seen some examples.
- Need to document each of the empty placeholder sections at the very least.
