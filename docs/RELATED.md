# Related work and positioning

To be completed as reading proceeds. The table below is the one that goes in the paper;
every row must be verified against the actual paper, never from memory.

| Work | Year | Approach | Multi-timescale | Continual | Wireless/RIC | Theory | Public benchmark |
|---|---|---|---|---|---|---|---|
| Nested Learning (Behrouz et al., NeurIPS) | 2025 | NL paradigm, CMS, Hope | yes | yes | no | yes | no |
| FedNL | 2026 | nested optimisation for federated LLM training | yes | partial | no | partial | no |
| Sun et al., IEEE TSP 70:1900-1917 | 2022 | bilevel continual wireless resource optimisation | two-level | yes | yes | yes | no |
| Continual Learning for Wireless Channel Prediction | 2025 | EWC/replay for channel prediction | no | yes | yes | no | no |
| ColO-RAN (TMC) | 2022 | DRL xApps on Colosseum | no | no | yes | no | dataset |
| Nagib et al., IEEE JSAC 42(2) | 2024 | transfer learning for O-RAN slicing | no | no | yes | no | no |
| RANPilot | 2026 | robustness to O-RAN reconfiguration (arbitration) | no | partial | yes | no | no |
| **This work** | 2026 | **NL mapped to RIC control loops + CMS** | **yes (L levels)** | **yes** | **yes** | **yes (bound)** | **yes (O-RAN-CL)** |

Verification tasks before submission:
- [ ] Re-run a Google Scholar "cited by" sweep on the Nested Learning paper; confirm no
      wireless application has appeared. Cite any concurrent work explicitly.
- [ ] Read FedNL in full and write one precise paragraph distinguishing this work.
- [ ] Confirm every row above against the actual PDF, including the volume/page numbers.
