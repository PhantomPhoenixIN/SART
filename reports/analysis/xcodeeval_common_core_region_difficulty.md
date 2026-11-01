# Frozen xCodeEval Common-Core per-region structural difficulty

Status: `FROZEN_PREMODEL`; no prediction or execution result is read.

Tasks: 1992; source parse valid: 1922; target parse valid: 1922; matched/source regions: 148694.

Definition: One-to-one minimum-cost source/reference region assignment. Pair cost is the equal-weight mean of normalized semantic region-type mismatch, log AST-depth difference, log child-count difference, and log byte-span-length difference. Unmatched source regions receive cost 1.

Limitation: Region correspondence is a structural proxy, not a semantic translation alignment. It is frozen before model outputs and is used only for retrospective association.

Task-manifest SHA256: `7b58147227e21e26575b7c94635f9a174148ef7ee3c956dd963947ed8d8c912b`

Reference-selection SHA256: `adf3970efe8a6510a7c58a3b4eab01b140c6f726de9429bd7afe74d7f22af287`
