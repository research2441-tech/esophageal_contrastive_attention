# Dataset Reporting Notes

The manuscript reports:
- `esophagus`: 1,689 images
- `no-esophagus`: 8,973 images
- total: 10,662 images

This repository preserves those raw labels and does not automatically reinterpret them as cancer/non-cancer.

The following must be completed from the authoritative dataset source before manuscript resubmission: official dataset title, source URL/accession, acquisition provenance, annotation procedure, inclusion/exclusion criteria, de-identification status, ethics/IRB basis, access terms, and any validated diagnostic mapping.

Default split: 70% train, 10% validation, 20% test. Splitting is performed before augmentation. If patient identifiers are available in a manifest, grouped splitting is supported. Exact duplicates are audited with SHA-256.
