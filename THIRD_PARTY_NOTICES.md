# Third-party notices

This repository bundles selected third-party source code to make the released DigitalOrg research pipeline easier to reproduce. Third-party components retain their original licenses and are not relicensed by the DigitalOrg project license.

## `third_party/sam3_repo`

This directory contains SAM-family source code distributed with a Meta SAM License file.

License file:

```text
third_party/sam3_repo/LICENSE
```

Users must review and comply with that license before using, modifying, or redistributing this component.

## `third_party/digitalorgdet_backend/ultralytics`

This directory contains Ultralytics-derived backend code. Source files are marked with the Ultralytics AGPL-3.0 license notice.

License file added for redistribution clarity:

```text
third_party/digitalorgdet_backend/LICENSE-AGPL-3.0.txt
```

Users must review and comply with the AGPL-3.0 terms and the upstream Ultralytics license information before using, modifying, or redistributing this component.

## Checkpoints and datasets

No model checkpoints, raw microscopy datasets, patient data, or private annotations are included in this repository. Configure local paths through `configs/default.yaml` and environment variables.
