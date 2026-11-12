# Dataset plan and STEP 3 results

On 2026-11-11, after explicit approval, three Roboflow YOLOv8 archives were downloaded.
Their combined compressed size was 1,447,693,119 bytes (1.448 decimal GB), below the
60 GB public-data budget. The project E: drive had 269.02 GiB free immediately before
resuming the interrupted fruit archive.

The selected exports contained 26,168 labeled images. A 64-bit perceptual hash clustered
near-duplicates at Hamming distance <= 6; 630 images were removed, leaving 25,538
representatives. With seed 42, the merged dataset contains 20,430 images in
[`data/public/train`](./data/public/train) and 5,108 in [`data/public/val`](./data/public/val).
Only these two destination splits were written; no public test or stress-test data was
created. Each retained hash cluster has one representative, so near-copies cannot cross
the train/val boundary.

The project taxonomy is the 97-class [`configs/classes.yaml`](./configs/classes.yaml).
The public-training config is generated from that taxonomy at
[`data/public/data.yaml`](./data/public/data.yaml); it uses an absolute dataset path and
defines only train and val.
The largest class has 2,192 images, so the 2,500-image-per-class cap did not remove any
additional images. `data/metadata.csv` contains one row per retained image and its source,
scene-type status, and cluster ID. Source scene types are mostly unknown, so per-scene
performance cannot be assessed from these exports.

The fruit export included YOLO polygon annotations. 60,003 polygon rows were converted to
bounding boxes; 39,904 same-class box annotations with >=0.9 intersection-over-smaller
area against a polygon were suppressed as likely duplicate annotations. 660 tiny
rounding-overflow boxes were clipped to image bounds, one degenerate box was dropped, and
two unusually large boxes were flagged. The complete per-class counts are in
[`runs/step3_merge_report.txt`](./runs/step3_merge_report.txt); the 40 rendered examples
and geometry flags are in [`runs/label_check`](./runs/label_check). Spot-checking four
renders showed plausible radish, banana, and capsicum boxes; the note about a dish that may
contain prepared food is in
[`manual_review.csv`](./runs/label_check/manual_review.csv). This small visual spot check
does not establish general label quality or raw-scene coverage.

## Approved Roboflow source preflight (2026-11-10)

The authenticated Roboflow API returned each project's `license` field. Existing YOLOv8
exports were checked via metadata and HTTP HEAD response headers; no archive body was
downloaded. This yielded exact compressed archive sizes for the selected versions.
Source/version listing counts are not unique-image counts. Full results from the
metadata-only API check are in
[`runs/step3_api_dry_run.txt`](./runs/step3_api_dry_run.txt).

| Selected source/version | Listed images | Generation augmentation | API licence | YOLOv8 ZIP size |
|---|---:|---|---|---:|
| [deep-food-project / food-detection-3 / v1](https://universe.roboflow.com/deep-food-project/food-detection-3/dataset/1) | 10,000 | None reported | CC BY 4.0 | 577,574,965 bytes (0.578 GB) |
| [fruiiiiit / fruits-hriyf / v2](https://universe.roboflow.com/fruiiiiit/fruits-hriyf/dataset/2) | 13,284 | None reported | CC BY 4.0 | 837,247,969 bytes (0.837 GB) |
| [vegetable-object-detection / vegetables-kacga-lvr1d / v1](https://universe.roboflow.com/vegetable-object-detection/vegetables-kacga-lvr1d/dataset/1) | 2,884 | None reported | CC BY 4.0 | 32,870,185 bytes (0.033 GB) |
| **Total** | **26,168 listed** | | | **1,447,693,119 bytes (1.448 GB)** |

GB above is decimal (1 GB = 1,000,000,000 bytes). ZIP sizes were obtained from the Roboflow
export metadata and confirmed with HTTP HEAD `Content-Length`, without fetching archive
bodies during preflight. The three ZIPs are below the 60 GB public-data budget.

The three-source recommendation favors versions with no generation augmentations and
combines broad produce/food classes, fruit classes, and vegetables. Scene diversity,
unique-image counts, and most annotation quality remain unknown after initial checks. This
is a source pool, not evidence that the data spans fridge, counter, market, basket, bag,
and checkout scenes.

Other API-checked licences: ece4078, bohni-tech, neeraj-chidella, fridgeitems, project-yijmw,
and vegetable-object-detection report CC BY 4.0; AICook reports MIT but has no versions
available through the API. Those sources are not in the proposed download set. AICook
remains excluded unless an exact downloadable version is verified.

The user explicitly approved these exact three archives before download. No other dataset
source was downloaded.

## Roboflow metadata dry run

| Workspace / project / version | API-reported images and splits | Classes | Augmentation | Licence | Scene type |
|---|---|---|---|---|---|---|
| [deep-food-project / food-detection-3 / v1](https://universe.roboflow.com/deep-food-project/food-detection-3/dataset/1) | 10,000 (train 7,000 / valid 2,000 / test 1,000) | 26 classes in the downloaded v1 export: apple, artichoke, banana, bell pepper, broccoli, cabbage, cantaloupe, carrot, cheese, common fig, cucumber, grape, grapefruit, lemon, mango, mushroom, orange, peach, pear, pomegranate, potato, pumpkin, radish, strawberry, tomato, zucchini. Project metadata listed 36 class entries. | `{}` | CC BY 4.0 | To verify |
| [ece4078-ss3zf / ece4078-object-detection-fruits / v5](https://universe.roboflow.com/ece4078-ss3zf/ece4078-object-detection-fruits/dataset/5) | 8,860 (train 8,176 / valid 518 / test 166) | 4: lime, pumpkin, orange, tomato | Image versions 3; brightness +/-15%; horizontal flip | CC BY 4.0 | To verify |
| [bohni-tech / fruits-and-vegi / v13](https://universe.roboflow.com/bohni-tech/fruits-and-vegi/dataset/13) | 6,805 (train 6,645 / valid 150 / test 10) | 35: carrot, cherry, peach, coconut, mango, maize, ginger, bottle gourd, tomato, bitter gourd, orange, okra, potato, strawberry, cauliflower, pineapple, radish, chilli, cabbage, pomegranate, beetroot, green orange, peas, pear, onion, watermelon, banana, apple, kiwi, eggplant, grape, turnip, capsicum, cucumber, melon | Image versions 3; darken brightness (40%); crop (up to 30%) | CC BY 4.0 | To verify |
| [neeraj-chidella / fruits-qhsig / v2](https://universe.roboflow.com/neeraj-chidella/fruits-qhsig/dataset/2) | 6,233 (train 6,082 / valid 102 / test 49) | 14: fruit, watermelon, strawberry, potato, pomegranate, pineapple, orange, mushroom, mango, grape, carrot, cabbage, banana, apple | Image versions 3; 90-degree rotations; horizontal flip | CC BY 4.0 | To verify |
| [objectdetectionlearning-4empo / aicook-lcv4d-zzbgs](https://universe.roboflow.com/objectdetectionlearning-4empo/aicook-lcv4d-zzbgs) | User-reported 516 originals; API returned no versions | 30 fridge/food classes, including apple, spinach, chicken, shrimp, onion, corn, blueberry, beef, tomato, cheese, potato, lime, and carrot | Version unavailable | MIT | Single fridge (user-reported) |
| [fridgeitems / object-fridge-items](https://universe.roboflow.com/fridgeitems/object-fridge-items) | v1: 7,407 (train 7,077 / valid 233 / test 97) | 89 class entries; mixed produce and packaged food | Image versions 3; horizontal flip; crop (up to 20%) | CC BY 4.0 | Fridge |
| [ai-workspace-q8pyt / fridge-fine-tune](https://universe.roboflow.com/ai-workspace-q8pyt/fridge-fine-tune) | No version selected; to verify | 43 class entries; mixed produce and packaged food | Version metadata not queried | To verify | Fridge |
| [project-yijmw / vegetables_obj_det-r78fa / v2](https://universe.roboflow.com/project-yijmw/vegetables_obj_det-r78fa) | 150 (train only) | 22 class entries, including papaya, cauliflower, bottle gourd, bitter gourd, brinjal, capsicum, radish, broccoli, tomato, potato, pumpkin, bean, carrot, and numeric labels 1-8 | `{}` | CC BY 4.0 | To verify |
| [vegetable-object-detection / vegetables-kacga-lvr1d / v1](https://universe.roboflow.com/vegetable-object-detection/vegetables-kacga-lvr1d) | 2,884 (train only) | 26 class entries, including cabbage, cucumber, corn, onion, broccoli, tomato, bell pepper, celery, peas, potato, beet, eggplant, garlic, avocado, pumpkin, carrot | `{}` | CC BY 4.0 | To verify |
| [fruiiiiit / fruits-hriyf / v2](https://universe.roboflow.com/fruiiiiit/fruits-hriyf) | 13,284 (train 9,299 / valid 2,656 / test 1,329) | 9: banana, lemon, peach, grapes, strawberry, watermelon, pear, orange, apple | `{}` | CC BY 4.0 | To verify |
| [merlams24 / fruits-zr4m9 / v3](https://universe.roboflow.com/merlams24/fruits-zr4m9/dataset/3) | 7,357 (train 5,150 / valid 1,473 / test 734) | 17: guava, papaya, watermelon, mango, rambutan, pomelo, orange, lemon, tambis, banana, apple, star fruit, jackfruit, durian, tomatoes, santol, pomegranate | `{}` | CC BY 4.0 | To verify |

`aicook` v1 was not found by the API; do not silently substitute another version. For
projects with no version specified, the updated dry-run CLI can list available versions
but does not choose one. Reported image counts do not establish unique-image counts.
Scene types remain unverified. Licences were retrieved from the Roboflow API project
metadata; see the tables above. The initial metadata-only dry run had no local labels to
audit; after approval, all three selected exports were downloaded, inspected, and merged.
Mapped image/instance counts and the post-dedup class totals are in
[`runs/step3_merge_report.txt`](./runs/step3_merge_report.txt). The downloaded archive
sizes are exact for the selected versions; uncompressed sizes are represented by the
measured extracted files, and free space was rechecked before resuming the interrupted
archive.

## Other sources

| Source / version | Classes | Listing count and unique-image estimate | Augmentation | Licence | Scene type | Proposed role |
|---|---|---|---|---|---|---|
| [Open Images V7](https://storage.googleapis.com/openimages/web/index.html), accessed through [FiftyOne](https://docs.voxel51.com/user_guide/dataset_zoo/datasets.html) | Select only supported produce classes; exact class/box coverage to verify before requesting; cap at about 1,500 images per class | No total estimate; apply per-class cap and measure unique IDs after selection | Not an augmented export; source image duplication still to be checked | Image-level terms/attribution must be checked for selected images; annotation licence to verify | General scenes; per-image scene taxonomy to verify | General detector pretraining; never request the full dataset |
| Bangladesh vegetable detection dataset, Mendeley, Pascal VOC (identity not yet confirmed) | To verify | About 3,500 reported by user; unique count and exact record to verify | To verify | To verify | To verify | Detector pretraining after VOC label audit and conversion |
| [FRUVEG67 paper](https://arxiv.org/abs/2409.13330) | 67 fruit/vegetable categories; exact class list to verify | Paper reports 5,000 collected images, around 2,000 manually annotated; unique count to verify | Semi-supervised boxes for remaining images described in paper | Dataset licence to verify | Unconstrained environments described in paper | Report only for now; no public data location verified from the paper reviewed |
| [VegNet: Vegetable Dataset with quality](https://data.mendeley.com/datasets/6nxnjbn9w6) | Bell Pepper, Tomato, Chili Pepper, New Mexico Chile; quality folders | Mendeley/DataCite record reports 6,850 images; unique count to verify | Record does not establish whether source images are augmented; to verify | CC BY 4.0 in DataCite metadata | To verify | Image-level crop-classifier pretraining only; not detection labels |
| [Synthetic Fruit repository](https://github.com/brad-dwyer/synthetic-fruit) | To verify | To verify; repository URL returned 404 during review | Synthetic; details to verify | To verify | Synthetic scenes | Optional pretraining only if the source, licence, and data are verified; never validation/test |
| FoodSeg103 | Cooked dishes / food segmentation | Not considered | N/A | N/A | Cooked dishes | Excluded from v1 |

The Mendeley record found for VegNet is a plausible match for the requested Indian
image-level vegetable dataset, not a confirmed match to the user's intended record.
The Bangladesh detection record could not be matched by title/format in the public
metadata search; identify its exact Mendeley URL before any use.

## Approved class alias decisions

| Surface names | Proposed canonical class | Proposal |
|---|---|---|
| capsicum, bell pepper, sweet pepper | `capsicum` | Merge spelling/regional variants |
| brinjal, eggplant, aubergine | `brinjal` | Merge regional/common names |
| coriander, cilantro | `coriander` | Merge names for the fresh herb |
| lemon vs lime | `lemon` and `lime` | Keep as distinct classes |
| onion vs shallot | `onion` and `shallot` | Keep as distinct classes |
| green chilli vs red chilli | `green_chilli`, `red_chilli` | Keep green and red separate; unqualified chilli maps to `chilli` |
| sweet corn, sweetcorn | `sweet_corn` | Merge; generic corn/maize maps to separate `corn` |
| `other_food` | `other_food` | Catch-all for mapped food boxes outside the named taxonomy; drop non-food boxes |

The user approved the alias decisions above, then requested further fruit and vegetable
classes. The taxonomy also includes eight raw protein classes (chicken, beef, pork, lamb,
goat meat, fish, shrimp, and prawn). `mutton` is intentionally not an alias because usage
varies by region. The current taxonomy has 97 classes. Only the three explicitly approved
public archives were downloaded; no model training was run.
