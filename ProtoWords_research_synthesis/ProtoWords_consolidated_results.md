# ProtoWords — Consolidated Research Results (E1–E14)

**Project:** ProtoWords: Discrete Foundation-Feature Prototypes for Data-Efficient and OOD-Robust Recognition  
**Dataset used for the main classification study:** CIFAR-10  
**Feature extractor:** Frozen DINOv2 ViT-S/14, global CLS representation (384 dimensions)  
**Development split:** 45,000 training + 5,000 validation images; official test: 10,000 images  
**Evidence source:** The uploaded `results.zip` and `outputs(2).zip` archives, inspected experiment summaries and result tables.

---

## 1. Executive finding

The experiments support ProtoWords as a **compact, reasonably class-aligned representation of frozen visual foundation-model features**. The strongest evidence is the storage–accuracy trade-off, not an improvement over the continuous DINOv2 representation.

The continuous DINOv2 + Logistic Regression baseline achieves **97.61%** test accuracy. The most useful compact variants are:

| Representation | Test accuracy | Difference vs baseline | Estimated total storage/image | Storage reduction |
|---|---:|---:|---:|---:|
| Continuous DINOv2 features | 97.61% | Reference | 1,536.31 B | — |
| K=128, UInt8 distances | 97.27% | −0.34 pp | 132.11 B | 91.40% |
| K=256, UInt8 distances | 97.35% | −0.26 pp | 264.21 B | 82.80% |
| K=256, Float16 distances | 97.36% | −0.25 pp | 520.15 B | 66.14% |
| K=256, Float32 distances | 97.40% | −0.21 pp | 1,032.15 B | 32.82% |

**Best compression:** K=128 UInt8, with 91.40% lower estimated per-image storage and a 0.34 percentage-point lower test accuracy.  
**Balanced compact option:** K=256 UInt8, with 82.80% lower storage and a 0.26 percentage-point lower test accuracy.

The loss relative to the continuous baseline is statistically significant for all four E13 quantized configurations under the paired test in E14, even though the absolute gaps are small. Therefore, the project should not claim that ProtoWords beats the continuous baseline or is statistically equivalent to it.

## 2. Research question and defensible contribution

A more precise central research question is:

> Can a shared discrete prototype vocabulary compress frozen visual foundation-model features into compact, interpretable representations while retaining classification performance close to the original continuous features?

A defensible contribution statement is:

> We study shared prototype vocabularies as a compact representation layer over frozen DINOv2 features. Across CIFAR-10 experiments, distance-to-prototype representations preserve high classification accuracy while reducing estimated per-image representation storage by up to 91.4%. The learned codewords exhibit high class purity, although exact cluster assignments vary across random initializations. Robustness and OOD experiments show promising dataset-specific behavior but do not establish general superiority over continuous-feature baselines.

The phrase **“data-efficient”** should be defined carefully: the results directly demonstrate representation-storage efficiency, not improved sample efficiency from using fewer labelled training examples.

## 3. Main clean-classification results

### 3.1 Official test-set results

| Experiment | Method | Test accuracy | Test macro-F1 | Difference vs baseline |
|---|---|---:|---:|---:|
| E1 | Continuous DINOv2 + LR | 97.61% | 97.61% | 0.00 pp |
| E2 | Class-wise Euclidean prototypes, 80 total | 96.13% | 96.13% | −1.48 pp |
| E2 | Class-wise cosine prototypes, 80 total | 95.93% | 95.93% | −1.68 pp |
| E6 | Global distance representation, K=256 | 97.41% | 97.41% | −0.20 pp |
| E6 | Local soft assignment, K=256, m=8 | 96.90% | 96.91% | −0.71 pp |
| E6 | Hybrid distance + soft representation | 97.50% | 97.50% | −0.11 pp |
| E13 | K=128 UInt8 distances | 97.27% | 97.27% | −0.34 pp |
| E13 | K=256 UInt8 distances | 97.35% | 97.35% | −0.26 pp |
| E13 | K=256 Float16 distances | 97.36% | 97.36% | −0.25 pp |
| E13 | K=256 Float32 distances | 97.40% | 97.40% | −0.21 pp |

The E5 hybrid configuration reached 97.84% validation accuracy, slightly above the 97.72% validation baseline. However, E6 test accuracy for that selected hybrid was 97.50%, below the 97.61% baseline. The validation gain must not be presented as a final test improvement.

### 3.2 Validation-only findings must remain separate

- E3 shared global vocabulary: K=256 distance representation reached 97.74% validation accuracy. E3 did not evaluate on the official test set.
- E4 localized soft assignment: best validation accuracy was 97.48%, below the continuous validation baseline of 97.72%.
- E5 hybrid: best validation accuracy was 97.84%; final test results are reported in E6.
- E10 seed analysis: hybrid validation accuracy averaged 97.74% ± 0.09% across five K-Means seeds, with the continuous validation baseline at 97.72%. This was not a five-seed official-test evaluation.
- E11 K=256 distance validation accuracy averaged 97.668% ± 0.039% across five seeds. Again, this is validation evidence, not official test evidence.
- E12 quantization: K=256 Float32 achieved 97.74% validation accuracy; K=256 UInt8 achieved 97.62%. Final test results are in E13.

## 4. Statistical reliability of the compact representations

E14 compared the E13 per-image predictions against the continuous baseline on the same 10,000 official test images. It used 10,000 paired bootstrap resamples for 95% confidence intervals, exact two-sided McNemar tests, and Holm correction across four comparisons.

| Method | Accuracy difference | Paired 95% CI | McNemar p | Holm-adjusted p |
|---|---:|---:|---:|---:|
| K=128 UInt8 | −0.34 pp | [−0.53, −0.15] | 0.000770 | 0.003080 |
| K=256 UInt8 | −0.26 pp | [−0.44, −0.08] | 0.005436 | 0.012195 |
| K=256 Float16 | −0.25 pp | [−0.42, −0.09] | 0.004065 | 0.012195 |
| K=256 Float32 | −0.21 pp | [−0.38, −0.05] | 0.016995 | 0.016995 |

All four differences are significant after Holm correction at α=0.05. The confidence intervals are entirely below zero. This establishes a reliable small performance cost for these fixed models on this test set; it does not establish a practically large loss, nor does it replace repeated training across independent data splits.

For K=256 UInt8, the paired correctness table contained 9,707 images correctly classified by both methods, 54 correct only for the continuous model, 28 correct only for ProtoWords, and 211 incorrect for both. The largest classwise drop for K=256 UInt8 was on `cat` (-0.7 pp).

## 5. Prototype interpretability and stability

For the K=256 codebook in E9:

- All 256 prototypes were occupied.
- Cluster sizes ranged from 63 to 384 images.
- Mean cluster purity was **95.58%**.
- Mean cluster entropy was **0.221 bits**.
- The experiment summary reports 105 perfectly pure prototypes and 223 prototypes with at least 90% purity.

Across five codebook seeds in E10, pairwise assignment agreement was:

- Adjusted Rand Index (ARI): **0.5044 ± 0.0096**.
- Normalized Mutual Information (NMI): **0.8011 ± 0.0043**.
- Mean cluster purity: **95.61% ± 0.16%**.

E11 shows the same trend across vocabulary sizes:

| K | Mean purity | Mean pairwise ARI | Mean pairwise NMI | Mean distance validation accuracy |
|---:|---:|---:|---:|---:|
| 32 | 94.50% | 0.741 | 0.871 | 97.04% |
| 64 | 94.99% | 0.702 | 0.859 | 97.40% |
| 128 | 95.11% | 0.627 | 0.836 | 97.50% |
| 256 | 95.62% | 0.504 | 0.802 | 97.67% |

Interpretation: higher K increases average class purity and validation accuracy, but exact point-to-prototype partitions become less consistent across seeds. The high NMI alongside moderate ARI suggests broad semantic groupings are more consistent than exact cluster assignments. Do not describe the codebook as perfectly stable.

The saved representative-image contact sheet and prototype-level CSV can be used as qualitative interpretation material. Interpretability should be presented as class alignment and inspectable prototypes, not as a claim that every prototype has a unique human-readable semantic meaning.

## 6. Corruption robustness (CIFAR-10-C)

E7 evaluated 15 corruption types at five severity levels, for 75 corruption conditions. Results are mean accuracy over these conditions:

| Method | Mean corruption accuracy | Mean macro-F1 | Drop from clean accuracy |
|---|---:|---:|---:|
| Continuous DINOv2 | 84.17% | 84.21% | 13.44 pp |
| Distance ProtoWords | 83.87% | 83.92% | 13.54 pp |
| Hybrid ProtoWords | 83.91% | 83.96% | 13.59 pp |
| Soft ProtoWords | 81.79% | 81.74% | 15.11 pp |

Distance and hybrid representations are close to the continuous baseline on the aggregate benchmark, while soft assignment is weaker. The continuous baseline remains best overall. The results do not support a claim that ProtoWords improves general corruption robustness.

## 7. Out-of-distribution detection

E8 evaluated CIFAR-100, SVHN and Tiny ImageNet as OOD datasets. Thresholds were calibrated from 5,000 in-distribution CIFAR-10 images to target 95% ID retention; OOD data were not used to choose thresholds.

| Method | Mean AUROC | Mean OOD FPR at calibrated threshold |
|---|---:|---:|
| Continuous energy | **0.9736** | **0.0982** |
| Continuous MSP | 0.9624 | 0.1528 |
| Distance confidence | 0.9580 | 0.1689 |
| Nearest codebook distance | 0.9561 | 0.2289 |
| Hybrid confidence | 0.9536 | 0.2030 |
| Soft confidence | 0.9262 | 0.3013 |

Continuous energy is the strongest method averaged over the three OOD datasets. The nearest-codebook-distance score is especially effective on Tiny ImageNet (AUROC 0.9827), but weaker on SVHN (0.9259). It should be described as a potentially useful dataset-dependent OOD signal, not a generally superior detector.

## 8. Representation storage and efficiency

Storage values are estimates of per-image feature payload plus an amortized share of codebook/scaler/classifier arrays, using 50,000 development images. They exclude the DINOv2 backbone and Python/joblib serialization overhead.

The most defensible storage claims are the E13 final-test estimates:

- K=128 UInt8: 132.11 bytes/image; 91.40% reduction.
- K=256 UInt8: 264.21 bytes/image; 82.80% reduction.
- K=256 Float16: 520.15 bytes/image; 66.14% reduction.
- K=256 Float32: 1,032.15 bytes/image; 32.82% reduction.

These numbers refer to the stored representation and amortized learned arrays, **not the complete deployed system**, because the frozen DINOv2 backbone is excluded. A deployment still needs the feature extractor unless it receives DINOv2 features from another component.

The CPU timing values in E9/E11 are not suitable for a headline speedup claim: measurement scopes differ, some timings exclude classifier prediction, and DINOv2 feature extraction is excluded. Treat latency as unestablished until a standardized end-to-end benchmark is run.

## 9. Recommended conclusion for the report/paper

> ProtoWords provides a compact prototype-distance representation for frozen DINOv2 features. On CIFAR-10, UInt8 distance vectors reduce estimated per-image representation storage by 82.8–91.4% while retaining 97.27–97.35% test accuracy, compared with 97.61% for the continuous-feature baseline. Paired statistical tests show that the remaining accuracy gaps are small but statistically significant on the fixed test set. The shared codebook is strongly class-aligned, with approximately 95.6% mean cluster purity, although exact assignments show only moderate seed agreement. Corruption and OOD evaluations reveal competitive or dataset-specific behavior rather than a consistent robustness advantage. Overall, the results support ProtoWords as a storage-efficient and interpretable representation layer, not as a more accurate replacement for continuous foundation-model features.

## 10. Limitations that should be stated explicitly

1. The main classification evidence is from one dataset (CIFAR-10) and one frozen backbone (DINOv2 ViT-S/14 CLS features).
2. The final reported compact variants use a fixed development split and seed for the official test evaluation. Repeated codebook seeds on one train/validation split do not establish robustness across data splits or independent training runs.
3. E14 tests the fixed prediction vectors on one test set. Its confidence intervals and p-values do not replace multi-seed or multi-dataset evaluation.
4. Storage estimates omit the DINOv2 backbone, serialization overhead, and system-level costs. The representation storage saving is not equal to total deployment memory or compute saving.
5. OOD and corruption results are mixed. No general robustness superiority over continuous DINOv2 is demonstrated.
6. The study focuses on global CLS features; it does not establish behavior for patch tokens, other backbones, or other image domains.
7. Class purity is measured using CIFAR-10 labels. It demonstrates label alignment, but is not by itself a complete human interpretability evaluation.

## 11. Recommended project title

The original title includes “OOD-Robust Recognition,” which may imply a robustness gain that the aggregate E7/E8 results do not show. A safer title is:

**ProtoWords: Compact and Interpretable Prototype Representations for Frozen Visual Foundation Features**

OOD and corruption robustness can remain evaluation dimensions in the paper, rather than promised outcomes in the title.

## 12. Suggested paper/report structure

1. **Introduction:** motivation for compacting frozen foundation-model features; define the representation-storage problem.
2. **Related Work:** visual vocabulary/vector quantization, prototype learning, post-hoc feature compression, OOD detection with distance scores.
3. **Method:** frozen DINOv2 extraction; global K-Means codebook; hard assignment, distance vectors, localized soft assignment, hybrid representation; affine UInt8 quantization.
4. **Experimental Setup:** CIFAR-10 split, validation-only model selection, final test protocol, compute environment, baselines, metrics, storage accounting.
5. **Main Results:** clean test accuracy and compression trade-off; E14 paired statistical analysis.
6. **Prototype Analysis:** cluster purity, entropy, visual examples, seed stability and K-size trade-offs.
7. **Robustness and OOD Evaluation:** CIFAR-10-C and OOD datasets; clearly report mixed findings.
8. **Limitations and Discussion:** backbone/dataset scope, excluded backbone storage, fixed splits and test-set inference.
9. **Conclusion:** compact representation contribution, without claiming higher accuracy or universal OOD robustness.

## 13. Next actions

The next productive step is to write the paper/report from this evidence, not to add another loosely motivated experiment. Before submission, the highest-value additions would be: (a) a related-work and novelty check against vector quantization/prototype compression literature; (b) repeated end-to-end results on at least one additional dataset or backbone if compute permits; and (c) a standardized storage/latency benchmark that includes the actual inference pipeline and clearly separates backbone cost from representation cost.

---

## Included files

- `clean_test_comparison.csv` — test metrics for all models with official CIFAR-10 test results.
- `experiment_ledger.csv` — E1–E14 overview with validation/test status kept separate.
- `figures/clean_test_accuracy.png`
- `figures/compression_accuracy_tradeoff.png`
- `figures/corruption_robustness.png`
- `figures/ood_auroc_by_dataset.png`
- `figures/prototype_purity_stability.png`
