# Cell Type Mapping Documentation

本文档记录了各数据集中的细胞类型及其标准化映射关系。

## 数据集细胞类型列表

### HumanPBMC.h5ad
- B cell
- CD4 T cell
- CD8 T cell
- Monocyte_CD14
- Monocyte_FCGR3A
- NK cell
- Megakaryocyte
- Plasmacytoid dendritic cell
- Hematopoietic stem cell

### Immune.h5ad
- CD4+ T cells
- CD8+ T cells
- CD10+ B cells
- CD14+ Monocytes
- CD16+ Monocytes
- NKT cells
- NK cells
- Plasma cells
- Plasmacytoid dendritic cells
- Monocyte-derived dendritic cells
- Erythroid progenitors
- Megakaryocyte progenitors
- HSPCs
- CD34+ progenitors
- Granulocytes
- Mast cells

### Myeloid.h5ad
- Macro_C1QC, Macro_FN1, Macro_GPNMB, Macro_IL1B, Macro_SPP1, Macro_ISG15
- Macro_C1QC-like, Macro_MHCII, Macro_CCL18, Macro_APOE, Macro_VEGFA, Macro_M2
- Mono_CD14
- cDC2_CD1C, cDC2_FCN1, cDC2_CXCL9, cDC2_CD1A, cDC2_IL1B, cDC2_ISG15
- cDC3_LAMP3
- pDC_LILRA4

### Lung.h5ad
- Type 1, Type 2
- Ciliated, Secretory
- Endothelium, Lymphatic
- Fibroblast
- Macrophages, Mast cell
- NK cell, T cell, B cell
- Transformed epithelium

### Liver.h5ad
- Hepatocyte_1, Hepatocyte_2, Hepatocyte_3, Hepatocyte_4
- Cholangiocytes
- Hepatic_Stellate_Cells
- Central_venous_LSECs, Portal_endothelial_Cells
- Inflammatory_Macrophage, Non-inflammatory_Macrophage
- Erythroid_Cells
- alpha-beta_T_Cells, gamma-delta_T_Cells_1, gamma-delta_T_Cells_2

### Heart.h5ad
- fibroblast
- endothelial cell, artery endothelial cell, vein endothelial cell, endothelial cell of artery
- smooth muscle cell
- pericyte
- endocardial cell
- mesothelial cell of epicardium
- cardiac mesenchymal cell
- innate lymphoid cell
- neuron
- adipocyte
- unknown

### Skin.h5ad
- T cell
- macrophage
- erythrocyte
- pericyte
- skin fibroblast
- keratinocyte
- stem cell of epidermis
- melanocyte
- endothelial cell of vascular tree
- endothelial cell of lymphatic vessel

## 标准化细胞类型汇总

| 标准化类型                          | 典型来源数据集                           |
| ----------------------------------- | ---------------------------------------- |
| T cell                              | PBMC, Immune, Liver, Lung, Skin, Heart   |
| B cell                              | PBMC, Immune, Lung                       |
| NK cell                             | PBMC, Immune, Lung                       |
| NKT cell                            | Immune                                   |
| Monocyte                            | PBMC, Immune, Myeloid                    |
| Dendritic cell                      | PBMC, Immune, Myeloid                    |
| Macrophage                          | Myeloid, Lung, Liver, Skin               |
| Plasma cell                         | Immune                                   |
| Mast cell                           | Immune, Lung                             |
| Erythroid progenitor                | Immune, Liver                            |
| Erythrocyte                         | Skin                                     |
| Megakaryocyte                       | PBMC, Immune                             |
| Granulocyte                         | Immune                                   |
| HSPC                                | PBMC, Immune                             |
| Fibroblast                          | Lung, Heart, Skin                        |
| Endothelial cell                    | Lung, Liver, Heart, Skin                 |
| Lymphatic endothelial cell          | Lung, Skin                               |
| Pericyte                            | Heart, Skin                              |
| Smooth muscle cell                  | Heart                                    |
| Hepatocyte                          | Liver                                    |
| Cholangiocyte                       | Liver                                    |
| Hepatic stellate cell               | Liver                                    |
| Alveolar type I cell                | Lung                                     |
| Alveolar type II cell               | Lung                                     |
| Ciliated cell                       | Lung                                     |
| Secretory airway cell               | Lung                                     |
| Epithelial cell                     | Lung                                     |
| Keratinocyte                        | Skin                                     |
| Epidermal stem cell                 | Skin                                     |
| Melanocyte                          | Skin                                     |
| Endocardial cell                    | Heart                                    |
| Mesothelial cell                    | Heart                                    |
| Mesenchymal cell                    | Heart                                    |
| Innate lymphoid cell                | Heart                                    |
| Neuron                              | Heart                                    |
| Adipocyte                           | Heart                                    |
| Unknown                             | Heart                                    |

## 细胞类型映射规则

映射规则详见 `cell_type_map.json` 文件。

### 映射原则
1. **免疫细胞统一化**: 将不同标注的同类免疫细胞统一到标准类型（如 CD4+ T cells → T cell）
2. **亚型合并**: 将功能相似的亚型合并到主类型（如 Macro_C1QC, Macro_FN1 → Macrophage）
3. **组织特异性保留**: 保留组织特异性细胞类型的详细信息（如 Hepatocyte, Keratinocyte）
4. **内皮细胞细分**: 区分普通内皮细胞和淋巴管内皮细胞
5. **祖细胞标注**: 明确标注祖细胞和干细胞类型（如 HSPC, Erythroid progenitor）

### 主要映射类别

#### 免疫细胞
- T cells (CD4+, CD8+, alpha-beta, gamma-delta) → T cell
- B cells (CD10+) → B cell
- Monocytes (CD14+, CD16+) → Monocyte
- Dendritic cells (cDC, pDC, monocyte-derived) → Dendritic cell
- Macrophages (各种亚型) → Macrophage

#### 组织特异性细胞
- 肝脏: Hepatocyte, Cholangiocyte, Hepatic stellate cell
- 肺: Alveolar type I/II cell, Ciliated cell, Secretory airway cell
- 心脏: Smooth muscle cell, Endocardial cell, Mesothelial cell
- 皮肤: Keratinocyte, Melanocyte, Epidermal stem cell

#### 基质细胞
- Fibroblast, Pericyte, Mesenchymal cell
- Endothelial cell, Lymphatic endothelial cell

## 使用说明

1. 数据处理时使用 `cell_type_map.json` 进行细胞类型标准化
2. 标准化后的类型用于模型训练和评估
3. 保持映射规则的一致性，确保跨数据集的可比性
