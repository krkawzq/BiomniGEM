from litellm import completion

messages = [
    {"role": "system", "content": """You are a helpful biology expert. You are given a task of assessing whether CRISPRi knockdown of a perturbation gene leads to a statistically significant change in the expression of another gene.

## Task

- **Goal**: Determine if the queried target gene shows a significant expression change after perturbing the specified gene using CRISPRi.  
- **Context**: You will be provided with the following information:
  1. The **background cell line** where the perturbation is applied.  
  2. A list of **highly expressed genes** in this cell line, obtained by averaging single-cell expression profiles across all cells (a bulk-like profile representing the baseline transcriptional state). The list is sorted in descending order of mean expression (genes appearing later have lower expression).  
  3. The **perturbation condition** (which gene is knocked down by CRISPRi).  
  4. The **target gene** whose potential expression change should be assessed.  

## Guidance

- Use your biological knowledge to reason about the potential relationship between the perturbation gene and the target gene.
- The provided gene list reflects the average expression profile of the H1-hESC, serving as a reference for inferring biological pathways and regulatory interactions.
- Knocking down a gene may disrupt its associated pathways, which can in turn affect the expression of the target gene.
- While most genes typically remain unchanged under a given perturbation, in this testing setup the probability has been artificially adjusted—around **50%** of the queries involve target genes that do show significant changes. Therefore, you should carefully evaluate the context, pathway involvement, and regulatory mechanisms before making a decision.
-  You should, based on your biological knowledge, independently determine from which perspective to conduct the analysis."""},
    {"role": "user", "content": """In H1-hESC cells, we apply CRISPRi knockdown of BIRC2.  
Here is the list of highly expressed genes in this background, sorted by mean expression (high → low):  
ACTB, CD24, POU5F1, EEF2, OAZ1, HSPA8, TMSB4X, ACTG1, TUBB2B, LDHB, LIN28A, LAPTM4B, SCD, HNRNPA2B1, CLDN6, DNMT3B, SLC16A1, L1TD1, MIF, HSPD1, SFRP1, FADS2, HSP90AA1, H1-5, HELLS, EEF1G, ENAH, CFL1, MAD2L2, APOE, COXFA4, PODXL, GJA1, MGST1, ITM2C, IGF2BP1, MARCKSL1, HNRNPD, PFN1, YBX1, SEPHS1, SELENOH, SNRPN, NACC1, ENO1, EIF4EBP2, PRDX1, TERF1, TUBB2A, PFN2, AASS, IDH1, SELENOW, DPYSL3, CLIC4, MCM4, NME2, LPCAT1, CTBP2, EIF4A1, PSAT1, NHP2, MCM7, PGRMC1, SFRP2, H3-3A, NT5DC2, NREP, ILF3, MEST, PHC1, NCL, VCAN, HNRNPA0, TMSB10, HNRNPU, SERBP1, APP, PSMB1, KDM1A, TXNDC17, ZNF121, KRT18, MFGE8, MDK, SRSF3, MCM5, LSM4, YWHAE, YWHAZ, ATP5F1B, CNN3, GNL3, DHCR24, CTSC, HSPE1, RBPMS, TUBA1C, PDIA3, FTL, PRDX2, CKS1B, UBE2I, DDX1, MYBL2, NUS1, AKIRIN1, MCL1, PPM1G, MYH10, DPPA4, CCT6A, HMGCS1, ACTN1, BEX3, FLNA, HNRNPL, SLC38A1, JARID2, MSH2, MAPK6, USP9X, CENPV, SNRPC, IGF2BP3, HMGN3, CDK2AP1, PEBP1, TPM1, NUDC, LRPPRC, H1-2, IVNS1ABP, CALU, SMARCC1, GPX4, RUVBL1, CALM2, MAD2L1, DDX5, RRAS2, GABRB3, BSG, FABP5, TXNL4A, HYOU1, OAZ2, TUFM, TIMM13, HMGCR, DPYSL2, G3BP2, NLN, DNMT3A, CEBPZ, HMGN1, ADSL, BCCIP, CCNG1, GLRX3, SRM, CTNND1, MSH6, U2SURP, MTHFD1, LARS1, NOP16, MFAP2, RTN3, PCDH1, SEC11A, KRT8, RCN1, PAWR, MCM3, TMEM167A, PRMT1, CDC37, SUPT16H, JMJD1C, DCUN1D5, NME4, UBE2D3, CARM1, HSP90B1, UCHL1, COX16, DSG2, LMAN2, FBL, HNRNPK, SERPINH1, SLC25A39, ADAR, TCOF1, NARS1, NES, TRAP1, RAB34, CKB  

Does PIK3CD show a significant change in expression under this perturbation?


Now that you know the answer is Yes, please explain why. You need to think carefully and observe the expression profile of the background cells (i.e., the given gene list), where the order is crucial information. Additionally, you need to explain from the perspectives of biological pathways and functions."""},
]

response = completion(
    model="claude-opus-4-20250514-thinking",  # 使用 reasoning 版本
    messages=messages,
    reasoning_effort="high",  # 强制高强度推理
)

msg = response["choices"][0]["message"]

print(msg)


if "reasoning_content" in msg:
    print("=== Think Process ===")
    print(msg["reasoning_content"])
    print("\n=== Final Answer ===")
print(msg["content"])
