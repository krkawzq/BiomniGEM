sc_ann_task_abstract = "You now need to determine the most probable cell type of a single cell based on its expression profile, which is represented by the top highly expressed genes sorted in descending order."

sc_ann_task_description_list = [
    "Given the expression profile of a single cell, identify its most probable cell type.",
    "You will receive a list of genes with the highest expression in one single cell. These highly expressed genes serve as a proxy for cell identity and can be used to infer the most likely cell type.",
    "The gene list is obtained from **single-cell RNA sequencing**, with genes sorted in descending order of expression (higher first, lower later)."
]


perturb_de_task_abstract = "You are given a task of assessing whether CRISPRi knockdown of a perturbation gene leads to a statistically significant change in the expression of another gene."

perturb_de_task_description_list = [
    "You will be provided with a list of **highly expressed genes** in this cell line, obtained by averaging single-cell expression profiles across all cells (a bulk-like profile representing the baseline transcriptional state). The list is sorted in descending order of mean expression (genes appearing later have lower expression).",
    "The **perturbation condition** (which gene is knocked down by CRISPRi).",
    "The **target gene** whose potential expression change should be assessed.",
    "The **background cell line** where the perturbation is applied."
]


perturb_dir_task_abstract = "You are given a task of determining the direction of expression change caused by a CRISPRi perturbation."

perturb_dir_task_description_list = [
    "You will be provided with a list of **highly expressed genes** in this cell line, obtained by averaging single-cell expression profiles across all cells (a bulk-like profile representing the baseline transcriptional state). The list is sorted in descending order of mean expression (genes appearing later have lower expression).",
    "The **perturbation condition** (which gene is knocked down by CRISPRi).",
    "The **target gene** whose potential expression change should be assessed.",
    "The **background cell line** where the perturbation is applied."
]


"""
Task prompt definitions
Define task descriptions and reasoning guidance for each specific task in each dataset
"""

# ============================================================================
# 1. Mol-Instructions Tasks
# ============================================================================

# 1.1 Catalytic Activity Prediction
mol_catalytic_activity_abstract = "You are tasked with predicting the biochemical reaction catalyzed by a protein and its EC number based on its amino acid sequence."

mol_catalytic_activity_description = [
    "You will be provided with a protein amino acid sequence.",
    "Your task is to predict the specific biochemical reaction catalyzed by this protein, including substrates, products, and reaction type.",
    "If possible, provide the EC number (Enzyme Commission number).",
    "Your answer should include the reaction equation and EC classification information.",
    "During reasoning, consider: active site motifs in the sequence, known catalytic domains, and enzyme family characteristics."
]

# 1.2 Domain and Motif Recognition
mol_domain_motif_abstract = "You are tasked with identifying functional domains and conserved sequence motifs in a protein sequence."

mol_domain_motif_description = [
    "You will be provided with a protein amino acid sequence.",
    "Your task is to identify functional domains and sequence motifs (conserved motifs) within the sequence.",
    "Domains are sequence segments with independent structure and function, typically containing tens to hundreds of amino acids.",
    "Motifs are shorter conserved sequence patterns, usually associated with specific functions (e.g., phosphorylation sites, DNA binding sites).",
    "During reasoning, consider: sequence length, hydrophobic regions, charged residue distribution, and characteristic sequences of known domains.",
    "Your answer should clearly specify the name of the domain/motif and its possible function."
]

# 1.3 General Function Prediction
mol_general_function_abstract = "You are tasked with predicting the general biological function category of a protein."

mol_general_function_description = [
    "You will be provided with a protein amino acid sequence.",
    "Your task is to predict the general functional category of this protein, such as enzymatic activity, transport, signal transduction, structural protein, etc.",
    "Your answer should be a general functional description, without requiring overly specific molecular mechanisms.",
    "During reasoning, consider: sequence features, conserved regions, possible localization signals, and similarity to known proteins."
]

# 1.4 Detailed Protein Function Annotation
mol_protein_function_abstract = "You are tasked with providing a detailed description of protein function and molecular mechanisms."

mol_protein_function_description = [
    "You will be provided with a protein amino acid sequence.",
    "Your task is to provide a detailed description of this protein's function, including molecular mechanisms, biological process involvement, and interactions.",
    "Your answer should be comprehensive and specific, including: primary function, mechanism of action, biological pathways involved, and known interacting proteins.",
    "During reasoning, comprehensively consider: sequence features, domain composition, functions of known homologous proteins, and species-specific characteristics."
]


# ============================================================================
# 2. UniProtQA Tasks
# ============================================================================

# 2.1 Protein Function Q&A
uniprotqa_function_abstract = "You are tasked with answering questions about the biological function of a protein."

uniprotqa_function_description = [
    "You will be provided with a protein sequence and a question about its function.",
    "The question is typically: 'What is the function of this protein?'",
    "Your answer should describe the protein's primary biological function, biological processes involved, and mechanism of action.",
    "During reasoning, consider: functional implications from sequence features, known functions of domains, possible enzymatic activity or binding properties.",
    "Your answer should be accurate, concise, and use standard biological terminology."
]

# 2.2 Protein Naming Q&A
uniprotqa_naming_abstract = "You are tasked with answering questions about the official names and aliases of a protein."

uniprotqa_naming_description = [
    "You will be provided with a protein sequence and a question about its name.",
    "The question is typically: 'What are the official names of this protein?'",
    "Your answer should include: recommended standard name, common aliases, and gene name.",
    "During reasoning, base your answer on: sequence features, functional inference, and naming conventions of known homologous proteins.",
    "Your answer format should be clear, distinguishing between primary and secondary names."
]

# 2.3 Protein Family Assignment
uniprotqa_family_abstract = "You are tasked with determining the protein family to which a protein belongs."

uniprotqa_family_description = [
    "You will be provided with a protein sequence and a question about its family assignment.",
    "The question is typically: 'What is the protein family that this protein belongs to?'",
    "Your answer should state the family name, such as 'Belongs to the ABC transporter family'.",
    "During reasoning, consider: conserved sequence features, domain composition, functional characteristics, and evolutionary relationships.",
    "Your answer should be precise to the specific family or subfamily."
]

# 2.4 Subcellular Localization
uniprotqa_location_abstract = "You are tasked with predicting the subcellular localization of a protein."

uniprotqa_location_description = [
    "You will be provided with a protein sequence and a question about its localization.",
    "The question is typically: 'What are the subcellular locations of this protein?'",
    "Your answer should specify the protein's localization within the cell, such as nucleus, cell membrane, mitochondria, etc.",
    "During reasoning, consider: N-terminal signal peptides, transmembrane domains, nuclear localization signals (NLS), and other targeting sequences.",
    "Your answer may include multiple locations (if the protein has different localizations under different conditions)."
]

# 2.5 Other Property Q&A
uniprotqa_other_abstract = "You are tasked with answering questions about other biochemical properties of a protein."

uniprotqa_other_description = [
    "You will be provided with a protein sequence and a question about a specific property.",
    "Questions may involve: post-translational modifications, protein interactions, subunit composition, cofactor binding, etc.",
    "Your answer should address the specific question and provide accurate biochemical information.",
    "During reasoning, analyze relevant features in the sequence based on the question type."
]


# ============================================================================
# 3. Pika-DS Tasks
# ============================================================================

# 3.1 General Protein Q&A
pika_ds_qa_abstract = "You are tasked with answering protein science questions based on SwissProt curated fields."

pika_ds_qa_description = [
    "You will be provided with a protein sequence and a free-form scientific question.",
    "Questions may involve: function, catalytic activity, domains, subunits, interactions, molecular weight, sequence length, etc.",
    "Your answer should provide appropriate information based on the question type:",
    "  - Classification/label questions: return category labels",
    "  - Numerical attribute questions: return numerical values with units",
    "  - Descriptive questions: provide concise and accurate descriptions",
    "During reasoning, comprehensively consider multiple aspects of sequence features.",
    "Your answer must be based on reasonable inference from sequence features; do not fabricate unsupported information."
]


# ============================================================================
# 4. ChatNT DNA Tasks
# ============================================================================

# 4.1 Promoter Detection
chatnt_promoter_abstract = "You are tasked with determining whether a DNA sequence is a promoter region."

chatnt_promoter_description = [
    "You will be provided with a DNA sequence.",
    "Your task is to determine whether this sequence is a gene promoter region.",
    "Promoters are typically located upstream of the transcription start site and contain regulatory elements such as TATA box and CAAT box.",
    "During reasoning, consider: conserved motifs in the sequence, GC content, and nucleotide composition bias.",
    "Your answer is typically a classification label: 'promoter' or 'non-promoter'.",
    "Some tasks may distinguish between TATA-containing and TATA-less promoters."
]

# 4.2 Enhancer Recognition
chatnt_enhancer_abstract = "You are tasked with identifying whether a DNA sequence is an enhancer and its type."

chatnt_enhancer_description = [
    "You will be provided with a DNA sequence.",
    "Your task is to determine whether this sequence is an enhancer and its possible type.",
    "Enhancers are distal regulatory elements that can enhance gene transcription, independent of distance and orientation relative to the promoter.",
    "During reasoning, consider: transcription factor binding sites, DNA accessibility features, and sequence conservation.",
    "Your answer may be binary classification (yes/no) or multi-class (developmental/housekeeping-related, etc.)."
]

# 4.3 Splice Site Prediction
chatnt_splice_site_abstract = "You are tasked with predicting the location and type of splice sites in a DNA sequence."

chatnt_splice_site_description = [
    "You will be provided with a DNA sequence.",
    "Your task is to identify splice sites in the sequence: donor sites or acceptor sites.",
    "Donor sites are typically GT dinucleotides (exon-intron boundary), and acceptor sites are typically AG dinucleotides.",
    "During reasoning, consider: the conserved GT-AG rule, sequence context around splice sites, and branch point sequences.",
    "Your answer may be a classification label or site position."
]

# 4.4 Chromatin Accessibility
chatnt_chromatin_abstract = "You are tasked with predicting the chromatin accessibility of a DNA region."

chatnt_chromatin_description = [
    "You will be provided with a DNA sequence.",
    "Your task is to predict the chromatin accessibility of this region (whether it is open chromatin).",
    "Open chromatin regions typically correspond to active regulatory elements accessible to transcription factor binding.",
    "During reasoning, consider: sequence composition, presence of regulatory motifs, and GC content.",
    "Your answer is typically a classification (accessible/inaccessible) or a continuous value (accessibility score)."
]

# 4.5 DNA Methylation
chatnt_methylation_abstract = "You are tasked with predicting the methylation status of CpG sites in a DNA sequence."

chatnt_methylation_description = [
    "You will be provided with a DNA sequence.",
    "Your task is to predict the methylation status of CpG dinucleotides in the sequence.",
    "DNA methylation is an important epigenetic modification, typically associated with gene silencing.",
    "During reasoning, consider: presence of CpG islands, sequence context, and genomic region type (promoter/gene body/enhancer).",
    "Your answer may be binary classification (methylated/unmethylated) or a continuous value (methylation level)."
]

# 4.6 Histone Modification
chatnt_histone_abstract = "You are tasked with predicting the histone modification status of a DNA region."

chatnt_histone_description = [
    "You will be provided with a DNA sequence.",
    "Your task is to predict specific histone modifications in this region (e.g., H3K4me3, H3K27ac).",
    "Different histone modifications are associated with different functional states:",
    "  - H3K4me3: active promoter mark",
    "  - H3K27ac: active enhancer mark",
    "  - H3K27me3: repressive mark",
    "During reasoning, consider: genomic location, sequence features, and associations with other epigenetic marks.",
    "Your answer is typically presence/absence or a continuous value of modification level."
]


# ============================================================================
# 5. ChatNT RNA Tasks
# ============================================================================

# 5.1 RNA Degradation Prediction
chatnt_rna_degradation_abstract = "You are tasked with predicting the degradation rate or stability of an RNA sequence."

chatnt_rna_degradation_description = [
    "You will be provided with an RNA sequence (may be mRNA or other RNA types).",
    "Your task is to predict the degradation rate or stability of this RNA.",
    "RNA stability is influenced by multiple factors: sequence features in 5'UTR and 3'UTR, poly(A) tail length, and secondary structure.",
    "During reasoning, consider: AU-rich elements (ARE), microRNA binding sites, and stability of RNA secondary structure.",
    "Your answer may be a classification (stable/unstable) or a continuous value (half-life)."
]

# 5.2 Long Non-coding RNA Identification
chatnt_lncrna_abstract = "You are tasked with identifying whether an RNA sequence is a long non-coding RNA (lncRNA)."

chatnt_lncrna_description = [
    "You will be provided with an RNA sequence.",
    "Your task is to determine whether this sequence is a long non-coding RNA (lncRNA).",
    "lncRNAs are non-coding RNAs longer than 200nt that do not encode proteins but have regulatory functions.",
    "During reasoning, consider: presence and length of open reading frames (ORFs), coding potential score, and sequence conservation patterns.",
    "Your answer is typically binary classification: lncRNA or protein-coding."
]

# 5.3 Poly(A) Signal Prediction
chatnt_polya_abstract = "You are tasked with predicting the location of poly(A) signals in an mRNA sequence."

chatnt_polya_description = [
    "You will be provided with an mRNA sequence (typically the 3' end region).",
    "Your task is to identify the location of the polyadenylation signal.",
    "The poly(A) signal is a key sequence for transcription termination and poly(A) tail addition, with the typical sequence being AAUAAA or variants.",
    "During reasoning, consider: conserved hexamer signals, upstream and downstream sequence elements, and GU/U-rich regions.",
    "Your answer may be the presence/location of the signal or the type of specific signal sequence."
]


# ============================================================================
# 6. ChatNT Protein Tasks
# ============================================================================

# 6.1 Protein Stability
chatnt_protein_stability_abstract = "You are tasked with predicting the thermodynamic stability of a protein or the effect of mutations on stability."

chatnt_protein_stability_description = [
    "You will be provided with a protein sequence or mutation information.",
    "Your task is to predict protein stability or mutation-induced stability changes (ΔΔG).",
    "Protein stability is related to its folding free energy, influenced by amino acid composition and interactions.",
    "During reasoning, consider: hydrophobic effects, hydrogen bond networks, disulfide bonds, charged residue distribution, and structural context of mutation sites.",
    "Your answer may be a stability classification (stable/unstable) or a ΔΔG numerical value."
]

# 6.2 Protein Fluorescence
chatnt_protein_fluorescence_abstract = "You are tasked with predicting the fluorescence intensity of fluorescent protein variants."

chatnt_protein_fluorescence_description = [
    "You will be provided with a fluorescent protein sequence (typically GFP or its variants).",
    "Your task is to predict the fluorescence intensity or relative brightness of this protein.",
    "Fluorescence intensity is influenced by the chromophore chemical environment, protein folding quality, and quantum yield.",
    "During reasoning, consider: amino acids surrounding the chromophore, effects of mutations on folding, and solvent accessibility.",
    "Your answer is typically a continuous value (fluorescence intensity or relative value)."
]

# 6.3 Protein Thermal Stability (Meltome)
chatnt_protein_meltome_abstract = "You are tasked with predicting the melting temperature (Tm) of a protein."

chatnt_protein_meltome_description = [
    "You will be provided with a protein sequence.",
    "Your task is to predict the melting temperature (Tm) of this protein, i.e., the temperature at which 50% of the protein is denatured.",
    "Tm reflects the thermal stability of the protein, related to its structural compactness and interaction strength.",
    "During reasoning, consider: number of disulfide bonds, size of hydrophobic core, charged residue distribution, and sequence length.",
    "Your answer is typically a temperature value (degrees Celsius)."
]


# ============================================================================
# 7. LLaMA-Gene Tasks
# ============================================================================

# 7.1 DNA Promoter Detection (LLaMA-Gene version)
llama_gene_dna_promoter_abstract = "You are tasked with determining whether a DNA sequence is a promoter (instruction format)."

llama_gene_dna_promoter_description = [
    "You will be provided with a DNA sequence and an instruction-formatted question.",
    "The question is typically: 'Determine promoter detection of following dna sequence...'",
    "Your task is to determine whether the sequence is a promoter, with the answer being 'promoter' or 'Non-promoter'.",
    "The reasoning method is similar to the ChatNT promoter task, but the answer format is more standardized.",
    "Strictly return the answer according to the dataset's label format."
]

# 7.2 Protein Function Prediction (LLaMA-Gene version)
llama_gene_protein_function_abstract = "You are tasked with predicting protein function (instruction format)."

llama_gene_protein_function_description = [
    "You will be provided with a protein sequence and an instruction-formatted task description.",
    "The task may involve functional classification, property prediction, etc.",
    "During reasoning, perform functional inference based on sequence features.",
    "Your answer should be the target label or text defined by the dataset; do not expand or rewrite."
]


# ============================================================================
# 8. Biology-Instructions Tasks
# ============================================================================

# 8.1 RNA-Protein Interaction
bio_rna_protein_interaction_abstract = "You are tasked with predicting whether there is an interaction between an RNA sequence and a protein sequence."

bio_rna_protein_interaction_description = [
    "You will be provided with an RNA sequence and a protein sequence (marked with <rna> and <protein> tags).",
    "Your task is to determine whether there is a biologically meaningful interaction between these two molecules.",
    "Key factors for RNA-protein interaction:\n\t- RNA secondary structure (stem-loops, bulges, internal loops)\n\t- Typical domains of RNA-binding proteins (RRM, KH, dsRBD, etc.)\n\t- Sequence-specific recognition motifs\n\t- Physicochemical complementarity",
    "During reasoning, consider:\n\t1. Whether the protein contains known RNA-binding domains\n\t2. Whether the RNA sequence has special structural features\n\t3. Complementarity of sequence composition (charged residues and phosphate backbone)\n\t4. Known interaction patterns",
    "Your answer is typically: 'positive' (interaction exists) or 'negative' (no interaction).",
    "In some cases, you need to provide a detailed explanation of the basis for your judgment."
]


# ============================================================================
# Plant-specific Tasks (ChatNT subset)
# ============================================================================

# Plant Promoter Strength
chatnt_plant_promoter_strength_abstract = "You are tasked with predicting the transcriptional strength of a plant promoter."

chatnt_plant_promoter_strength_description = [
    "You will be provided with a plant DNA sequence (promoter region).",
    "Your task is to predict the strength of gene transcription driven by this promoter (strong/medium/weak).",
    "Plant promoters have different characteristics from animal promoters: TATA box position, CAAT box, and specific regulatory elements.",
    "During reasoning, consider: core promoter elements, upstream activating sequences (UAS), light-responsive elements, and hormone-responsive elements.",
    "Your answer may be a classification (strong/weak) or a continuous value (relative transcriptional activity)."
]


# ============================================================================
# 9. Biology-Instructions Tasks (Merged from 47 original tasks)
# ============================================================================

# 9.1 DNA: Promoter Detection
bio_promoter_detection_abstract = "You are tasked with identifying whether a DNA sequence contains promoter elements that regulate gene transcription."

bio_promoter_detection_description = [
    "You will be provided with a DNA sequence of varying length (70bp core promoter or 300bp region).",
    "Your task is to determine if this sequence functions as a promoter region.",
    "Promoter sequences contain regulatory elements such as TATA box, CAAT box, Initiator elements, and transcription factor binding sites.",
    "During reasoning, consider: sequence motifs (TATA box, Inr, DPE), GC content, position of regulatory elements, and nucleotide composition bias.",
    "Some sequences may be distinguished by the presence or absence of specific elements (TATA-containing vs TATA-less promoters).",
    "Your answer should be: 'Yes' (is a promoter) or 'No' (is not a promoter), sometimes with additional classification of promoter type."
]

# 9.2 DNA: Epigenetic Marks
bio_epigenetic_marks_abstract = "You are tasked with predicting histone modifications and epigenetic marks in a DNA region."

bio_epigenetic_marks_description = [
    "You will be provided with a DNA sequence from yeast or other organisms.",
    "Your task is to predict the presence of specific histone modifications (e.g., H3K4me3, H3K36me3, H3K9ac, H4ac).",
    "Different histone modifications are associated with different chromatin states and gene activity:",
    "  - H3K4me3: active promoter mark, associated with transcription initiation",
    "  - H3K36me3: gene body mark of actively transcribed genes",
    "  - H3K9ac/H3K14ac: transcriptional activation marks",
    "  - H3K79me3: associated with active transcription elongation",
    "  - H4ac: open chromatin and active transcription",
    "During reasoning, consider: genomic context (promoter/gene body/intergenic), sequence composition, DNA accessibility features, and co-occurrence of other epigenetic marks.",
    "Your answer should indicate the presence or absence of the specific epigenetic modification."
]

# 9.3 DNA: Transcription Factor Binding
bio_tf_binding_abstract = "You are tasked with identifying transcription factor binding sites (TFBS) in DNA sequences."

bio_tf_binding_description = [
    "You will be provided with a 100bp DNA sequence from human or mouse genomes.",
    "Your task is to determine whether this region contains functional transcription factor binding sites.",
    "TFBS are short DNA motifs (typically 6-20bp) recognized by transcription factors that regulate gene expression.",
    "During reasoning, consider: presence of known binding motifs, sequence conservation, DNA accessibility, GC content, and surrounding genomic context.",
    "Note that different cell types may have different binding profiles due to chromatin state differences.",
    "Your answer should be: 'Yes' (contains TFBS) or 'No' (does not contain TFBS)."
]

# 9.4 DNA: Enhancer Activity
bio_enhancer_activity_abstract = "You are tasked with predicting the regulatory activity of DNA enhancer sequences."

bio_enhancer_activity_description = [
    "You will be provided with a DNA sequence that may function as an enhancer.",
    "Your task is to predict the enhancer's transcriptional activity level in different contexts.",
    "Enhancers are distal regulatory elements that can enhance gene transcription regardless of distance or orientation relative to the promoter.",
    "Enhancers can be classified as:",
    "  - Housekeeping (HK): constitutively active enhancers that maintain basal gene expression",
    "  - Developmental (Dev): tissue-specific or stage-specific enhancers that regulate developmental programs",
    "During reasoning, consider: transcription factor binding site clusters, sequence conservation across species, CpG content, and regulatory motif composition.",
    "Your answer should provide enrichment scores for HK and Dev enhancer activity (higher scores indicate stronger activity)."
]

# 9.5 DNA: Regulatory Interaction
bio_regulatory_interaction_abstract = "You are tasked with predicting functional interactions between promoter and enhancer DNA sequences."

bio_regulatory_interaction_description = [
    "You will be provided with two DNA sequences: one promoter region and one enhancer region.",
    "Your task is to determine whether these two regulatory elements interact functionally in specific cell types.",
    "Promoter-enhancer interactions are mediated by:",
    "  - Shared transcription factor binding profiles (co-binding of TFs)",
    "  - 3D chromatin looping that brings distant elements into physical proximity",
    "  - Coordinated epigenetic signatures (similar histone modifications)",
    "  - Cell type-specific regulatory circuits",
    "During reasoning, consider: cell type context (GM12878, K562, HUVEC, HeLa-S3, IMR90, NHEK), sequence complementarity in TF binding motifs, and regulatory logic.",
    "Your answer should be: 'Yes' (functional interaction exists) or 'No' (no interaction)."
]

# 9.6 RNA: Isoform
bio_rna_isoform_abstract = "You are tasked with predicting alternative polyadenylation and transcript isoform usage patterns."

bio_rna_isoform_description = [
    "You will be provided with an RNA sequence containing polyadenylation signals.",
    "Your task is to predict the percentage of transcripts that use the proximal (closer to the stop codon) polyadenylation site.",
    "Alternative polyadenylation (APA) produces mRNA isoforms with different 3'UTR lengths, affecting mRNA stability, localization, and translation.",
    "During reasoning, consider: polyadenylation signals (AAUAAA and variants like AUUAAA), upstream and downstream sequence elements (U-rich and GU-rich regions), cleavage site positioning, and regulatory protein binding sites.",
    "Your answer should be a percentage value (0-1) indicating the proximal isoform usage ratio, where higher values indicate preference for the proximal site."
]

# 9.7 RNA: Modification
bio_rna_modification_abstract = "You are tasked with identifying chemical modifications on RNA sequences."

bio_rna_modification_description = [
    "You will be provided with an RNA sequence.",
    "Your task is to predict the type of RNA modification present (e.g., m6A, Am, Pseudouridine, m5C, etc.).",
    "RNA modifications are chemical alterations that regulate RNA stability, translation efficiency, localization, and protein binding.",
    "During reasoning, consider: sequence context around modification sites, consensus motifs (e.g., DRACH motif for m6A methylation), RNA secondary structure stability, and modification enzyme recognition patterns.",
    "Your answer should specify the type of modification detected."
]

# 9.8 RNA: Translation Regulation
bio_rna_translation_regulation_abstract = "You are tasked with predicting RNA translation efficiency and regulatory mechanisms."

bio_rna_translation_regulation_description = [
    "You will be provided with an RNA sequence (may include 5'UTR, coding region, or full transcript).",
    "Your task may involve predicting:",
    "  - Mean ribosome loading (ribosome density) on the transcript",
    "  - siRNA knockdown efficiency (percentage of mRNA remaining after treatment)",
    "  - RNA switch ON/OFF behavior (expression levels in different states)",
    "Ribosome loading reflects translation efficiency and is influenced by: 5'UTR structure and length, Kozak sequence context, uORFs (upstream open reading frames), codon usage and tRNA availability.",
    "siRNA efficiency depends on: sequence composition, GC content, thermodynamic properties, target site accessibility, and potential off-target effects.",
    "RNA switches undergo conformational changes: consider RNA secondary structure, aptamer domains, expression platforms, and switching mechanism.",
    "Your answer format depends on the specific task: ribosome loading intensity, percentage of remaining mRNA, or ON/OFF state values with dynamic range."
]

# 9.9 RNA: Function
bio_rna_function_abstract = "You are tasked with classifying RNA sequences and predicting their functional properties."

bio_rna_function_description = [
    "You will be provided with an RNA sequence.",
    "Your task may involve:",
    "  - Classifying non-coding RNA into functional families (e.g., HACA-box, snoRNA, miRNA, lncRNA)",
    "  - Predicting CRISPR guide RNA on-target efficiency (gene disruption success rate)",
    "For ncRNA classification, consider: sequence motifs and consensus sequences, secondary structure patterns (stem-loops, pseudoknots), length distributions, and conserved functional elements.",
    "For CRISPR efficiency, consider: GC content and distribution, nucleotide composition at key positions, guide RNA secondary structure, target site chromatin accessibility, and PAM sequence context.",
    "Your answer should be: ncRNA family classification or CRISPR efficiency score (0-1)."
]

# 9.10 Protein: Stability
bio_protein_stability_abstract = "You are tasked with predicting protein stability and folding properties."

bio_protein_stability_description = [
    "You will be provided with a protein amino acid sequence.",
    "Your task may involve predicting:",
    "  - Protein solubility in aqueous solution (soluble/insoluble)",
    "  - Protein stability score or mutation-induced stability changes",
    "  - Melting temperature (Tm) - temperature at which 50% of protein is denatured",
    "Protein stability is determined by: hydrophobic core packing, hydrogen bond networks, disulfide bonds, electrostatic interactions (salt bridges), and conformational entropy.",
    "Solubility depends on: hydrophobicity distribution, charged residue content and distribution, aggregation-prone regions, and structural disorder propensity.",
    "Thermostability is influenced by: number and position of disulfide bonds, salt bridges and ion pairs, hydrophobic core size, protein compactness, and sequence length.",
    "During reasoning, consider: amino acid composition, predicted secondary structure, physicochemical properties, and sequence-based aggregation predictors.",
    "Your answer format depends on the task: binary classification (soluble/insoluble), stability score, or temperature value in degrees Celsius."
]

# 9.11 Protein: Function
bio_protein_function_abstract = "You are tasked with predicting protein functional properties and characteristics."

bio_protein_function_description = [
    "You will be provided with a protein amino acid sequence.",
    "Your task may involve predicting:",
    "  - Enzymatic function and EC number (Enzyme Commission classification)",
    "  - Fluorescence intensity for fluorescent protein variants (typically GFP family)",
    "EC numbers are hierarchical (EC X.X.X.X): first digit indicates enzyme class (oxidoreductase, transferase, hydrolase, etc.), followed by progressively specific subclasses.",
    "For EC prediction, consider: presence of catalytic domains, active site motifs (catalytic triads, zinc-binding sites), and sequence similarity to known enzymes.",
    "Fluorescence intensity depends on: chromophore chemical environment, protein folding quality and structural stability, quantum yield, maturation efficiency, and solvent accessibility.",
    "For fluorescence, consider: mutations affecting the chromophore, structural changes in the beta-barrel, and amino acids in the chromophore vicinity.",
    "Your answer should be: one or more EC numbers (e.g., EC 3.2.2.9) or a numerical fluorescence intensity value."
]

# 9.12 Molecular Interaction
bio_molecular_interaction_abstract = "You are tasked with predicting binding interactions between biological molecules."

bio_molecular_interaction_description = [
    "You will be provided with two molecular sequences that may interact.",
    "Interaction types include:",
    "  - RNA-protein interaction: RNA sequence and protein sequence",
    "  - Antibody-antigen interaction: antibody sequence and antigen sequence",
    "RNA-protein interactions are mediated by: RNA-binding domains (RRM, KH, dsRBD, Zinc fingers) in proteins, RNA secondary structure elements (stem-loops, bulges, internal loops), sequence-specific recognition motifs, and electrostatic complementarity between charged residues and RNA phosphate backbone.",
    "Antibody-antigen binding involves: CDR (Complementarity-Determining Region) loops in antibodies (especially CDR3), epitope accessibility on antigen surface, shape and chemical complementarity, electrostatic and hydrophobic interactions.",
    "During reasoning, consider: presence of interaction domains/motifs, structural features enabling binding, physicochemical complementarity, and known binding patterns.",
    "Your answer should be: 'Positive' (interaction exists) or 'Negative' (no interaction), sometimes with affinity indication."
]

# ============================================================================
# Task Registry
# ============================================================================
tasks = {
    'Cell': {
        'sc_ann': {
            'abstract': sc_ann_task_abstract,
            'description': sc_ann_task_description_list,
            'task_type': 'cell',
            'file': 'sc_ann/dataset.csv'
        },
        'perturb_de': {
            'abstract': perturb_de_task_abstract,
            'description': perturb_de_task_description_list,
            'task_type': 'cell',
            'file': 'de/de.csv'
        },
        'perturb_dir': {
            'abstract': perturb_dir_task_abstract,
            'description': perturb_dir_task_description_list,
            'task_type': 'cell',
            'file': 'dir/dir.csv'
        },
    },
    'Mol-Instructions': {
        'catalytic_activity': {
            'abstract': mol_catalytic_activity_abstract,
            'description': mol_catalytic_activity_description,
            'task_type': 'protein',
            'file': 'vc/Mol_Instructions.csv'
        },
        'domain_motif': {
            'abstract': mol_domain_motif_abstract,
            'description': mol_domain_motif_description,
            'task_type': 'protein',
            'file': 'vc/Mol_Instructions.csv'
        },
        'general_function': {
            'abstract': mol_general_function_abstract,
            'description': mol_general_function_description,
            'task_type': 'protein',
            'file': 'vc/Mol_Instructions.csv'
        },
        'protein_function': {
            'abstract': mol_protein_function_abstract,
            'description': mol_protein_function_description,
            'task_type': 'protein',
            'file': 'vc/Mol_Instructions.csv'
        },
    },
    'UniProtQA': {
        'function': {
            'abstract': uniprotqa_function_abstract,
            'description': uniprotqa_function_description,
            'task_type': 'protein',
            'file': 'vc/UniProtQA.csv'
        },
        'naming': {
            'abstract': uniprotqa_naming_abstract,
            'description': uniprotqa_naming_description,
            'task_type': 'protein',
            'file': 'vc/UniProtQA.csv'
        },
        'family': {
            'abstract': uniprotqa_family_abstract,
            'description': uniprotqa_family_description,
            'task_type': 'protein',
            'file': 'vc/UniProtQA.csv'
        },
        'location': {
            'abstract': uniprotqa_location_abstract,
            'description': uniprotqa_location_description,
            'task_type': 'protein',
            'file': 'vc/UniProtQA.csv'
        },
        'other': {
            'abstract': uniprotqa_other_abstract,
            'description': uniprotqa_other_description,
            'task_type': 'protein',
            'file': 'vc/UniProtQA.csv'
        },
    },
    'Pika-DS': {
        'ds_qa': {
            'abstract': pika_ds_qa_abstract,
            'description': pika_ds_qa_description,
            'task_type': 'protein',
            'file': 'vc/Pika_DS.csv'
        },
    },
    'ChatNT-DNA': {
        'promoter': {
            'abstract': chatnt_promoter_abstract,
            'description': chatnt_promoter_description,
            'task_type': 'dna',
            'file': 'vc/ChatNT.csv'
        },
        'enhancer': {
            'abstract': chatnt_enhancer_abstract,
            'description': chatnt_enhancer_description,
            'task_type': 'dna',
            'file': 'vc/ChatNT.csv'
        },
        'splice_site': {
            'abstract': chatnt_splice_site_abstract,
            'description': chatnt_splice_site_description,
            'task_type': 'dna',
            'file': 'vc/ChatNT.csv'
        },
        'chromatin': {
            'abstract': chatnt_chromatin_abstract,
            'description': chatnt_chromatin_description,
            'task_type': 'dna',
            'file': 'vc/ChatNT.csv'
        },
        'methylation': {
            'abstract': chatnt_methylation_abstract,
            'description': chatnt_methylation_description,
            'task_type': 'dna',
            'file': 'vc/ChatNT.csv'
        },
        'histone': {
            'abstract': chatnt_histone_abstract,
            'description': chatnt_histone_description,
            'task_type': 'dna',
            'file': 'vc/ChatNT.csv'
        },
    },
    'ChatNT-RNA': {
        'rna_degradation': {
            'abstract': chatnt_rna_degradation_abstract,
            'description': chatnt_rna_degradation_description,
            'task_type': 'rna',
            'file': 'vc/ChatNT.csv'
        },
        'lncrna': {
            'abstract': chatnt_lncrna_abstract,
            'description': chatnt_lncrna_description,
            'task_type': 'rna',
            'file': 'vc/ChatNT.csv'
        },
        'polya': {
            'abstract': chatnt_polya_abstract,
            'description': chatnt_polya_description,
            'task_type': 'rna',
            'file': 'vc/ChatNT.csv'
        },
    },
    'ChatNT-Protein': {
        'protein_stability': {
            'abstract': chatnt_protein_stability_abstract,
            'description': chatnt_protein_stability_description,
            'file': 'vc/ChatNT.csv',
            'task_type': 'protein'
        },
        'protein_fluorescence': {
            'abstract': chatnt_protein_fluorescence_abstract,
            'description': chatnt_protein_fluorescence_description,
            'file': 'vc/ChatNT.csv',
            'task_type': 'protein'
        },
        'protein_meltome': {
            'abstract': chatnt_protein_meltome_abstract,
            'description': chatnt_protein_meltome_description,
            'file': 'vc/ChatNT.csv',
            'task_type': 'protein'
        },
    },
    'LLaMA-Gene': {
        'gene_dna_promoter': {
            'abstract': llama_gene_dna_promoter_abstract,
            'description': llama_gene_dna_promoter_description,
            'file': 'vc/LLaMA_Gene.csv',
            'task_type': 'dna'
        },
        'gene_protein_function': {
            'abstract': llama_gene_protein_function_abstract,
            'description': llama_gene_protein_function_description,
            'file': 'vc/LLaMA_Gene.csv',
            'task_type': 'protein'
        },
    },
    'Biology-Instructions': {
        'promoter_detection': {
            'abstract': bio_promoter_detection_abstract,
            'description': bio_promoter_detection_description,
            'task_type': 'dna',
            'file': 'vc/Biology_Instructions/by_task_type/promoter_detection.csv',
            'samples': 1000,
            'filter': 'split == "train"'
        },
        'epigenetic_marks': {
            'abstract': bio_epigenetic_marks_abstract,
            'description': bio_epigenetic_marks_description,
            'task_type': 'dna',
            'file': 'vc/Biology_Instructions/by_task_type/epigenetic_marks.csv',
            'samples': 1000,
            'filter': 'split == "train"'
        },
        'tf_binding': {
            'abstract': bio_tf_binding_abstract,
            'description': bio_tf_binding_description,
            'task_type': 'dna',
            'file': 'vc/Biology_Instructions/by_task_type/tf_binding.csv',
            'samples': 1000,
            'filter': 'split == "train"'
        },
        'enhancer_activity': {
            'abstract': bio_enhancer_activity_abstract,
            'description': bio_enhancer_activity_description,
            'task_type': 'dna',
            'file': 'vc/Biology_Instructions/by_task_type/enhancer_activity.csv',
            'samples': 1000,
            'filter': 'split == "train"'
        },
        'regulatory_interaction': {
            'abstract': bio_regulatory_interaction_abstract,
            'description': bio_regulatory_interaction_description,
            'task_type': 'dna',
            'file': 'vc/Biology_Instructions/by_task_type/regulatory_interaction.csv',
            'samples': 1000,
            'filter': 'split == "train"'
        },
        'rna_isoform': {
            'abstract': bio_rna_isoform_abstract,
            'description': bio_rna_isoform_description,
            'task_type': 'rna',
            'file': 'vc/Biology_Instructions/by_task_type/rna_isoform.csv',
            'samples': 1000,
            'filter': 'split == "train"'
        },
        'rna_modification': {
            'abstract': bio_rna_modification_abstract,
            'description': bio_rna_modification_description,
            'task_type': 'rna',
            'file': 'vc/Biology_Instructions/by_task_type/rna_modification.csv',
            'samples': 1000,
            'filter': 'split == "train"'
        },
        'rna_translation_regulation': {
            'abstract': bio_rna_translation_regulation_abstract,
            'description': bio_rna_translation_regulation_description,
            'task_type': 'rna',
            'file': 'vc/Biology_Instructions/by_task_type/rna_translation_regulation.csv',
            'samples': 1000,
            'filter': 'split == "train"'
        },
        'rna_function': {
            'abstract': bio_rna_function_abstract,
            'description': bio_rna_function_description,
            'task_type': 'rna',
            'file': 'vc/Biology_Instructions/by_task_type/rna_function.csv',
            'samples': 1000,
            'filter': 'split == "train"'
        },
        'protein_stability': {
            'abstract': bio_protein_stability_abstract,
            'description': bio_protein_stability_description,
            'task_type': 'protein',
            'file': 'vc/Biology_Instructions/by_task_type/protein_stability.csv',
            'samples': 1000,
            'filter': 'split == "train"'
        },
        'protein_function': {
            'abstract': bio_protein_function_abstract,
            'description': bio_protein_function_description,
            'task_type': 'protein',
            'file': 'vc/Biology_Instructions/by_task_type/protein_function.csv',
            'samples': 1000,
            'filter': 'split == "train"'
        },
        'molecular_interaction': {
            'abstract': bio_molecular_interaction_abstract,
            'description': bio_molecular_interaction_description,
            'task_type': 'multi',
            'file': 'vc/Biology_Instructions/by_task_type/molecular_interaction.csv',
            'samples': 1000,
            'filter': 'split == "train"'
        },
    },
    'Plant-Specific': {
        'promoter_strength': {
            'abstract': chatnt_plant_promoter_strength_abstract,
            'description': chatnt_plant_promoter_strength_description,
            'task_type': 'dna',
            'file': 'vc/ChatNT.csv'
        },
    }
}

def format_list(info_list):
    fmt_str = ""
    for info in info_list:
        if info.strip(" \t\n"):
            # 检查是否以 - 开头(列表项)
            if info.strip().startswith('-'):
                fmt_str += f"  {info}\n"
            else:
                fmt_str += f"- {info}\n"
    return fmt_str.strip()
