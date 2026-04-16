#Deep Learning Models for Classification of RNA-seq Data:
#TCGA Data Downloading and Pre-Processing

#Rhys Lossman
#Georgetown University
#MATH 5310 Deep Learning
#April 23, 2026


library(TCGAbiolinks)
library(SummarizedExperiment)
library(arrow)
library(dplyr)

query <- GDCquery(project = "TCGA-BRCA",
                  data.category = "Transcriptome Profiling",
                  data.type = "Gene Expression Quantification", 
                  workflow.type = "STAR - Counts")

GDCdownload(query)
brca_se <- GDCprepare(query)
counts <- assay(brca_se)
meta <- colData(brca_se)

subtypes <- TCGAquery_subtype("BRCA")
labels <- select(subtypes, patient, BRCA_Subtype_PAM50)

sample_type <- substr(colnames(counts), 14, 15)
tumor_samples <- sample_type == "01"
counts_tumor <- counts[, tumor_samples]
meta_tumor <- meta[tumor_samples, ]

X <- t(counts_tumor)
rownames(X) <- sub("\\..*", "", rownames(X))
patient_X <- data.frame(patient = meta_tumor$patient, X, check.names = FALSE)

data <- inner_join(patient_X, labels, by = 'patient')
data <- data[!duplicated(data$patient), ]
write_feather(data, 'tcga-brca-expression-labeled.feather')
