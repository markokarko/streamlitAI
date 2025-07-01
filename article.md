**Recent advances in text embedding: A Comprehensive**

**Review of Top-Performing Methods on the MTEB**

**Benchmark**

## 1 Abstract

arXiv:2406.01607v1  [cs.IR]  27 May 2024

Text embedding methods have become increasingly popular in both industrial and academic fields due to their critical role in a variety of natural language processing tasks. The significance of universal text embeddings has been further highlighted with the rise of Large Language Models (LLMs) applications such as Retrieval-Augmented Systems (RAGs). While previous models have attempted to be general-purpose, they often struggle to generalize across tasks and domains. However, recent advancements in training data quantity, quality and diversity; synthetic data generation from LLMs as well as using LLMs as backbones encourage great improvements in pursuing universal text embeddings. In this paper, we provide an overview of the recent advances in universal text embedding models with a focus on the top performing text embeddings on Massive Text Embedding Benchmark (MTEB). Through detailed comparison and analysis, we highlight the key contributions and limitations in this area, and propose potentially inspiring future research directions.

## 2 Introduction

Text embedding methods have gained considerable interest in both industry and academia due to their important role in various natural language processing tasks such as text classification (Li, Li, Xie, &amp; Li, 2021), text clustering (Xu, Xie, Li, Wang, Wang, &amp; Li, 2023; Reimers &amp; Gurevych, 2019), sentiment analysis (Suresh &amp; Ong, 2021; Zhang, Li, Xie, Lau, Cheng, Li, &amp; Zhang, 2022), information retrieval (Rajapakse, 2023), question answering (Yue, Kratzwald, &amp; Feuerriegel, 2021), dialogue systems (Long, Gao, Zou, Xu, Xie, Guo, Xu, Jiang, Xing, &amp; Yang, 2022), semantic textual similarity (Grill, Strub, Altch´e, Tallec, Richemond, Buchatskaya, Doersch, Avila Pires, Guo, Gheshlaghi Azar, et al., 2020), item recommendation (Steck, Ekanadham, &amp; Kallus, 2024) and so on (Choi, Kim, Joe, &amp; Gwon, 2021; Li, Zhang, Zhang, Long, Xie, &amp; Zhang, 2023; Wang, Yang, Huang, Yang, Majumder, &amp; Wei, 2023a). With the increasing popularity of Large Language Models (LLMs) based applications such as Retrieval-Augmented Systems (RAGs), the pivotal role of text embeddings has been underscored recently. This is mainly due to the fact that these LLM based applications are heavily dependent on the high quality text embeddings for tasks like vector search, a process where the most relevant documents are retrieved for LLM Question Answering (QA) (Li et al., 2023; Asai, Min, Zhong, &amp; Chen, 2023). Source attribution of generated text is another important application of text embeddings (Gao, Yen, Yu, &amp; Chen, 2023) that can improve the interpretability and trustworthiness of LLMs (Wang, Yang, Huang, Yang, Majumder, &amp; Wei, 2023b).

<!-- image -->

Figure 1: The 4 different eras of text embeddings. 1st era: Count-based Embeddings (with dimension reduction techniques); 2nd era: Static dense word embeddings, 3rd era: Contextualized embeddings; 4th era: Universal text embeddings.

The field of text embeddings in natural language processing (NLP) has experienced significant changes over the past few decades. The shift from basic task specific representations to complex universal embeddings highlights the progress in this area (as shown in Figure

1):

- 1st era: Count-based Embeddings. Bag of Words and Term Frequency-Inverse Document Frequency (TF-IDF) are two representative works in this era. Bag of Words (Harris, 1954) is one of the earliest text representation methods, which counts the presence or occurrence of each word in the documents and use them as features. TF-IDF measures how important a word is to a document relative to a corpus, by increasing proportionally to the number of times a word appears in the document but offset by the frequency of the word in the corpus (Manning, Raghavan, &amp; Schu¨tze, 2008). Both BoW and TF-IDF highlight the word/term relevancy instead of using the context information or the meaning of words (Petukhova, Matos-Carvalho, &amp; Fachada, 2024a). There are also other works in this era transforming texts into low-dimensional dense embeddings such as Latent Semantic Indexing (LSA) (Deerwester, Dumais, Furnas, Landauer, &amp; Harshman, 1990) (generating document embeddings with the decomposition of a word-document co-occurrence matrix) (Wang, Yang, Huang, Jiao, Yang, Jiang, Majumder, &amp; Wei, 2022).
- 2nd era: Static dense word embeddings. Word2Vec (Mikolov, Chen, Corrado, &amp; Dean, 2013), GloVe (Pennington, Socher, &amp; Manning, 2014) and FastText (Bojanowski, Grave, Joulin, &amp; Mikolov, 2017) are representative works that showed a significant step forward in the field of text representations using the surroundings of words to generate dense vector representations. Word2Vec focuses on local context using either Continuous Bag of Words (CBOW) approach (given the context, it predicts the target word) or Skip-gram approach (given the word, it predicts the context). Instead of only focusing on local context like Word2Vec, GloVe also takes the global corpus statistics into account. FastText further improves word embeddings by capturing the internal structure or morphology of words with a focus on character-level information of words and learning representations of sub-words (Patil, Boit, Gudivada, &amp; Nandigam, 2023). Even though these models can capture a range of semantic and syntactic similarities successfully, they provide a single static vector per word, which ignores the fact that a word’s meaning can be influenced by its surrounding context.
- 3rd era: Contextualized embeddings. The third era of text embeddings ushers in a new phase of embedding sophistication: context-sensitive dynamic embeddings that adapt or change based on context. Representative works include Embeddings From Language Models (ELMo) (Neumann, Iyyer, Gardner, Clark, Lee, &amp; Zettlemoyer,

2018), Generative Pre-trained Transformer (GPT) (Radford, Narasimhan, Salimans, Sutskever, et al., 2018) and Bidirectional Encoder Representations from Transformers (BERT) (Devlin, Chang, Lee, &amp; Toutanova, 2018). ELMo models the polysemy using a bidirectional Long Short Term Memory Network (LSTM) with the concatenation of the left-to-right and right-to-left representations. Unlike ELMo, GPT uses Transformer (one-way instead of bi-directional) (Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, &amp; Polosukhin, 2017) to learn the text embedding using a combination of unsupervised pre-training and supervised fine-tuning. It was observed that attentional memory of the transformer assisted in (better) transfer compared to LSTMs (Patil et al., 2023). BERT instead uses a bidirectional Transformer encoder to take into account both the left and right context for Masked Language Model (MLM) and Next Sentence Prediction (NSP) tasks during pre-training, which allows for a deeper understanding of word relationships by considering the full context of a word in a sentence in both directions. (Devlin et al., 2018; Petukhova, Matos-Carvalho, &amp; Fachada, 2024b).

- 4th era: Universal text embeddings. The pursuit of developing a unified model to address a multitude of downstream tasks has been long-standing (Li et al., 2023). Despite attempting to be general-purpose in previous models such as (Cer, Yang, Kong, Hua, Limtiaco, John, Constant, Guajardo-Cespedes, Yuan, Tar, et al., 2018; Raffel, Shazeer, Roberts, Lee, Narang, Matena, Zhou, Li, &amp; Liu, 2020; Ni, Abrego, Constant, Ma, Hall, Cer, &amp; Yang, 2021a), studies indicate that these embedding models struggle to generalize across tasks and domains (Lee, Dai, Ren, Chen, Cer, Cole, Hui, Boratko, Kapadia, Ding, et al., 2024). Thanks to the increasing number and improved quality of diverse text datasets across different tasks (Xiao, Liu, Zhang, &amp; Muennighof, 2023; Asai, Schick, Lewis, Chen, Izacard, Riedel, Hajishirzi, &amp; Yih, 2022), good quality synthetic data generated by LLMs (Lee et al., 2024; Wang et al., 2023b) as well as benchmarks with the focus on novel task and domain generalization such as the Massive Text Embedding Benchmark (MTEB) (Muennighoff, Tazi, Magne, &amp; Reimers, 2022a); the universality of text embeddings can be improved and evaluated across various languages and tasks such as retrieval, ranking, clustering, among others. The creation of unified models trained across diverse tasks has started to make progress with representative works like GTE (Li et al., 2023), BGE (Xiao et al., 2023), E5 (Wang et al., 2022, 2023b; Wang, Yang, Huang, Yang, Majumder, &amp; Wei, 2024), Gecko (Lee et al., 2024), LLM2Vec (BehnamGhader, Adlakha, Mosbach, Bahdanau, Chapados, &amp; Reddy, 2024), etc.

There are several reviews on text embeddings such as in (Wang, Zhou, &amp; Jiang, 2020a; Patil et al., 2023; Selva Birunda &amp; Kanniga Devi, 2021; Liu, Kusner, &amp; Blunsom, 2020; Kashyap, Nguyen, Schlegel, Winkler, Ng, &amp; Poria, 2024), but none of the existing work focus on the recent advances in the universal text embeddings in the fourth era. To fill in the gap, the main focus of this work is to review recent advances in universal text embeddings. More specifically, the top performing text embeddings in the Massive Text Embedding Benchmark (MTEB) (Muennighoff et al., 2022a) are the main focus of this review. The remainder of this paper is organized as follows: the preliminaries, background and categorization of 4th era universal text embeddings are introduced in Section 2. In Section 3, 4 and 5, the overview of the top performing state of the art text embeddings and their main contributions are explained. We describe the trends, performance and efficiency analysis of the state of the art text embeddings as well as their limitations in Section 6.

Finally, the conclusion and future directions in text embeddings are given in Section 7.

## 3 Preliminaries

### 3.1 Definitions

**Text embedding** In the context of Natural Language Processing (NLP) or Natural Language Understanding (NLU), text refers to a collection of words, phrases, sentences, paragraphs or larger utterance that convey meaningful information (Indurkhya &amp; Damerau, 2010). The form and length of text often vary on the task such as text classification/clustering, sentiment analysis, information retrieval, dialogue systems, item recommendation, etc. However, an embedding is a fixed-length low-dimensional dense vector representation (Wang et al., 2022). Text embedding then can be defined as a numerical dense representation of a word, phrase, sentence, or larger utterance in natural language in a certain space where texts with similar meanings are near each other (Reimers &amp; Gurevych, 2019; Lee et al., 2024; Liu et al., 2020; Li et al., 2023). The meaning of a word is influenced by its context, and it is from this context that a word embedding is usually learnt. The meaning of a sentence is more complex because it depends on the words used in the sentence, the syntactic structure as well as the surrounding sentences (Li, Zhao, &amp; Moens, 2022). The meaning of a document is even more complex as it is a high-level abstraction of the whole text (words, sentences, paragraphs, etc.). The definition of ”meaning”, ”local information” or ”context” changes when the text length changes, which makes it a great challenge to learn the embedding for an ”arbitrary span of contiguous text” (Devlin et al., 2018).

**Universal text embedding** In recent works, universal text embedding (Xiao et al., 2023; BehnamGhader et al., 2024; Solatorio, 2024) or general-purpose text embedding as used in (Lee et al., 2024; Li et al., 2023; Wang et al., 2022; Springer, Kotha, Fried, Neubig, &amp; Raghunathan, 2024) generally means a unified comprehensive text embedding model that can address a multitude of downstream tasks. In other words, the universal text embedding is not just proficient in a single particular task, but it proves to be consistently beneficial across a range of tasks such as text classification, text clustering, sentiment analysis, semantic textual similarity, summarization, retrieval tasks, etc. The objective of creating universal text embeddings is to mimic the fundamental process of how humans understand and process text, which can be beneficial in various domains (Li et al., 2022). With the recent work such as (Wang et al., 2024; Muennighoff, Su, Wang, Yang, Wei, Yu, Singh, &amp; Kiela, 2024), the definition of universal text embedding has been extended to multi-task, multi-lingual, while (Li et al., 2023) shows that a natural language model can also understand well programming languages. In this work, we define universal text embedding as **a unified comprehensive text embedding model that can address a multitude of input text length, downstream tasks, domains and languages** . The research of universal text embedding has been stimulated by several recent developments. These include the growth in quantity and refinement in quality of diverse text datasets across various tasks (Xiao et al., 2023; Asai et al., 2022), the production of high-quality synthetic data by LLMs (Lee et al., 2024; Wang et al., 2023b), and benchmarks that emphasize new task and domain generalization, such as the multi-lingual Massive Text Embedding Benchmark (MTEB) (Muennighoff et al., 2022a).

### 3.2 Background

In this work, we study and analyze the top performing text embedding models that are either open-source or well documented from MTEB English benchmark (because the English benchmark has more and diverse evaluation tasks compared to other languages). It can be found that BERT-based models used in (Li et al., 2023; Xiao et al., 2023; Wang et al., 2022, 2024; Li &amp; Li, 2023) and LLMs used in (Wang et al., 2024; Muennighoff et al., 2024; BehnamGhader et al., 2024; Rui, Ye, Shafiq Rayhan, Caiming, Yingbo, &amp; Semih, 2024; Springer et al., 2024; Lee et al., 2024) are two most popular backbones of the top performing universal text embedding models on the MTEB English benchmark.

**BERT (Devlin et al., 2018)** To generate contextual embeddings, BERT, pre-trained on a massive corpus and fine-tuned using labeled data from the downstream tasks, employs a bidirectional Transformer encoder to take into account both the left and right context in all layers. To alleviate the uni-directionality constraint, BERT proposes a masked language modelling (MLM) objective, where some of the tokens of a input sequence are randomly masked, and the objective is to predict the vocab-ids of the masked tokens based only on its context (Devlin et al., 2018). Additionally, a Next Sentence Prediction (NSP) task is also used to jointly pre-train text-pair representations with the objective to help tasks that require reasoning over text pairs (Liu et al., 2020). WordPiece embeddings with a 30,000 tokens vocabulary (Wu, Schuster, Chen, Le, Norouzi, Macherey, Krikun, Cao, Gao, Macherey, et al., 2016) is used by BERT, with special tokens including [CLS] token (a special classification token as the first token of each sequence) and [SEP] token to separate sentence pairs. The final hidden state of [CLS] is used for sentence-level tasks and the final hidden state of each token is used for token-level tasks (Devlin et al., 2018; Liu et al., 2020). Some important details about BERT include:

- Pre-training data: BooksCorpus (800M words) (Zhu, Kiros, Zemel, Salakhutdinov, Urtasun, Torralba, &amp; Fidler, 2015) and English Wikipedia ignoring lists, tables, and headers (2,500M words).
- Fine-tuning: task-specific inputs and outputs are fed into BERT to Fine-tuning all the parameters end-to-end.
- Loss function: the sum of the mean MLM likelihood and the mean NSP likelihood (Devlin et al., 2018).
- Model size: *BERTBASE* : 110 *M,BERTLARGE* : 340 *M* .
- Training: Training of *BERTBASE* was performed on 4 Cloud TPUs in Pod configuration (16 TPU chips total). Training of *BERTLARGE* was performed on 16 Cloud TPUs (64 TPU chips total). Each pre-training took 4 days to complete.

Following the success of BERT, several BERT-based models have been introduced, such as Robustly Optimized BERT Pretraining Approach (RoBERTa) (Liu, Ott, Goyal, Du, Joshi, Chen, Levy, Lewis, Zettlemoyer, &amp; Stoyanov, 2019a), Distilled version of BERT

(DistilBERT) (Sanh, Debut, Chaumond, &amp; Wolf, 2019), and A Lite BERT (ALBERT) (Lan, Chen, Goodman, Gimpel, Sharma, &amp; Soricut, 2019), each offering unique enhancements and optimizations while maintaining the core bidirectional approach of the original BERT model. One of the limitations of the BERT network structure is that no independent sentence embeddings are computed, which makes it difficult to use for various pair regression tasks due to large number of combinations. To allow for more efficient sentence-level embeddings, Sentence-BERT (SBERT) introduces the siamese and triplet network structures to generate highly effective semantically meaningful sentence embeddings that can be compared with cosine similarity, which has served as a cornerstone for further research (Reimers &amp; Gurevych, 2019; Kashyap et al., 2024). Another cornerstone work is Simple Contrastive Learning of Sentence Embeddings (SimCSE) (Gao, Yao, &amp; Chen, 2021) using unsupervised and supervised contrastive learning, which is widely adopted by recent state of the art text embeddings.

**Large Language Models** The widespread use of ChatGPT has showcased the impressive abilities of Large Language Models (LLMs) in following instructions, in-context learning with minimal few-shot examples and amazing conversation abilities with humans. While some of the best performing LLMs like GPT-4 (Achiam, Adler, Agarwal, Ahmad, Akkaya, Aleman, Almeida, Altenschmidt, Altman, Anadkat, et al., 2023) are proprietary with limited technical information available, some open-source LLM models like LLaMA-2 (Touvron, Martin, Stone, Albert, Almahairi, Babaei, Bashlykov, Batra, Bhargava, Bhosale, et al., 2023a), LLaMA-3 (AIMeta, 2024) and Mistral (Jiang, Sablayrolles, Mensch, Bamford, Chaplot, Casas, Bressand, Lengyel, Lample, Saulnier, et al., 2023) have made some notable efforts to catch up(Wang et al., 2023b). One advantage of using LLMs for text embedding is that they are extensively pre-trained on web-scale data already, which does not need the contrastive pre-training step used in existing state of the art text embedding models. At present, the foundation for the majority of LLMs is the Transformer architecture, which employs layers of multi-head attention in a very deep neural network. Decoder-only LLMs utilize the causal attention mechanism, where the representation of a token at a specific position *i* is exclusively impacted by the representations of tokens that come before it. The authors from (BehnamGhader et al., 2024) hypothesize that causal attention mechanism might partly be the reason of the slow adoption of decoder-only LLMs

<!-- image -->

Figure 2: Representative state of the art universal text embeddings and their main focus/contributions.

for text embedding tasks as it inherently limits their ability to produce rich contextualized representations. Several recent works such (Wang et al., 2024; Muennighoff et al., 2024; BehnamGhader et al., 2024; Rui et al., 2024; Springer et al., 2024; Lee et al., 2024) have proposed several solutions to mitigate such limitations.

**Massive Text Embedding Benchmark (MTEB)** The objective of MTEB is to provide comprehensive understandings on the universality of text embedding models, including 58 datasets covering 112 languages from 8 embedding tasks: Bitext mining, Classification, Clustering, Pair classification, Reranking, Retrieval, Semantic Textual Similarity (STS) and Summarization (Muennighoff et al., 2022a). The leader-board results are available on the Hugging Face Hub, where the results of English (56 datasets), Chinese (35 datasets),

French (26 datasets) and Polish (26 datasets) benchmark results can be found respectively.

### 3.3 Taxonomy of universal text embeddings

In this section, the main focuses and contributions of the some of the MTEB top performing state of the art text embedding methods are analyzed (shown in Figure 2), including: E5: EmbEddings from bidirEctional Encoder rEpresentations (Wang et al., 2022), GTE: General-purpose Text Embedding model (Li et al., 2023), BGE: Beijing Academy of Artificial Intelligence (BAAI) General Embedding (Xiao et al., 2023), UAE: Universal AnglE Embedding (Li &amp; Li, 2023), MRL: Matryoshka Representation Learning (Kusupati, Bhatt, Rege, Wallingford, Sinha, Ramanujan, Howard-Snyder, Chen, Kakade, Jain, et al., 2022), 2DMSE: 2D Matryoshka Sentence Embeddings (Li, Li, Li, Xie, &amp; Li, 2024), GRIT: Generative Representational Instruction Tuning (Muennighoff et al., 2024), LLM2Vec: (BehnamGhader et al., 2024), Multilingual E5: (Wang et al., 2024), E5-mistral-7b-instruct: (Wang et al., 2023b), Gecko: (Lee et al., 2024), Echo-mistral: (Springer et al., 2024), SFR-EmbeddingMistral: (Rui et al., 2024). The main focus/contributions are summarized and simplified as the following 4 aspects:

- Real world data: one way to learn the universal text embedding is using a multi-stage contrastive learning strategy with diverse training data mixture. For example, GTE (Li et al., 2023) uses diverse datasets for both pre-training and fine-tuning stage. BGE (Xiao et al., 2023) introduces a compressive data package C-Pack, while E5 (Wang et al., 2022) constructed a curated web-scale text pair dataset named Colossal Clean text Pairs (CCPairs) containing heterogeneous training signals by combining various semi-structured data sources along with aggressive filtering (270M text pairs filtered from 1.3B noisy text pairs) with a consistency-based filter to improve data quality (Dai, Zhao, Ma, Luan, Ni, Lu, Bakalov, Guu, Hall, &amp; Chang, 2022). Some works like GISTEmbed (Solatorio, 2024) also focus on improving the quality of hard negatives used for training. • Loss function: another research direction is to focus on improving the loss functions. As many existing text embedding works employed the cosine function in their training objective to measure the pairwise semantic similarity, the authors from UAE (Li &amp; Li, 2023) point out that there is the gradient vanishing issue due to the saturation zones of cosine function, which hinder the ability to learn subtle distinctions between texts in back propagation. Hence they propose a novel angle-optimized text embedding model called AnglE with angle optimization in a complex space which substantially improve the text embedding quality in various scenarios. Matryoshka Representation Learning (MRL) (Kusupati et al., 2022) and 2D Matryoshka Sentence Embeddings (2DMSE) propose new loss functions in order to reduce the computational cost of downstream tasks.
- LLMs are used to improve the universal text embeddings in two different ways:
    - 1. use synthetic data generated by LLMs: In (Li &amp; Li, 2023), the authors apply LLMs as data annotators to label the pseudo-supervised data for the training to improve the model performance. (Wang et al., 2023b) and (Wang et al., 2024) use proprietary LLMs including GPT-35-Turbo and GPT-4 to generate synthetic data covering a various range of text embedding tasks in 93 languages (among which 25% are generated by GPT-35-Turbo and others are generated by GPT-4) to increase the training data diversity, while (Lee et al., 2024) use synthetic data generation to distill knowledge from large language models into a retriever.
    - 2. use LLMs as backbone for text embeddings: as LLMs are extensively pretrained on web-scale data already, which does not need the large scale contrastive pre-training step used in existing state of the art text embedding models, many works also try to get embeddings directly from LLMs. For example, E5-mistral7b-instruct perform multi-task fine-tuning on Mistral 7b model which is one of the best performing method on MTEB. Echo-mistral (Springer et al., 2024), LLM2Vec (BehnamGhader et al., 2024) and GRIT (Muennighoff et al., 2024) propose various different ways so that decoder only LLMs can generate high quality text embeddings using bidirectional attention.

From Figure 2, it can be seen that most works have multiple contributions. To make the taxonomy easier, the state of the art text embeddings are divided into 3 groups based on their main contributions/focuses: Data focused text embeddings (detailed in Section 3), Loss focused text embeddings (detailed in Section 4) and LLM focused text embeddings (detailed in Section 5).

## 4 Data focused universal text embeddings

One way to learn the universal text embedding is using a multi-stage contrastive learning strategy with improved training data mixture in terms of data quantity, quality and diversity as summarized in Table 1. For example, GTE (Li et al., 2023) uses diverse datasets for both pre-training and fine-tuning stage. BGE (Xiao et al., 2023) introduces a compressive data package C-Pack, while E5 (Wang et al., 2022) constructed a curated web-scale text pair dataset named Colossal Clean text Pairs (CCPairs) containing heterogeneous training signals by combining various semi-structured data sources along with aggressive filtering (270M text pairs filtered from 1.3B noisy text pairs) with a consistency-based filter to improve data quality (Dai et al., 2022). Some works like GISTEmbed (Solatorio, 2024) also focus on improving the quality of hard negatives used for training. More details about each text embedding methods can be found below.

**General-purpose Text Embedding model (GTE)** With the focus on developing a unified more comprehensive model for general text representation to address a multitude of downstream tasks, the authors from (Li et al., 2023) introduce a multi-stage contrastive learning strategy with diverse training data mixture: in the initial stage, a large corpus of open-source data without any filtering or cleaning are used to learn basic language patterns with unsupervised contrastive learning; in the second stage, supervised fine-tuning refines the embeddings using contrastive learning with a smaller, high-quality dataset. At both stages, the number of training data are significantly increased.

For a query *q* , a relevant/positive document *d* +, a set of irrelevant/negative documents *D* − = { *d* 1− *,...,dn* −}, the InfoNCE loss (Oord, Li, &amp; Vinyals, 2018) is defined as in Equation

1:

<!-- image -->

(1)

where *s* ( *q,d* ) estimates the similarity between two pieces of text *q* and *d* via vector distance between their embeddings *q* = *E* ( *q* ) and *d* = *E* ( *d* ).

In GTE, given a batch of positive text pair samples {( *q* 1 *,d* 1) *,* ( *q* 2 *,d* 2) *,...,* ( *qn,dn* )}, the authors propose an improved contrastive loss (icl) can be viewed as a combination of loss variants proposed by (Radford, Kim, Hallacy, Ramesh, Goh, Agarwal, Sastry, Askell, Mishkin, Clark, et al., 2021; Ren, Lv, Qu, Liu, Zhao, She, Wu, Wang, &amp; Wen, 2021; Moiseev, Abrego, Dornbach, Zitouni, Alfonseca, &amp; Dong, 2023):

<!-- image -->

(2)

| where                                                         |     |
|---------------------------------------------------------------|-----|
| Z = Xes(qi,dj)/τ + Xes(qi,qj)/τ + Xes(qj,di)/τ + Xes(dj,di)/τ | (3) |

*j	j* ̸= *i	j	j* ̸= *i*

Table 1: The main contributions of data focus universal text embeddings: quantity, quality and diversity.

| Model names                 | Data focus contribution                                                                                                                                                                                                                                                                                                                                                                                                          |
|-----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GTE (Li et al., 2023)       | Substantial performance gains are achieved by notably augmenting data volume during both unsupervised pre-training (800M text pairs used for pre-training) and supervised fine-tuning stages with diverse mixture of datasets from multiple sources.                                                                                                                                                                             |
| BGE (Xiao et al., 2023)     | The largest dataset C-MTP was developed for general Chinese embedding with the focuses on: 1. data quality improvement by filtering the irrelevant text pairs in unlabelled data for general purpose fine-tuning (around 100M text pairs); 2. multi-task high quality labelled data (838,465 text pairs) for task-specific fine-tuning. Note: English data (for English version of BGE) is 2 times larger than the Chinese data. |
| GISTEmbed (Solatorio, 2024) | GIST is fine-tuned on top of BGE on MEDI and MTEB classification datasets with improved inbatch negative data quality.                                                                                                                                                                                                                                                                                                           |
| E5 (Wang et al., 2022)      | Development of CCPairs: curated large-scale text pair dataset by harvesting heterogeneous semi-structured data sources using consistencybased filter for quality improvement (reducing 1.3B text pairs to 270M text pairs for pretraining).                                                                                                                                                                                      |

Multilingual-E5 (Wang et al., 2024) **Multilingual** focus: diverse mixture of multilingual text pairs obtained from various sources (1B text pairs). Additional 500k **synthetic data** generated by GPT-3.5/4 which encompasses 150k unique instructions and covers 93 languages were used for fine-tuning.

The cosine similarity is used as the similarity measure *s* ( *q,d* ). GTE models are initialized with pre-trained language models such as BERT with mean pooling on top of the contextualized token representations produced by the language model for text embeddings. Some other details about GTE include:

• Pre-training data: around 800M text pairs text pairs for the unsupervised pre-training (a multinomial distribution is used to sample data batches from different data sources, taking into account their respective sizes.):

Web page (147M): Common Crawl, Clue Webs, MS MARCO documents, title as query and the body text as document.

- Academic Paper (45M): arXiv, bioRxiv, medRxiv, PubMed and Semantic Scholar, title as query and its abstract as document
- Hyperlink (106M): ClueWeb, Wikipedia and Semantic Scholar paper citations, the citation argument and the text from reference as relevant text pairs for contrast.
- Social Media (327M): Reddit, title body pair, post comment pair
- Knowledge Base (38M): WikiPedia and DBPedia, entity, description pairs
- Community QA (12M): StackExchange, Yahoo Answers, WikiHow and Amazon QA, summaritive title and a descriptive body pairs and question answer pairs
- News (3M): CCNews, MicrosoftNews, NPR, CNNDaily, title body pairs
- Code (20M): GitHub (CodeSearchNet) and StackOverflow, text-code pairs
- Others (91M): Amazon reviews about the goods, debate websites about one argument, googaq query answer pairs by prompting google search box with search log queries.
- Fine-tuning data:
    - Web Search: MS MARCO (Bajaj, Campos, Craswell, Deng, Gao, Liu, Majumder, McNamara, Mitra, Nguyen, et al., 2016) passage retrieval benchmarks where hard negatives are mined by sampling from high-ranked documents retrieval system, excluding positive ones.
    - Open QA: Natural Questions (NQ), Trivia QA (Karpukhin, O˘guz, Min, Lewis, Wu, Edunov, Chen, &amp; Yih, 2020a; Kwiatkowski, Palomaki, Redfield, Collins,

Parikh, Alberti, Epstein, Polosukhin, Devlin, Lee, et al., 2019), Web Questions, HotpotQA (Yang, Qi, Zhang, Bengio, Cohen, Salakhutdinov, &amp; Manning, 2018), etc. Top ranked passage by retrieval system which do not include answer to the question is regarded as hard negatives.

- Natural Language Inference: MNLI (Williams, Nangia, &amp; Bowman, 2017) and SNLI (Bowman, Angeli, Potts, &amp; Manning, 2015), entailment as positive pairs and contradiction as negative pairs
- Fact Verification: training set from FEVER (Thorne, Vlachos, Christodoulopoulos, &amp; Mittal, 2018)
- Paraphrase: Quora (Iyer, Dandekar, &amp; Csernai, 2017) and StackExchange Dupquestion
- Others: miscellaneous datasets from different NLP tasks and domains released in MEDI (Su, Shi, Kasai, Wang, Hu, Ostendorf, Yih, Smith, Zettlemoyer, &amp; Yu, 2022) and BERRI (Asai et al., 2022). • Loss function: improved contrastive loss as Equation 2
- Negative sampling:

Pre-training: enlarged in-batch negatives,

**–** Fine-tuning: hard negatives mined by an extra retriever to form text triples.

- Model size:
- *GTEbase* : 110M (backbone: bert-base-uncased)
- *GTElarge* : 330M (backbone: bert-large-uncased)

**Beijing Academy of Artificial Intelligence (BAAI) General Embedding (BGE)** Similar to the objective of GTE, BGE also tries to learn general-purpose text embeddings, a comprehensive, unified embedding model which is capable of managing all types of uses, including retrieval, ranking, and classification, across various application settings such as question answering, language modeling, and conversation (Xiao et al., 2023). BGE introduces C-Pack, a comprehensive package designed to advance the general Chinese embedding (other languages version of BGE are also available), along with their training recipe: pretraining of an embedding-oriented text encoder, general-purpose contrastive learning, and task-specific fine-tuning. BERT-like architecture is used by BGE models where the last layer’s hidden state of the special token [CLS] is trained to work as the embedding (unlike GTE). Another major difference from GTE is that BGE uses instruction-based fine-tuning to deal with potentially mutually contradicted tasks: a task specific instruction which describes the nature of the task (e.g. search relevant passages for the query) is added to the query side for each text pair. Some other details about BGE include:

- Pre-training data (English version): unsupervised datasets including datasets like Wikipedia, CC-net, StackExchange, Reddit, S2ORC (Lo, Wang, Neumann, Kinney, &amp; Weld, 2019), and datasets from sentence-transformers.
- Fine-tuning data (English version): supervised datasets including NLI (Gao et al., 2021), FEVER (Thorne et al., 2018), NQ (Karpukhin et al., 2020a; Kwiatkowski et al., 2019), HotpotQA (Yang et al., 2018), Quora (Iyer et al., 2017), StackExchange Duplicates and MEDI (Su et al., 2022). • Loss function: the contrastive loss as in Equation 1
- Negative sampling:
- Fine-tuning: in addition to the in-batch negative samples, one hard negative sample is mined for each text pair from the task’s original corpus, following the ANN-style sampling strategy in (Xiong, Xiong, Li, Tang, Liu, Bennett, Ahmed, &amp; Overwijk, 2020)

*BGEsmall* : 24M (BERT-like architecture),

**–** *BGElarge* : 102M (BERT-like architecture), **–** *BGElarge* : 326M (BERT-like architecture).

**Guided In-sample Selection of Training Negatives for Text Embedding Finetuning (GISTEmbed)** GIST-large-Embedding-v0 is another top performing text embeddings on the MTEB benchmark which uses *BGElarge* as backbone. The main focus of GISTEmbed is to propose a novel strategy that enhances in-batch negative selection during contrastive training through a guide model (Solatorio, 2024), which improves the baseline performance slightly. However, the GIST-large-Embedding-v0 performance increase on MTEB benchmark compared to *BGElarge* is limited (0.11%). It is difficult to analyze if the limited performance increase is due to the proposed guided in-sample negative selection or due to the fact that they added in-domain MTEB training data to fine-tune the BGE embedding models.

**EmbEddings from bidirEctional Encoder rEpresentations (E5)** With the objective of creating high-quality general-purpose text embeddings suitable for any tasks requiring single-vector representations in both zero-shot or fine-tuned settings, the authors from (Wang et al., 2022) constructed a curated web-scale text pair dataset named Colossal Clean text Pairs (CCPairs) containing heterogeneous training signals by combining various semi-structured data sources such as CommunityQA, Common Crawl and Scientific papers along with aggressive filtering (270M text pairs filtered from 1.3B noisy text pairs) with a consistency-based filter to improve data quality (Dai et al., 2022). Some other details about E5 include:

- Pre-training data: (post, comment) pairs from Reddit, (question, upvoted answer) pairs from Stackexchange, (entity name + section title, passage) pairs from English Wikipedia, (title, abstract) and citation pairs from Scientific papers (Lo et al., 2019), and (title, passage) pairs from Common Crawl web pages, various News sources and others including “SimpleWiki”, “GooAQ”, “WikiHow”, “Yahoo Answers”. • Fine-tuning data: Natural Language Inference (NLI (Bowman et al., 2015)), MSMARCO passage ranking dataset (Bajaj et al., 2016), and Natural Questions (NQ) dataset (Karpukhin et al., 2020a; Kwiatkowski et al., 2019)
- Loss function:
- Fine-tuning: a linear interpolation between contrastive loss for hard labels and KL divergence for distilling soft labels from the teacher model

**–** Pre-training: in-batch negative samples (with large 32,768 batch size)

Fine-tuning: in-batch negative samples, mined hard negatives and knowledge distillation from a cross-encoder (CE) teacher model for the MS-MARCO and NQ datasets. For the NLI dataset, contradiction sentences are regarded as hard negatives.

- Model size:
- *E* 5 *base* : 110M, initialized from bert-base-uncased
- *E* 5 *large* : 330M, initialized from bert-large-uncased-whole-word-masking

**Multilingual-E5** In order to extend the English E5 models, the authors from (Wang et al., 2024) released Multilingual-E5 series models by using diverse mixture of multilingual text pairs obtained from various sources with around 1 billion text pairs. The English E5 model recipe is used for the training procedure, which involves contrastive pre-training on 1 billion multilingual text pairs and fine-tuning on a blend of labeled datasets, with the Multilingual-E5-large-instruct adopting the data mixture from (Wang et al., 2023b) that includes an additional 500k synthetic data created by GPT-3.5/4 and encompasses 150k unique instructions across 93 languages. Similar to BGE, instructions data are used to better inform embedding models about the task at hand for Multilingual-E5-large-instruct model. Some other details about Multilingual-E5 include:

- Pre-training data: around 1 billion multilingual text pairs from: Wikipedia, mC4, Multilingual CC News, NLLB, Reddit, S2ORC, Stackexchange, xP3 and Misc. SBERT Data.
- Fine-tuning data: blend of labeled datasets (around 1.6M) from MS-MARCO Passage, MS-MARCO Document NQ, TriviaQA, SQuAD, NLI, ELI5, NLLB, DuReader Retrieval, Fever, HotpotQA, Quora Duplicate Questions, Mr. TyDi and MIRACL (additional synthetic data with 150k unique instructions and covers 93 languages are used for fine-tuning Multilingual-E5-large-instruct model).
- Loss function:
- Fine-tuning: a linear interpolation between contrastive loss for hard labels and KL divergence for distilling soft labels from the teacher model
- Fine-tuning: in-batch negative samples, mined hard negatives and knowledge distillation from a cross-encoder (CE) teacher model.

2020b)),

Multilingual-E5-base: 278M (initialized from xlm-roberta-base (Conneau, Khandelwal, Goyal, Chaudhary, Wenzek, Guzm´an, Grave, Ott, Zettlemoyer, &amp; Stoyanov, 2019)),

- Multilingual-E5-large: 560M (initialized from xlm-roberta-large (Conneau et al., 2019))
- Multilingual-E5-large-instruct: 560M, fine-tuned with instruction data on MultilingualE5-large.

**Summary** In this section, the state of the art methods trying to achieve universal text embeddings with improved data quantity, quality and diversity are introduced. Most of these methods use datasets from Common Crawl , Wikipedia, social media, academic papers and sentence-transformers (fully or partially) as one part of the pre-training data. Code data and hyperlinks are also used by GTE, which enables GTE to understand both natural language and code. Similarly, Multilingual-E5 improve the data diversity by adding both real world and synthetic multilingual datasets in order to improve the universality across languages. On the other hand, most of these methods use hard negatives to improve the quality of negative samples. GISTEmbed proposes in-batch negative selection for better negative samples. E5 uses preliminary filters and consistency based filter to improve the training data quality while reducing pre-training data size from 1.3B to 270M. High quality multi-task datasets are also used by most of the methods during fine-tuning stage to improve the universality across downstream tasks.

## 5 Loss focused universal text embeddings

Contrastive learning with InfoNCE loss (Equation 1) is used by most of the state of the art universal text embedding models. Several loss variants have been proposed by (Radford et al., 2021; Ren et al., 2021; Moiseev et al., 2023) and the authors of GTE propose an improved contrastive loss (Equation 2) which combines these loss variants. As many existing text embedding works employed the cosine function in their training objective to measure the pairwise semantic similarity, the authors from UAE (Li &amp; Li, 2023) point out that there is the gradient vanishing issue due to the saturation zones of cosine function, which hinder the ability to learn subtle distinctions between texts in back propagation. Hence they propose a novel angle-optimized text embedding model called AnglE with angle optimization in a complex space which substantially improve the text embedding quality in various scenarios. Matryoshka Representation Learning (MRL) (Kusupati et al., 2022) and 2D Matryoshka Sentence Embeddings (2DMSE) propose new loss functions in order to reduce the computational cost of downstream tasks. More details about different novel losses proposed by the MTEB top performing universal text embedding models can be found below.

**Universal AnglE Embedding (UAE)** Similar to GTE and BGE, UAE also uses the pre-trained BERT model (uncased BERT base model with 110M parameters) as the backbone model (note: UAE large V1 uses roberta-large (Liu, Ott, Goyal, Du, Joshi, Chen, Levy, Lewis, Zettlemoyer, &amp; Stoyanov, 2019b) as the default backbone). As many existing

<!-- image -->

Figure 3: Cosine function’s saturation zones exhibit near-zero gradients, which makes it difficult for the model to learn during backpropagation.

text embedding works employed the cosine function in their training objective to measure the pairwise semantic similarity, the authors from (Li &amp; Li, 2023) point out that there is the gradient vanishing issue due to the saturation zones of cosine function (as shown in Figure 3), which hinder the ability to learn subtle distinctions between texts in backpropagation. To deal with the problem of vanishing gradients, a novel angle-optimized text embedding model called AnglE is proposed by introducing angle difference optimization in a complex space which substantially improve the text embedding quality in various scenarios. Given input text embedding pair ( *E* ( *q* ) *,E* ( *d* )), the chunking strategy (Sun, Deng, Nie, &amp; Tang, 2019) is used to get their representations in the complex space ( **z** *,* **w** ), followed by the angle difference ∆ *θqd* between **z** and **w** . Then the angle loss can be defined as:

<!-- image -->

(4)

where *sim* ( *E* ( *i* ) *,E* ( *j* )) is the similarity between the embedding of *i* and the embedding of *j* . The authors also propose LLM-supervised learning (use LLMs as data annotators to label the pseudo-supervised data) to effectively deal with the domain-supervised data scarcity problem (Li &amp; Li, 2023). Some other details about UAE include:

- Training data: the NLI datasets MNLI (Williams et al., 2017) and SNLI (Bowman et al., 2015), and/or LLM supervised data
- Loss function: AnglE loss: the combination of cosine objective, in-batch negative objective and angle objective.
- Negative sampling: in-batch negative samples and/or hard negatives
- Model size:
    - AnglE-BERT: 110M (backbone: uncased BERT),
    - UAE Large V1: 355M (backbone: roberta-large),

**Matryoshka Representation Learning (MRL) (Kusupati et al., 2022)** Deploying deep representation or text embedding involves two steps: a constant forward pass to compute the representation, and its use for downstream applications (Sato, 2021; Varma, 2019). The computation costs for the second step rise with the embedding dimensionality, data size, and label space, which can exceed the feature computation cost for large scale systems (Dean et al., 2009; Sun, Shrivastava, Singh, &amp; Gupta, 2017). The rigid nature of these representations requires high-dimensional embedding vectors for different tasks, even though varying resource and accuracy constraints call for flexibility (Kusupati et al., 2022). Given that we can’t predict the computational and statistical demands for each subsequent task, fixed-capacity representations/embeddings may not always be suitable and could either exceed or fall short of the task’s requirements. Could we create an adaptable representation that can adjust to a variety of downstream tasks with fluctuating computational resources?

MRL introduces a novel method for learning representations of data through a nested structure to induce flexibility in the learned representation, similar to Russian Matryoshka dolls, which encodes information at different granularities and allows a single embedding to adapt to the computational constraints of downstream tasks (Kusupati et al., 2022). The representation/embedding *z* is a *d* dimensional vector, *M* = [ *m* 1 *,m* 2 *,...d* ] are the chosen dimensions which define different representation sizes. MRL makes each of the first *m* dimensions *z* 1: *m* to be independently capable of being a general purpose representation of the data point *x* . Given a labelled dataset *D* = {( *x* 1 *,y* 1) *,* ( *x* 2 *,y* 2) *,...* ( *xN,yN* )} where *N* is the datasize and *yi* is the label of data *xi* , MRL uses standard empirical risk minimization to optimize multi-class classification loss for each nested dimension *m* ∈ *M* using a separate linear classifier, parameterized by **W** ( *m* ):

<!-- image -->

)	(5)

where L is the multi-class softmax cross-entropy loss function, *F* (·; *θF* ) is the deep neural network to get the representation/embedding *z* , *cm* is the importance scales. The authors also show that MRL extends seamlessly to web-scale datasets across vision, language, and vision + language. The experimental results show that MRL can be effectively used for large-scale adaptive classification and retrieval, providing similar accuracy to fixed-feature baseline with a significantly smaller representation size, and offering a more cost-effective and faster adaptive shortlisting and re-ranking system (Kusupati et al., 2022).

**2D Matryoshka Sentence Embeddings (2dMSE)** Despite MRL’s enhanced efficiency, it still necessitates going through all transformer layers before obtaining the transformer based text embedding, leading to significant compute and memory consumption. This raises questions about the impact of the fixed number of transformer layers on representation quality and the feasibility of using intermediate layers for sentence representation. With the aim to enhance the flexibility and scalability of the original MRL’s sentence embedding learning, two-dimensional Matryoshka Sentence Embedding (2DMSE) is proposed in (Li et al., 2024). 2DMSE uses *BERTbase* as backbone to encode text data *x* :

<!-- image -->

**X** (6)

where *cls* means the pooling strategy using “CLS” embeddings as the sentence embeddings; *l* ∈ [1 *,L* ] denotes the *l* -th layer of the L-layer transformer backbone; *m* ∈ *M* = [ *m* 1 *,m* 2 *,...d* ] (same as MRL) represents the first *m* dimensions in the *d* -dimensional embeddings. *l* allows 2DMSE scaling the encoder model in the dimension in terms of the number of layers while *m* allows 2DMSE scaling the encoder model in the dimension in terms of the embedding size. To ensure the quality of embeddings, full-capacity embeddings from the last attention layer **X** *dL* are trained consistently with the following objective:

L *dL* = *loss* ( **X** *dL* ; *A* )	(7)

The auxiliary information A is utilized for loss computation, typically indicating positive or negative samples or providing ranking details (Li et al., 2024). During the same training step, a shallower Transformer layer *l* is randomly chosen following a uniform distribution *l* ∼ *U* (1 *,L* − 1), and its complete embedding vector is directly utilized for representation learning:

L *dl* = *loss* ( **X** *dl* ; *A* )	(8)

2DMSE also uses MRL for nested low-dimensional vectors at both the last layer **X** *L* :

<!-- image -->

)	(10)

where *m* is the MRL embedding size.

The next step is to improve the shallow layer’s performance by aligning its embeddings to the last layer’s:

<!-- image -->

)	(11)

where KLDiv(,) denotes the Kullback-Leibler divergence. The weighted sum of [L *dL* , L *dl* , L *mL* , L *ml* , L *align* ] is used as the final objective.

Based on MRL and 2DMSE, several well performing text embeddings including mxbaiembed-large-v1 (335M) and mxbai-embed-2d-large-v1 (335M) are released in (Sean, Aamir, Darius, &amp; Julius, 2024). However, the training details of these models are not documented.

**Summary** In this section, the MTEB top performing universal text embedding models with the focus on proposing new loss functions are introduced. Apart from proposing variants on the classic InfoNCE loss, UAE introduces AnglE loss by introducing angle optimization in a complex space to deal with the vanishing gradients problem from cosine function’s saturation zone. Another line of research focuses on adaptable representations that can adjust to a variety of downstream tasks with fluctuating computational resources, where MRL proposes new loss function to make each of the first *m* dimensions of the text embedding to be independently capable of being a general purpose representation and 2dMSE proposes new loss function based on MRL to make each of the first *m* dimensions of each layer of the transformer of the text embedding to be independently capable of being a general purpose representation.

Table 2: The comparison among LLM focused universal text embedding models. Some methods test multiple backbone models, only the best performing ones are listed. LLM gen data indicates whether synthetic data generated by LLMs are used to train the model. The sign - means no information available.

| Models                   | Backbone                                   | Key contributions                                                                                      | Fine-tune strategy   | Fine-tune efficiency                                  | LLM gen data   |
|--------------------------|--------------------------------------------|--------------------------------------------------------------------------------------------------------|----------------------|-------------------------------------------------------|----------------|
| E5mistral7binstruct      | Mistral- 7b                                | Fine-tune decoder only LLMs with a mix of real and synthetic data generated by LLMs                    | LoRA	with rank 16 (42M parameters); Batch	size: 2048                      | 576 GPU hours on V100 GPU (18 hours on 32 V100 GPUs ) | Yes            |
| SFREmbeddingMistral      | Mistral- 7b                                | Multi-task	finetuning over	E5-mistral-7binstruct with improved hard negatives                                                                                                        | LoRA	with rank 8 (21M parameters); Batch	size: 2048                      | 120 GPU hours on A100 GPU (15 hours on 8 A100 GPUs)   | Yes            |
| Echomistral              | Mistral- 7b                                | Use bidirectional attention: repeat the input twice and extract embeddings from the second occurrence. | LoRA	with rank 16 (42M parameters); Batch	size: 2048                      | 192 GPU hours on A100 GPU (two days on 4 A100 GPUs)   | No             |
| LLM2Vec                  | Llama-3 Mistral7b                          | Enabling bidirectional attention + Masked next token prediction + Unsupervised con- trastive learning  | LoRA	with rank 16                      | -                                                     | No             |
| GRIT                     | Mistral7b Mistral- 8x7b                    | Unify generative and embedding tasks by distinguishing between them through instructions               | Batch	size: 2048 for embedding data; 256 for generative data                      | 7B	model:	3072 GPU hours on A100 80GB GPU; 8X7B model: 20,480 GPU hours	on	H100 80GB GPU                                                       | Yes            |
| Gecko                    | gtr-t5-xl (1.2B, encoder from T5-3B model) | Use LLMs to generate Few-shot	Prompted Retrieval	dataset (FRet) to improve text embedding models                                                                                                        | -                    | -                                                     | Yes            |
| gte- Qwen1.57B- instruct | Qwen1.5- 7B                                | Use bidirectional attention along with a vast, multilingual, diverse text corpus                       | -                    | -                                                     | -              |

## 6 LLMs focused universal text embeddings

LLMs are are extensively pre-trained on diverse large quantity of web-scale data, which can be used to improve the universal text embeddings in two different ways as summarized in Table 2. Firstly, LLMs can be used to generate high quality multilingual multi-task synthetic data as demonstrated by researchers from Microsoft and Google (Wang et al., 2023b, 2024; Lee et al., 2024). Secondly, LLMs can be used as backbone for text embeddings as they do not need the contrastive pre-training step used in existing state of the art text embedding models. For example, E5-mistral-7b-instruct perform multi-task fine-tuning on Mistral 7b model which is one of the best performing method on MTEB. Echo-mistral (Springer et al., 2024), LLM2Vec (BehnamGhader et al., 2024), gte-Qwen1.5-7B-instruct (Li et al., 2023) and GRIT (Muennighoff et al., 2024) propose various different solutions so that decoder only LLMs with casual attention can generate high quality text embeddings using bidirectional attention. More details about how different universal text embeddings leverage LLMs to improve their universality can be found below.

**E5-mistral-7b-instruct** E5-mistral-7b-instruct is one of the best performing text embeddings on the MTEB benchmark, which is also a representative text embedding model leveraging LLMs. Firstly, proprietary LLMs including GPT-35-Turbo and GPT-4 are used to generate synthetic data covering a diverse range of text embedding tasks in 93 languages (among which 25% are generated by GPT-35-Turbo and others are generated by GPT-4) (Wang et al., 2023b). In terms of the quality generated data, the authors find that the overall quality is acceptable despite a portion of GPT-35-Turbo outputs do not follow the instructions in the prompt templates strictly. Secondly, pre-trained open source LLM Mistral-7b checkpoint (Jiang et al., 2023) is selected to be fine-tuned on a mixture of synthetic and labeled data (collection of 13 public datasets) with around 1.8M examples after sampling. One advantage of using LLMs such as Mistral (Jiang et al., 2023) for text embedding is that they are extensively pre-trained on web-scale data already, which does not need the contrastive pre-training step used in existing state of the art text embedding models. Given a pre-trained LLM, an [EOS] token is appended to the end of the query and document. The last layer [EOS] vector is used as the text embeddings. To help the model accommodate different tasks, instruction templates (which are used by all LLMs focused universal text embeddings described in this section as well as some of the previously mentioned universal text embeddings such as BGE (Xiao et al., 2023) ) are applied to the original query *q* + to generate a new one *qinst* + given a relevant query-document pair ( *q* + *,d* +):

*qinst* +	= *Instruct* : { *task definition* } \ *n Query* : { *q* +}	(12)

where “task definition” is a placeholder for a one-sentence description of the embedding task added only to the query side but not to the document side (Wang et al., 2023b). Some other details about E5-mistral-7b-instruct include:

- Fine-tuning data: generated synthetic data, ELI5 (Fan, Jernite, Perez, Grangier, Weston, &amp; Auli, 2019) (sample ratio 0.1), HotpotQA (Yang et al., 2018), FEVER (Thorne et al., 2018), MIRACL (Zhang, Thakur, Ogundepo, Kamalloo, Alfonso-Hermelo, Li, Liu, Rezagholizadeh, &amp; Lin, 2023a), MS-MARCO passage ranking (sample ratio 0.5) and document ranking (sample ratio 0.2) (Bajaj et al., 2016), NQ (Karpukhin et al., 2020a), NLI (Bowman et al., 2015), SQuAD (Karpukhin et al., 2020a), TriviaQA (Karpukhin et al., 2020a), Quora Duplicate Questions (Iyer et al., 2017) (sample ratio 0.1), Mr-TyDi (Zhang, Ma, Shi, &amp; Lin, 2021), DuReader (Qiu, Li, Qu, Chen, She, Liu, Wu, &amp; Wang, 2022), and T2Ranking (Xie, Dong, Wang, Lv, Yao, Gan, Wu, Li, Li, Liu, et al., 2023) (sample ratio 0.5) datasets.
- Loss function: standard InfoNCE loss as in Equation 1
- Negative sampling: in-batch negative samples and hard negatives (for the datasets without hard negatives, mE5base (Wang et al., 2024) is used to to mine top 100 hard negatives).
- Model size: 7B (42M trainable parameters using Low-rank adaptation (LoRA) (Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, &amp; Chen, 2021))

The experimental results from (Wang et al., 2023b) shows that even with only synthetic data, the performance of E5-mistral-7b-instruct on MTEB English benchmark is still very competitive. E5-mistral-7b-instruct also has the multilingual capabilities with good performances over high-resource languages. Furthermore, the authors discovered that the method of incorporating instructions has a considerable impact on the performance. Their hypothesis is that the model is better informed about the embedding task at hand through natural language instructions, thereby allowing the model to produce more distinctive embeddings (Wang et al., 2023b).

**SFR-Embedding-Mistral** Built on top of the E5-mistral-7b-instruct, SFR-EmbeddingMistral is also one of the top-ranking universal text embeddings on the MTEB English benchmark with 0.93% performance increase compared to E5-mistral-7b-instruct. The authors summarized their main takeaways in (Rui et al., 2024) (the detailed report is not released) as:

- The retrieval performance of text embeddings significantly improves when integrated with clustering tasks and further enhanced through multi-task knowledge transfer. • Task-homogeneous batching, a method that forms batches from a single task, improves the performance of text embedding by making in-batch negatives more challenging.
- Improving the construction of hard negatives enhances the model’s capacity to accurately identify misleading documents.

To be noted that, the following multi-task datasets are used by SFR-Embedding-Mistral to fine-tune the E5-mistral-7b-instruct model, including

- Retrieval tasks: MS-MARCO, NQ, FiQA, SciFact, NFCorpus, DBPedia, FEVER, HotpotQA, Quora and NLI.
- Clustering tasks: arXiv, bioRxiv, medRxiv, applying filters to exclude development and testing sets in the MTEB clustering framework.
- Classification tasks: AmazonReview, Emotion, MTOPIntent, ToxicConversation , and TweetSentiment.
- Semantic Textual Similarity (STS) tasks: STS12, STS22 , and STSBenchmark
- Reranking tasks: SciDocs and StackOverFlowDupQuestions.

Among the selected training datasets, most are from the MTEB benchmark. Even the development and testing sets are excluded, it might have an unfair advantage comparing to other text embedding methods that do not use the MTEB training data.

**Echo-mistral** Even though constructing text embeddings from autoregressive pretrained LLMs seems promising, the authors from (Springer et al., 2024) identified a striking failure mode of autoregressive language models trained on the next-token objective: the contextualized token embeddings, represented by the vector of last-hidden-layer activations at a specific input token’s position, lack information from tokens appearing later in the sentence because of the causal attention mask. Given the following example provided in (Springer et al., 2024):

- q: [She loves summer] [but dislikes the heat]
- *d* −: [She loves summer] [for the warm evenings]
- *d* +: [She loves summer] [but not the temp]

In this example, the classical LLMs based contextualized embeddings of the first half of *d* − and *d* + are both similar to q because they do not attend to the second half of the sentence, which leads to the overestimation of the similarity between q and *d* − by any pooling strategy that uses information from the first half (Springer et al., 2024).

To mitigate this striking failure mode and take advantage of the bidirectional context information, a simple fix is proposed by presenting the input sentence twice to LLMs. The final contextualized embeddings can then be extracted from the second occurrence of the sentence. LLMs are instructed to undertake basic task such as rewriting or repeating in order to prompt the second occurrence to effectively ”encode” information from the first (Springer et al., 2024). Despite twice the computational cost of classical embeddings, experimental results show that Echo embeddings can improve the LLM based text embedding quality significantly under both zero-shot setting and fine-tuning setting. Some other details about Echo-mistral (echo-mistral-7b-instruct-last) include:

- Fine-tuning data: same as E5-mistral-7b-instruct (Wang et al., 2023b) without synthetic data
- Loss function: standard InfoNCE loss as in Equation 1
- Negative sampling: in-batch negative samples and mined hard negatives
- Model size: 7B

**LLM2Vec** Similar to the idea of Echo-mistral, the authors of (BehnamGhader et al., 2024) believe that the slow adoption of decoder-only LLMs in text embedding tasks is partly due to their causal attention mechanism, which restricts their ability to create bidirectional contextualized representations from encompassing information from the whole input sequence (a necessary trade-off for generative capabilities). Improving the architectural flaw of decoder-only LLMs for text embedding tasks is highly desirable because: 1. decoderonly LLMs are much more sample-efficient than encoder-only models (Clark, Luong, Le, &amp; Manning, 2020); 2. LLMs are supported by a robust ecosystem, including comprehensive tools and well tested pre-training techniques, leading to their continuous enhancement by the community; 3. the good instruction following ability of LLMs (Wang, Mishra, Alipoormolabashi, Kordi, Mirzaei, Arunkumar, Ashok, Dhanasekaran, Naik, Stap, et al., 2022; Ouyang, Wu, Jiang, Almeida, Wainwright, Mishkin, Zhang, Agarwal, Slama, Ray, et al., 2022) makes them ideal for creating universal text embedding models that can handle a wide range of tasks using instructions.

To improve the text embeddings from decoder-only LLMs, LLM2Vec proposes a simple unsupervised approach that can transform any decoder-only LLM into a strong text encoder in three simple steps: 1. enabling bidirectional attention by replacing the causal attention mask of decoder-only LLMs with an all-ones matrix; 2. Masked Next Token Prediction (MNTP): combining next token prediction with masked language modeling (Devlin et al., 2018) to make the model aware of its bidirectional attention; and 3. unsupervised contrastive learning for better sequence representations: the model processes an input sequence twice with independently sampled dropout masks to generate two distinct representations, and is trained to increase the similarity between these representations while decreasing similarity with other sequence representations in the batch (BehnamGhader et al., 2024). Their empirical results show that LLMs can be efficiently converted into universal text embeddings without requiring costly adaptation or synthetic GPT-4 generated data. Some other details about LLM2Vec include:

- Unsupervised training data: English Wikipedia
- Supervised contrastive learning data: adaptations of E5 (Wang et al., 2023b): the public portion of the E5 dataset (Wang et al., 2023b) curated by (Wang et al., 2024)
- Loss function: Contrastive loss, masked next token prediction loss
- Negative sampling: in-batch negatives and hard negatives
- Model size: the best performing model of LLM2Vec on MTEB is LLM2Vec-Mistral7BIns-v2-sup (backbone: Mistral 7B): 7B

**Generative Representational Instruction Tuning (GRIT)** Similar to the idea from Echo-mistral and LLM2Vec, the authors in (Muennighoff et al., 2024) also highlight the importance of bidirectional attention for general purpose universal text embeddings. However, GRIT takes the general purpose model to the next level by training a large language model to handle both generative and embedding tasks (all text-based language problems) distinguished through instructions.

Both representational instruction tuning (Su et al., 2022; Wang et al., 2023b; Asai et al., 2022) and generative instruction tuning (Muennighoff, Wang, Sutawika, Roberts, Biderman, Scao, Bari, Shen, Yong, Schoelkopf, et al., 2022b; Sanh, Webson, Raffel, Bach, Sutawika, Alyafeai, Chaffin, Stiegler, Scao, Raja, et al., 2021; Wei, Bosma, Zhao, Guu, Yu, Lester, Du, Dai, &amp; Le, 2021) are combined into one unified model by GRIT. Firstly, GRIT uses bidirectional attention with mean pooling over the final hidden state to get the text embedding. Contrastive objective with in-batch negatives are used to finetune a pretrained large language model following prior works (Chen, Kornblith, Norouzi, &amp; Hinton, 2020; Gao et al., 2021). The average of the final hidden states of only the input sample is calculated during mean pooling, while disregarding the instruction and format tokens. Nonetheless, these tokens continue to impact the final representation via the self-attention mechanism (Vaswani et al., 2017). Secondly, the language modeling objective of next token prediction (Radford et al., 2018) is used to compute the loss on generative data, where a language modeling head on top of the hidden states predicts the next tokens (Muennighoff et al., 2024). Finally, the representational and generative objectives are summed with optional loss weights. Furthermore, sliding window attention (Child, Gray, Radford, &amp; Sutskever, 2019; Beltagy, Peters, &amp; Cohan, 2020) is used by GRIT to handle generative and embedding inputs of arbitrary length.

The primary drawback of GRIT is its increased computational demand (as shown in Table 2), resulting from the need to train with two objective functions. GRITLM 7B is fine-tuned from Mistral 7B (Jiang et al., 2023) and GRITLM 8x7B (Jiang, Sablayrolles, Roux, Mensch, Savary, Bamford, Chaplot, Casas, Hanna, Bressand, et al., 2024) is finetuned from Mistral 8x7B. Both models have top performance on MTEB English benchmark. GRITLM 7B has better performance than GRITLM 8X7B on embedding tasks, while GRITLM 8X7B is significantly better than GRITLM 7B on generative tasks. The authors provide some hypothesis on the reason why GRIT works on both embedding and generative tasks: 1. Generative language modeling and text embeddings are interconnected, requiring deep understanding of natural language, but expressed differently. 2. The unified model may contain parameters acting as a switch for either embedding tasks or generative tasks (Muennighoff et al., 2024). Some other details about GRIT include:

- Fine-tuning data: adaptations of E5 (Wang et al., 2023b): adding S2ORC (Lo et al., 2019) to increase its scientific data (“E5S”); adaptations of Tu¨lu 2 data (Ivison, Wang, Pyatkin, Lambert, Peters, Dasigi, Jang, Wadden, Smith, Beltagy, et al., 2023): filtering out their custom prompts that contain answers related to the origin of their model.
- Loss function: Contrastive loss with next token prediction loss • Negative sampling: in-batch negative samples and hard negatives
- Model size:
    - GRITLM 7B: 7B
    - GRITLM 8X7B: 47B

**Gecko** LLM based text embeddings have several disadvantages including high computational cost, longer response time and high embedding dimensions (which makes the downstream tasks training also computationally expensive). A recent paper named Versatile Text Embeddings Distilled from Large Language Models (Lee et al., 2024) tries to mitigate these limitations using knowledge distillation from LLMs with synthetic data generation and refinement, where queries are generated from LLMs given the contexts, and their positive and negative passages are mined and refined by LLMs.

The main contribution of (Lee et al., 2024) is designing the two-stage approach that uses LLMs to generate Few-shot Prompted Retrieval dataset (FRet). The first stage is LLMbased Diverse Query Generation: unlike (Wang et al., 2023b), FRet uses LLM to analyze a selected web passage and produce both a description of the task *t* and a pertinent query *q* related to the task:

*LLM* ( *PQG,pseed* ) →− ( *t,q* )	(13)

where *pseed* is a passage drawn randomly from the web corpus C and *PQG* is a fixed few-shot prompt that is identical for every example. By drawing from a variety of free-form task descriptions, LLM is guided to generate a diverse set of queries. These pairs are subsequently utilized to train the embedding models, instructing the models to link a query and its associated instructions with the target passage (Lee et al., 2024). To further encourage the diversity of generated task descriptions and queries, many diverse task descriptions are added in the prompt.

The second stage of FRet is LLM-based positive and negative mining. Unlike previous works (Jeronymo, Bonifacio, Abonizio, Fadaee, Lotufo, Zavrel, &amp; Nogueira, 2023; Dai et al., 2022) assuming that the query *q* generated from a given passage *pseed* forms a good training pair ( *q,pseed* ), the authors hypothesize that there could be a better positive target passage for *q* than *pseed* somewhere in the corpus of web passages as *pseed* is not guaranteed to maximize *P* ( *p* | *q,t* ) over all the passages in the corpus (BehnamGhader et al., 2024). To mine better positives for the generated query, they train an initial embedding model using passage and generated query pairs ( *q,pseed* ) with in-batch negatives. The trained embedding model is used to retrieve top *K* neighbors *P* = { *p* (1) *,...,p* ( *K* )} from the corpus given a generated query *q* . These retrieved passages are ranked by the LLM with two few-shot prompted LLM ranking functions:

- Query Likelihood (QL) (Sachan, Lewis, Joshi, Aghajanyan, Yih, Pineau, &amp; Zettlemoyer, 2022): *QL* ( *q,p* ) = *LLM* ( *q* | *p,* P *QL* ), where P *QL* is a prompt containing an instruction for judging query likelihood and several few-shot examples of relevant query and passage pairs (Drozdov, Zhuang, Dai, Qin, Rahimi, Wang, Alon, Iyyer, McCallum, Metzler, et al., 2023).
- Relevance Classification (RC) (Zhuang, Qin, Hui, Wu, Yan, Wang, &amp; Berdersky, 2023): *RC* ( *q,p* ) = *LLM* ( *label* | *p,* P *RC* ), where P *RC* is a prompt with few-shot examples for grading the relevance of each query-passage pair.

The final ranking function *R* ( *q,p* ) is obtained by combining the rankings from QL and RC with the standard Reciprocal Rank Fusion (RRF) approach (Cormack, Clarke, &amp; Buettcher, 2009). The top ranked passage is then selected as the new positive passage *p* + given the generated query ( *p* + ̸= *pseed* happens for around 15% cases in their dataset). In terms of negative passage selection, they propose two methods: 1. a random nearest neighbor passage that is different from the original passage; 2. the k-th passage in the ranking. The generated FRet dataset has in total 6.6M examples, each containing a task, a query, a positive passage, and a negative passage (Lee et al., 2024). The authors propose a new embedding model Gecko based on a 1.2B parameter pre-trained transformer language model and fine-tuned on the generated FRet dataset, which is one of the top performing text embeddings on the English MTEB benchmark with small embedding dimensions (256 and 768). Some other details about Gecko include:

- Pre-training data: large-scale community QA dataset (Ni, Qu, Lu, Dai, Abrego, Ma,´ Zhao, Luan, Hall, Chang, et al., 2021b) with title-body text pairs from the Web.
- Unified fine-tuning data: FRet (6.6M) along with the following academic training datasets: Natural Questions (Kwiatkowski et al., 2019), HotpotQA (Yang et al., 2018), FEVER (Thorne et al., 2018), MedMCQA (Pal, Umapathi, &amp; Sankarasubbu, 2022), SNLI (Bowman et al., 2015), MNLI (Williams et al., 2017), and several classification datasets from Huggingface. For the multilingual model, training sets from MIRACL (Zhang et al., 2023a) is added.
- Loss function:
    - Pre-training: the contrastive loss **–** Fine-tuning: in-batch cross-entropy loss
- Negative sampling:
    - Pre-training: in-batch negative samples
    - Fine-tuning: in-batch negative samples, hard negatives and same-tower negatives (other queries in the batch) (Moiseev et al., 2023)
- Model size (google-gecko-preview-0409): 1.2B (backbone: gtr-t5-xl (Ni et al., 2021b)) **gte-Qwen1.5-7B-instruct** The authors from GTE (Li et al., 2023) also proposed their LLMs focused universal text embedding model based on Qwen1.5-7B large language model (Bai, Bai, Chu, Cui, Dang, Deng, Fan, Ge, Han, Huang, Hui, Ji, Li, Lin, Lin, Liu, Liu, Lu, Lu, Ma, Men, Ren, Ren, Tan, Tan, Tu, Wang, Wang, Wang, Wu, Xu, Xu, Yang, Yang, Yang, Yang, Yao, Yu, Yuan, Yuan, Zhang, Zhang, Zhang, Zhang, Zhou, Zhou, Zhou, &amp; Zhu, 2023), which is one of the the top-ranking embedding models on both MTEB English and Chinese benchmarks. While the full details of fine-tuning are not disclosed, the authors summarized their key contributions as : 1. the use of bidirectional attention mechanisms to enhance contextual understanding; 2. the use of instruction tuning; 3. the use of a large, multilingual text corpus that covers various domains and scenarios.

**Summary** In this section, the universal text embeddings leveraging LLMs (which make up the majority of the top 10 best performing models on the MTEB English benchmark) are introduced. Most of these models share the finding that LLMs acquire good text representations through comprehensive auto-regressive pre-training, requiring only minimal finetuning to become effective universal text embedding models. E5-mistral-7b-instruct from Microsoft and Gecko from Google DeepMind demonstrate two different ways to generate synthetic data from LLMs in order to improve universal text embeddings, while Echo-mistral (Springer et al., 2024) and LLM2Vec (BehnamGhader et al., 2024) show that good universal text embeddings can be achieved with the focus on enabling the bidirectional attentions for decoder only LLMs without using synthetic data. LoRA is widely used for the fine-tuning of LLMs based universal text embeddings, where LoRA ranks are found not affecting the overall performance substantially in (Wang et al., 2023b). Instructions are used by all LLM focused text embedding models introduced in this section. One of the main reasons is the good instruction following ability of LLMs which makes them ideal for creating universal text embedding models that can handle a wide range of tasks using instructions. From Table 2, it can be told that Mistral-7B model is the most popular backbone model for LLMs focused text embeddings. One of the reasons is that enabling bidirectional attention (even without any training) works well for Mistral-7B. The authors from (BehnamGhader et al., 2024) speculate that Mistral models may be pre-trained with some form bidirectional attention. On the other hand, the full evaluation on MTEB of LLM based universal text embedding models is reported to be computationally expensive: it takes about 3 days on 8 V100 GPUs for E5-mistral-7b-instruct and 40 hours on 8x A100 GPUs for LLM2Vec with Mistral-7B as backbone.

## 7 Analysis on performances and limitations

### 7.1 Overall performance on MTEB English benchmark

Due to the differences in training data, back-bone model, loss function, training strategy, negative-sampling strategy, embedding dimension and so on, it is difficult to have a fair comparison among different text embedding models. But we can still get some insights from the overall performance comparison. The overall performance of the top 25 best text embeddings methods on MTEB English benchmark are shown in Table 3, where the Model size is measured in Million Parameters, #Memory is Memory Usage measured in (GB, fp32), #Embedding is the Embedding dimension. It can be seen that some of the top performing text embeddings are not introduced in this review including voyage-lite-02instruct, voyage-lite-01-instruct, text-embedding-3-large, Cohere-embed-english-v3, Cohereembed-multilingual-v3, ember-v1, sf model e5, etc. The main reason is that these models do not disclose any detailed documentation.

For the models with documentations available, it can be seen that SFR-EmbeddingMistral has the best performance, with the average performance over 56 MTEB datasets of 67.6%. SFR-Embedding-Mistral increases the performance over e5-mistral-7b-instruct by 0.93% by fine-tuning on top of e5-mistral-7b-instruct using more datasets including MTEB training data. GritLM-7B is ranked the 3rd place, outperforming GritLM-8x7B by 1.1%, even though GritLM-8x7B has much more parameters (46.7B parameters) than GritLM-7B (7.2B parameters). To be noted that, GritLM-7B and GritLM-8x7B has unified both text embedding and text generation in the same model, which is different from other text embedding models. Among the top 5 performing text embeddings, google-gecko-preview-0409 and voyage-lite-02-instruct have the smallest parameters (around 1.2B), while google-geckopreview-0409 has the smallest embedding dimension which is favored by downstream tasks. LLM2Vec-Mistral7B-Ins-v2-sup and echo-mistral-7b-instruct-lasttoken both use Mistral 7B as backbone and both focus on making decoder only LLMs use bidirectional attention to get better text embeddings. Even though their performances are similar, LLM2Vec-Mistral7BIns-v2-sup has the advantage of being more computational efficient.

Starting from mxbai-embed-large-v1 ranked at the 9th place till gte-large ranked at 25th place, most text embedding models are BERT based with relatively smaller model size compared to LLM based text embeddings. Both mxbai-embed-large-v1 (rank 9) and UAELarge-V1 (rank 10) propose innovative loss function improvement in the field. GIST-large-

Embedding-v0 (rank 16) is built on top of bge-large-en-v1.5 (rank 17) with improvement Table 3: The top 25 best performing text embeddings methods on MTEB English benchmark. Model size is measured in Million Parameters, #Memory is Memory Usage measured in (GB, fp32), #Embedding is the Embedding dimension. Results can be found from HuggingFace website: https://huggingface.co/spaces/mteb/leaderboard.

| model names                   |   rank | Model Size   | #Memory #Embedding   | #Memory #Embedding   | Max Tokens   |   Average |
|-------------------------------|--------|--------------|----------------------|----------------------|--------------|-----------|
| SFR-Embedding-Mistral         |      1 | 7111         | 26.49                | 4096                 | 32768        |     67.56 |
| voyage-lite-02-instruct       |      2 | 1220         | 4.54                 | 1024                 | 4000         |     67.13 |
| GritLM-7B                     |      3 | 7242         | 26.98                | 4096                 | 32768        |     66.76 |
| e5-mistral-7b-instruct        |      4 | 7111         | 26.49                | 4096                 | 32768        |     66.63 |
| google-gecko-preview- 0409    |      5 | 1200         | 4.47                 | 768                  | 2048         |     66.31 |
| GritLM-8x7B                   |      6 | 46703        | 173.98               | 4096                 | 32768        |     65.66 |
| LLM2Vec-Mistral7B- Ins-v2-sup |      7 | -            | -                    | -                    | -            |     64.8  |
| echo-mistral-7b-instructlast  |      8 | 7111         | 26.49                | 4096                 | 32768        |     64.68 |
| mxbai-embed-large-v1          |      9 | 335          | 1.25                 | 1024                 | 512          |     64.68 |
| UAE-Large-V1                  |     10 | 335          | 1.25                 | 1024                 | 512          |     64.64 |
| text-embedding-3-large        |     11 | -            | -                    | 3072                 | 8191         |     64.59 |
| voyage-lite-01-instruct       |     12 | -            | -                    | 1024                 | 4000         |     64.49 |
| Cohere-embed-englishv3.0      |     13 | -            | -                    | 1024                 | 512          |     64.47 |
| multilingual-e5-largeinstruct |     14 | 560          | 2.09                 | 1024                 | 514          |     64.41 |
| google-gecko-256preview-0409  |     15 | 1200         | 4.47                 | 256                  | 2048         |     64.37 |
| GIST-large-Embeddingv0        |     16 | 335          | 1.25                 | 1024                 | 512          |     64.34 |
| bge-large-en-v1.5             |     17 | 335          | 1.25                 | 1024                 | 512          |     64.23 |
| LLM2Vec-Llama2-7bsup          |     18 | -            | -                    | -                    | -            |     64.14 |
| Cohere-embedmultilingual-v3.0 |     19 | -            | -                    | 1024                 | 512          |     64.01 |
| GIST-Embedding-v0             |     20 | 109          | 0.41                 | 768                  | 512          |     63.71 |
| bge-base-en-v1.5              |     21 | 109          | 0.41                 | 768                  | 512          |     63.55 |
| ember-v1                      |     22 | 335          | 1.25                 | 1024                 | 512          |     63.54 |
| sf model e5                   |     23 | 335          | 1.25                 | 1024                 | 512          |     63.34 |
| mxbai-embed-2d-largev1        |     24 | 335          | 1.25                 | 1024                 | 512          |     63.25 |
| gte-large                     |     25 | 335          | 1.25                 | 1024                 | 512          |     63.13 |

on in-sample selection of negatives as well as the usage of MTEB training data. gtelarge, bge-base-en-v1.5, bge-large-en-v1.5 and multilingual-e5-large-instruct models show the strong performance of BERT based models with smaller model size and embedding dimensions. Among the top 25 text embeddings, google-gecko-256-preview-0409 has the smallest embedding dimension (256) but still has good performance (rank 15).

### 7.2 The universality analysis

The pursuit of developing a unified model to address a multitude of downstream tasks has been long-standing (Li et al., 2023). Despite attempting to be general-purpose in previous models such as (Cer et al., 2018; Raffel et al., 2020; Ni et al., 2021a), studies indicate that these embedding models struggle to generalize across tasks and domains (Lee et al., 2024). In this section, we study whether the MTEB top performing text embeddings are becoming more universal due to the increasing number and improved quality of diverse text datasets across different tasks (Xiao et al., 2023; Asai et al., 2022), good quality synthetic data generated by LLMs (Lee et al., 2024; Wang et al., 2023b) as well as larger backbones such as LLMs.

SimCSE (2021) (Gao et al., 2021) is selected as the baseline method as it is one of the cornerstone work in text embedding which is cited and adopted by most of the recent works. The improvements over different tasks of the top performing text embeddings compared to the baseline method SimCSE is shown in Table 4. Each text embedding model’s performance is divided by the baseline performance in the table: 1 means the model has the same performance as the baseline, larger than 1 value means the model improves the performance of the baseline, smaller than 1 value means the baseline outperforms the model. For the averaged metric, all the top performing text embeddings outperforms the baseline with a considerable gap (SimCSE is ranked 101th place). However, the improvements across different individual tasks are heavily imbalanced:

- Classification tasks: The logistic regression classifier, with a maximum of 100 iterations, is trained using the train set embeddings and its performance is evaluated on the test set (Muennighoff et al., 2022a). It can be seen from Table 4 that all of these top 25 best performing models are better than the baseline method SimCSE with varied improvements between 9% and 21%.
- Clustering tasks: A mini-batch k-means model is trained on the embedded texts, utilizing a batch size of 32 and setting k to match the total number of unique labels with the v-measure as the metric (Muennighoff et al., 2022a). All of the top performing text embedding models outperform the baseline by around 35%-57% increase over the baseline performance.
- Pair Classification (Pair-C): Duplicate or paraphrase pairs with binary labels are embedded and the average precision score based on cosine similarity on text embeddings is used as the main metric (Muennighoff et al., 2022a). The performance of all the top performing text embedding models is superior (with varied improvements between 15% and 20%) to the baseline in the Pair Classification task.
- Reranking tasks: Given a query and a list of relevant and irrelevant reference texts, cosine similarity is used to compare the embeddings and rank the references with MAP Table 4: The improvements over different tasks of the top performing text embeddings compared to the baseline method SimCSE. Each text embedding model’s performance is divided by the baseline performance in the table: 1 means the model has the same performance as the baseline, larger than 1 values means the model improves the performance of the baseline, smaller than 1 values means the baseline outperforms the model. Classi is short for Classification task, Pair-C is short for Pair Classification task and Summa is short for Summarization task in this table.

model names	Avg	Classi	Clustering Pair-	Reranking Retrieval STS	Summa

C

| SFR-Embedding- Mistral            | 1.3824 1.1635 1.5456   |   1.2017 1.2756 |   1.2017 1.2756 |   2.7039 | 1.0749 0.9997   |
|-----------------------------------|------------------------|-----------------|-----------------|----------|-----------------|
| voyage-lite-02-instruct           | 1.3736 1.1772 1.5681   |          1.179  |          1.2251 |   2.594  | 1.0843 0.9949   |
| GritLM-7B                         | 1.3661 1.1803 1.5139   |          1.183  |          1.2724 |   2.6311 | 1.0535 0.9743   |
| e5-mistral-7b-instruct            | 1.3634 1.1656 1.5034   |          1.199  |          1.2665 |   2.6072 | 1.0696 1.0074   |
| google-gecko-preview- 0409        | 1.3569 1.2057 1.4203   |          1.1891 |          1.239  |   2.5527 | 1.0752 1.0468   |
| GritLM-8x7B                       | 1.3436 1.1665 1.4999   |          1.1532 |          1.2579 |   2.5247 | 1.0523 0.9567   |
| LLM2Vec-Mistral7B- Ins-v2-sup     | 1.3260 1.1383 1.3622   |          1.1942 |          1.2289 |   2.566  | 1.0628 0.9612   |
| echo-mistral-7binstruct-lasttoken | 1.3235 1.1502 1.3856   |          1.1854 |          1.223  |   2.5445 | 1.0435 0.9859   |
| mxbai-embed-large-v1              | 1.3235 1.1236 1.3972   |          1.1835 |          1.2644 |   2.4927 | 1.0743 1.0494   |
| UAE-Large-V1                      | 1.3227 1.1227 1.3978   |          1.1842 |          1.2596 |   2.505  | 1.0685 1.0276   |
| text-embedding-3large             | 1.3217 1.1208 1.4660   |          1.1634 |          1.2444 |   2.5408 | 1.0330 0.9599   |
| voyage-lite-01-instruct           | 1.3196 1.1110 1.4179   |          1.1749 |          1.2566 |   2.5472 | 1.0482 0.9936   |
| Cohere-embed-englishv3.0          | 1.3192 1.1362 1.4188   |          1.165  |          1.2202 |   2.5206 | 1.0442 0.9682   |
| multilingual-e5-largeinstruct     | 1.3180 1.1521 1.4089   |          1.1698 |          1.2322 |   2.4047 | 1.0715 0.9750   |
| google-gecko-256preview-0409      | 1.3172 1.1735 1.3482   |          1.1842 |          1.2154 |   2.4033 | 1.0734 1.0382   |
| GIST-large- Embedding-v0          | 1.3166 1.1291 1.3925   |          1.1767 |          1.2631 |   2.4491 | 1.0691 0.9933   |
| bge-large-en-v1.5                 | 1.3143 1.1285 1.3784   |          1.1824 |          1.2627 |   2.4881 | 1.0504 1.0141   |
| LLM2Vec-Llama2-7bsup              | 1.3125 1.1338 1.3533   |          1.1948 |          1.207  |   2.5023 | 1.0583 0.9140   |
| Cohere-embedmultilingual-v3.0     | 1.3098 1.1291 1.3940   |          1.1692 |          1.2171 |   2.4675 | 1.0509 0.9942   |
| GIST-Embedding-v0                 | 1.3037 1.1294 1.3823   |          1.1716 |          1.2488 |   2.3973 | 1.0555 0.9904   |
| bge-base-en-v1.5                  | 1.3004 1.1220 1.3691   |          1.1747 |          1.2381 |   2.4404 | 1.0415 0.9968   |
| ember-v1                          | 1.3002 1.1288 1.3634   |          1.1858 |          1.2629 |   2.3795 | 1.0533 0.9888   |
| sf model e5                       | 1.2961 1.0986 1.3943   |          1.1787 |          1.2592 |   2.374  | 1.0598 1.0141   |
| mxbai-embed-2dlarge-v1            | 1.2943 1.1013 1.3781   |          1.1657 |          1.2398 |   2.3566 | 1.0731 1.0122   |
| gte-large                         | 1.2918 1.0893 1.4011   |          1.1536 |          1.2438 |   2.3932 | 1.0535 1.0157   |

being the main metric (Muennighoff et al., 2022a). The Reranking tasks show an improved performance (between 22% and 28%) from all MTEB leading text embedding models compared to the baseline.

- Retrieval tasks: Given a corpus, queries and a mapping for each query to relevant documents from the corpus, cosine similarity scores on the embeddings between query and documents are used to rank documents for each query, with nDCG@10 being the main metric (Muennighoff et al., 2022a). The most considerable enhancement in the top-rated text embedding models of MTEB is observed in Retrieval tasks, with the majority of these models more than doubling the performance of baseline model. • Semantic Textual Similarity (STS) tasks: Given sentence pairs labeled with continuous scores with higher numbers indicating more similar sentences, Spearman correlation based on cosine similarity between sentence pair embeddings is main metric (Muennighoff et al., 2022a). The increase in performance is moderate in STS tasks compared to other tasks for all top performing MTEB text embedding models, with the best performing model increasing 8.4% over the baseline performance. • Summarization tasks: Given human-written and machine-generated summaries, cosine similarity between embeddings of machine summary and human summary is used to score the machine summaries with Spearman correlation being the main metric (Muennighoff et al., 2022a). Unlike other tasks, most of the top performing text embedding models are not able to outperform the baseline performance on summarization tasks.

From the results in Table 4, it can be seen that compared to the baseline text embedding SimCSE published in 2021, most the top 25 best performing MTEB text embedding models (mostly published in 2023 and 2024) are not remarkably better than the baseline on all tasks, especially on Summarization tasks. All the top 25 text embedding models are notably better than the baseline model on Retrieval, Reranking, Clustering and Pair Classification tasks, especially on Retrieval task. The proposed methodologies appear to primarily impact the performance of retrieval tasks. However, it might be related to the training and fine-tuning datasets used by the top performing models and their similarity to MTEB benchmarks. Popular datasets used by the top performing models include StackExchange, Reddit, S2ORC, NLI (Gao et al., 2021), FEVER (Thorne et al., 2018), NQ (Karpukhin et al., 2020a; Kwiatkowski et al., 2019), HotpotQA (Yang et al., 2018), Quora (Iyer et al., 2017), MSMARCO, etc. These datasets are similar to MTEB benchmark datasets especially on Retrieval and Clustering tasks. Apart from datasets similarity, there are many efforts made by the state of the art embeddings to deal with the asymmetric tasks such as Retrieval, including generation of more synthetic asymmetric datasets as in (Wang et al., 2023b, 2024), instruction based embeddings as in (Springer et al., 2024; Wang et al., 2024, 2023b; BehnamGhader et al., 2024; Muennighoff et al., 2024; Lee et al., 2024; Xiao et al., 2023), asymmetric formatting as in (Lee et al., 2024) and so on. Generally speaking, The results from Table 4 show that: the overall performance on MTEB benchmark are improved considerably by recent advances in universal text embeddings especially on Retrieval tasks while the performance on Summarization task sees no notable improvement compared to the baseline method.

In terms of universality on languages, most of these models are trained on specific languages, typically English, and do not inherently accommodate multilingual data. This lack of language universality restricts their application in global, multilingual contexts. In the work of (Wang et al., 2024, 2023b), the authors use proprietary LLMs to generate synthetic data for a diverse range of text embedding tasks in 93 languages, covering hundreds of thousands of embedding tasks, which shows good performance on high-resource languages. However, for low-resource languages, there is still room for improvement as current opensource LLMs are not adequately pre-trained on them. In terms of the universality on text length, MTEB has Sentence to Sentence (S2S) tasks as well as Paragraph to Paragraph (P2P) tasks where the former only compare titles, while the latter include both title and content (Muennighoff et al., 2022a). For clustering tasks, Arxiv, Biorxiv, Medrxiv, Reddit and StackExchange have both S2S and P2P version, where S2S tasks have short texts with on average 57-115 chars and P2P tasks have long texts with on average 728-1981 chars. Most top performing text embeddings have better performances on P2P tasks on Arxiv, Biorxiv, Medrxiv, Reddit. However, on StackExchange data, most top performing text embeddings have much better performance on S2S tasks. This might be more related to the informativeness nature of datasets instead of to the text length. Better benchmark datasets design related to text length is needed. For example, comparing the clustering performance on long text data before and after different extends of summarization could be an option.

### 7.3 Model efficiency analysis

In the field of AI and NLP, Occam’s Razor could be applied in the process of comparing algorithms or models. If two models perform similarly well, the principle would suggest opting for the simpler one, as it is likely to be more efficient and less prone to overfitting. To compare the efficiency of different text embedding models, the average performance on MTEB English benchmark of state of the art text embedding models and their corresponding model parameters (log wise) are plotted in Figure 4. The efficiency of the downstream tasks using text embedding as input is related to the dimension of the embeddings. Larger embedding dimension indicates higher computational cost, storage/memory cost and latency for downstream tasks. Hence, the embedding dimension for each model is also illustrated in Figure 4, with varying colors denoting different dimensions. The spectrum ranges from light yellow (representing a dimension of 256) to deep red (representing a dimension of 4096). The max token size which is related to the model efficiency when dealing with long input texts is illustrated by different shapes in Figure 4 with: small circle (512/514 max input tokens), triangle (2048 max input tokens), square (4000 max input tokens), pentagon (8192 max input tokens), hexagon (32768 max input tokens).

**Model sizes:** In previous studies (Muennighoff et al., 2022a), it was found that the performance strongly correlates with model size, which can be identified in Figure 4. For example, when the parameters of SGPT increases from 1.3B to 5.8B, the performance increases from 56.2% to 58.93%. Such kind of scaling behavior encourages many studies to scale model size up in order to provide state of the art results across different embedding tasks. Recently, there are more and more models focus on generating text embeddings from LLMs because it does not need the contrastive pre-training step used in existing state of the art text embedding models as LLMs are extensively pre-trained on web-scale data already

<!-- image -->

Figure 4: The top performing text embeddings on MTEB benchmark: X-axis is the average performance over 56 MTEB benchmark datasets, Y-axis is the log of Model parameter numbers (in Millions). Different colors indicate different embedding dimensions and different shapes indicate different max token sizes.

(Wang et al., 2024; Muennighoff et al., 2024; BehnamGhader et al., 2024; Rui et al., 2024; Springer et al., 2024; Lee et al., 2024). However, LLMs are computationally expensive, resource-intensive, and difficult to deploy in real-world applications, particularly on devices with limited processing power. Moreover, the marginal gains in performance do not always justify the substantial increase in parameter size, complexity and resource requirements. Additionally, we can see that when GritLM 7B is scaled up to GritLM 7x8B, the overall performance on MTEB benchmark decreases across all tasks (note that Grit models are both embedding and generation models). The performances of 7B parameters models vary a lot from 57.59% (sgpt-bloom-7b1-msmarco) to 67.56% (SFR-Embedding-Mistral) as shown in Figure 4, while jina-embeddings-v2-small-en achieves better performance than sgpt-bloom-7b1-msmarco with only 33M parameters. Furthermore, the two 1.2B models voyage-lite-02-instruct and google-gecko.text-embedding-preview-0409 demonstrate comparable or superior performances to most 7B LLMs based models, which suggests that there is significant room for enhancement in the efficiency of numerous state of the art text embedding models.

**Embedding sizes:** Deploying text embedding involves two steps: a constant forward pass to compute the embedding, and its use for downstream applications (Sato, 2021; Varma, 2019). The computation costs for the second step rise with the embedding dimensionality, data size, and label space, which can exceed the feature computation cost for large scale systems (Dean et al., 2009; Sun et al., 2017). In some RAG systems where documents are stored as text embedding vectors, the embedding dimension is also related to the storage and memory cost, especially for large scale RAG systems. The top-performing text embedding dimension sizes vary from 256 to 4096, while the largest embedding dimension reported in MTEB English benchmark is 12288 from text-similarity-davinci-001 and textsearch-davinci-001. MRL (Kusupati et al., 2022) and 2dMSE (Li et al., 2024) propose new loss functions to allow first *m* dimensions of the embedding to be independently capable of being a general purpose text embedding too. Among the top performing text embedding models, Gecko (Lee et al., 2024) embeddings are the most compact with google-gecko.textembedding-preview-0409 (768 dimensions) and google-gecko-256.text-embedding-preview0409 (256 dimensions).

**Max token sizes:** The max token size limits the length of the input text to be embedded. When the input length exceeds the max token size, the most straightforward solution is to truncate the input text to the maximum allowed length. However, this approach has the drawback of eliminating potentially relevant text. An alternative strategy involves partitioning the input text into smaller chunks, embedding each chunk separately, and then combine the embeddings of all chunks. Although this method preserves the entirety of the input text, it reduces the efficiency of the embedding model. The max token sizes of top performing text embedding models in MTEB English benchmark vary from 512 to 32768. For BERT like based models, their max token size is usually 512, while text embedding models based on Mistral-7B have the max token size of 32768. To be noted that different LLMs may have different max token sizes. For example, Llama 2 (Touvron, Martin, Stone, Albert, Almahairi, Babaei, Bashlykov, Batra, Bhargava, Bhosale, et al., 2023b) with 7B, 13B, and 70B parameters have a max token size of 4096, which shows that the max token size is not necessarily correlated with model parameter size. Further more, the max token size can be extended in various ways for both LLMs (Zhang, Liu, Xiao, Shao, Ye, &amp; Dou, 2024) and BERT like models (Nussbaum, Morris, Duderstadt, &amp; Mulyar, 2024).

### 7.4 Limitations

Apart from the limitations analyzed above in the previous sections, several other limitations are identified in this section:

**Data:** The complexity of comparing different models arises due to variations in numerous factors such as training data, back-bone model, loss function, training strategy, negativesampling strategy, embedding dimension, among others. It is challenging to establish a fair comparison due to these differences. Few papers analyze the similarity between their training, pre-training or fine-tuning data and the MTEB benchmark datasets which makes it unclear whether MTEB test datasets are in-domain, partially in-domain or out-of-domain for these text embedding models. Many studies claim that the dataset diversity is important to achieve the universal text embeddings (Xiao et al., 2023; Li et al., 2023; Wang et al., 2024; Lee et al., 2024). However, the current literature lacks a metric to accurately measure this dataset diversity, further complicating the issue. This gap in the literature underscores the need for a more rigorous approach to assessing dataset diversity in future studies.

**Instruction:** Instruction refers to the task instruction, which specifies a description of the task that the embedding will be used for (as shown in Equation 12) in order to build universal text embedding models that can generalize across a large variety of tasks (Springer et al., 2024; Wang et al., 2023b; Asai et al., 2022). Many studies have shown that adding instructions has a considerable impact on the performance. However, there are several limitations. Firstly, the effectiveness of the instruction is highly dependent on its quality and specificity. If the instruction is vague or ambiguous, the model may fail to embed the text properly, leading to poor performance on the task. Additionally, creating precise and comprehensive instructions for every possible task can be a labor-intensive and timeconsuming process. Secondly, the model’s ability to interpret and follow the instructions is limited by its current understanding of language, which may not perfectly align with human understanding. This could lead to misinterpretations and errors. Furthermore, the incorporating instructions into text embeddings increases the input length which can be computationally intensive, particularly for large datasets and large models. Finally, few papers explain how instruction impacts the text embedding for symmetric and asymmetric tasks and helps improve the performance theoretically. How out-of-domain instructions impact the model performance is not clear neither.

**Benchmark:** Massive Text Embedding Benchmark (MTEB) is the most popular and used benchmark for universal text embeddings. There are several already identified limitations of MTEB including: lacking long texts datasets (most test datasets MTEB have fewer than 500 chars), task imbalance (15 datasets on Retrieval task, 12 datasets on Classification task while only 1 dataset for Summarization task), limited multi-languages evaluation datasets and no programming language (code) datasets (Muennighoff et al., 2022a). Understanding syntax thoroughly is essential for a text embedding model to accurately determine the relationships between words, which aids in achieving a level of language comprehension that mirrors human cognitive processes (Zhang, Feng, Teng, Liu, &amp; Li, 2023b). The capacity of text embedding models to generalize across various syntactic contexts is not sufficiently examined in the existing benchmark. Therefore, to evaluate the proficiency of text embedding models in understanding syntax, it would be beneficial to incorporate more datasets that focus on syntactic aspects. The variety of datasets can certainly be enhanced. For instance, out of the 11 datasets used for clustering in MTEB, six originate from scientific articles published on platforms like Arxiv, Biorxiv, and Medrxiv. It would be beneficial to include datasets from different fields like finance, business, arts, culture, health, travel, and more to broaden the scope.

**Similarity measures** Distance metrics *d* (· *,* ·) in vector spaces must obey certain axioms or geometric constraints (Cao, Bernard, Sabourin, &amp; Heutte, 2019; Cao, 2019) including:

- Reflexivity: *d* ( **x** *i,* **x** *i* ) = 0
- Nonnegativity: *d* ( **x** *i,* **x** *j* ) ≥ 0
- Symmetry: *d* ( **x** *i,* **x** *j* ) = *d* ( **x** *j,* **x** *i* )
- Triangle inequality: *d* ( **x** *i,* **x** *k* ) ≤ *d* ( **x** *i,* **x** *j* ) + *d* ( **x** *j,* **x** *k* )

Cosine similarity is widely used in the literature and MTEB benchmark to measure similarity between text embeddings, which also obeys symmetry and an analogue of the triangle inequality (Griffiths, Steyvers, &amp; Tenenbaum, 2007). However, psychological representations of similarity do not always obey these constraints. The authors from (Tversky, 1977; Tversky &amp; Hutchinson, 1986) show that some important aspects of human judgments of item similarity can not be captured by some of the geometric axioms of vector spaces. Researchers from (Peterson, Chen, &amp; Griffiths, 2020; Tversky, 1977) demonstrate that human relational similarity judgments violate the geometric constraints of symmetry and the triangle inequality. A famous example in terms of violation of symmetry is that people judge North Korea to be more similar to China than the other way around (Peterson et al., 2020). Furthermore, the authors from (Steck et al., 2024) conclude that cosine-similarity can yield arbitrary and meaningless similarities. Compared to the term of distance or kernel, dissimilarity and similarity are more general terms, which do not have the constraints to be a metric or positive semi-definite (Pekalska, 2005; Cao, Bernard, Sabourin, &amp; Heutte, 2021). New (dis)similarity measures that aligns better with human judgments could be an interesting and important future directions.

## 8 Conclusions

In this article, an overview of the recent advances in universal text embedding models is provided. Various definitions of universal text embeddings from the literature are integrated in this work: universal text embedding is a unified comprehensive text embedding model that can address a multitude of input text length, downstream tasks, domains and languages. The top performing universal text embedding models on MTEB benchmark are categorized into three groups: data focus, loss function focus and LLM focus. Representative works of each category are presented and compared. These state of the art methods have made significant improvements and innovations in terms of training data quantity, quality and diversity; synthetic data generation for universal text embeddings as well as using large language models as backbones. The overall performance on MTEB English benchmark are remarkably improved by these recent universal text embedding models especially on Retrieval, Reranking, Clustering and Pair Classification tasks.

However, there remains a significant gap that needs to be addressed in the current state of the art universal text embedding models. First of all, unlike the considerable improvements on Retrieval tasks, little improvement is made by current state of the art solutions on summarization tasks. Secondly, most of existing text embeddings are trained on specific languages, typically English, and do not inherently accommodate multilingual data. This lack of language universality restricts their application in multilingual contexts. Thirdly, current benchmarks lack domain diversity. Datasets from different fields like finance, business, arts, culture, health, travel, and more with diverse text lengths should be included to broaden the scope and test the domain generalization ability of universal text embedding models.

In terms of future research, there are numerous broad areas that merit further exploration. One such area is the construction of a more comprehensive and diverse benchmark that can test the universality holistically across domains, tasks, input lengths and languages. The redundancy of the benchmark datasets should be minimized to reduce the computational cost of testing. Secondly, developing solutions to make universal text embeddings more sustainable and cost-effective in terms of training, inference and downstream tasks usage is also an interesting direction. Additional future research could focus on in-depth understanding on instructions, its impact on symmetric and asymmetric tasks, its generalization ability and so on. Finally, another interesting future direction could be proposing novel (dis)similarity measures that can produce human-like asymmetries from vector-space text embeddings.

## 9 Acknowledgments

The authors wish to thank Eoin Thomas for his comments and advice. We also thank our anonymous reviewers for their comments.

## 10 References

Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. (2023). Gpt-4 technical report..

AIMeta (2024). Llama 3 model card..

Asai, A., Min, S., Zhong, Z., &amp; Chen, D. (2023). Retrieval-based language models and applications. In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 6: Tutorial Abstracts)* , pp. 41–46.

Asai, A., Schick, T., Lewis, P., Chen, X., Izacard, G., Riedel, S., Hajishirzi, H., &amp; Yih, W.-t. (2022). Task-aware retrieval with instructions..

Bai, J., Bai, S., Chu, Y., Cui, Z., Dang, K., Deng, X., Fan, Y., Ge, W., Han, Y., Huang, F., Hui, B., Ji, L., Li, M., Lin, J., Lin, R., Liu, D., Liu, G., Lu, C., Lu, K., Ma, J., Men,

R., Ren, X., Ren, X., Tan, C., Tan, S., Tu, J., Wang, P., Wang, S., Wang, W., Wu, S., Xu, B., Xu, J., Yang, A., Yang, H., Yang, J., Yang, S., Yao, Y., Yu, B., Yuan, H., Yuan, Z., Zhang, J., Zhang, X., Zhang, Y., Zhang, Z., Zhou, C., Zhou, J., Zhou, X., &amp; Zhu, T. (2023). Qwen technical report..

Bajaj, P., Campos, D., Craswell, N., Deng, L., Gao, J., Liu, X., Majumder, R., McNamara, A., Mitra, B., Nguyen, T., et al. (2016). Ms marco: A human generated machine reading comprehension dataset..

BehnamGhader, P., Adlakha, V., Mosbach, M., Bahdanau, D., Chapados, N., &amp; Reddy, S. (2024). Llm2vec: Large language models are secretly powerful text encoders..

Beltagy, I., Peters, M. E., &amp; Cohan, A. (2020). Longformer: The long-document transformer..

Bojanowski, P., Grave, E., Joulin, A., &amp; Mikolov, T. (2017). Enriching word vectors with subword information. *Transactions of the association for computational linguistics* , *5* , 135–146.

Bowman, S. R., Angeli, G., Potts, C., &amp; Manning, C. D. (2015). A large annotated corpus for learning natural language inference..

Cao, H. (2019). *Random forest for dissimilarity based multi-view learning: application to radiomics* . Ph.D. thesis, Normandie Universit´e; Universit´e du Qu´ebec. Ecole de tech-´ nologie sup´erieure.

Cao, H., Bernard, S., Sabourin, R., &amp; Heutte, L. (2019). Random forest dissimilarity based multi-view learning for radiomics application. *Pattern Recognition* , *88* , 185–197.

Cao, H., Bernard, S., Sabourin, R., &amp; Heutte, L. (2021). A novel random forest dissimilarity measure for multi-view learning. In *2020 25th International Conference on Pattern Recognition (ICPR)* , pp. 1344–1351. IEEE.

Cer, D., Yang, Y., Kong, S.-y., Hua, N., Limtiaco, N., John, R. S., Constant, N., GuajardoCespedes, M., Yuan, S., Tar, C., et al. (2018). Universal sentence encoder..

Chen, T., Kornblith, S., Norouzi, M., &amp; Hinton, G. (2020). A simple framework for contrastive learning of visual representations. In *International conference on machine learning* , pp. 1597–1607. PMLR.

Child, R., Gray, S., Radford, A., &amp; Sutskever, I. (2019). Generating long sequences with sparse transformers..

Choi, H., Kim, J., Joe, S., &amp; Gwon, Y. (2021). Evaluation of bert and albert sentence embedding performance on downstream nlp tasks. In *2020 25th International conference on pattern recognition (ICPR)* , pp. 5482–5487. IEEE.

Clark, K., Luong, M.-T., Le, Q. V., &amp; Manning, C. D. (2020). Electra: Pre-training text encoders as discriminators rather than generators..

Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzm´an, F., Grave, E., Ott, M., Zettlemoyer, L., &amp; Stoyanov, V. (2019). Unsupervised cross-lingual representation learning at scale..

Cormack, G. V., Clarke, C. L., &amp; Buettcher, S. (2009). Reciprocal rank fusion outperforms condorcet and individual rank learning methods. In *Proceedings of the 32nd international ACM SIGIR conference on Research and development in information retrieval* , pp. 758–759.

Dai, Z., Zhao, V. Y., Ma, J., Luan, Y., Ni, J., Lu, J., Bakalov, A., Guu, K., Hall, K. B., &amp; Chang, M.-W. (2022). Promptagator: Few-shot dense retrieval from 8 examples..

Dean, J., et al. (2009). Challenges in building large-scale information retrieval systems. In *Keynote of the 2nd ACM international conference on web search and data mining (WSDM)* , Vol. 10.

Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K., &amp; Harshman, R. (1990). Indexing by latent semantic analysis. *Journal of the American society for information science* , *41* (6), 391–407.

Devlin, J., Chang, M.-W., Lee, K., &amp; Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding..

Drozdov, A., Zhuang, H., Dai, Z., Qin, Z., Rahimi, R., Wang, X., Alon, D., Iyyer, M., McCallum, A., Metzler, D., et al. (2023). Parade: Passage ranking using demonstrations with llms. In *The 2023 Conference on Empirical Methods in Natural Language*

*Processing* .

Fan, A., Jernite, Y., Perez, E., Grangier, D., Weston, J., &amp; Auli, M. (2019). Eli5: Long form question answering..

Gao, T., Yao, X., &amp; Chen, D. (2021).	Simcse: Simple contrastive learning of sentence embeddings..

Gao, T., Yen, H., Yu, J., &amp; Chen, D. (2023). Enabling large language models to generate text with citations..

Griffiths, T. L., Steyvers, M., &amp; Tenenbaum, J. B. (2007). Topics in semantic representation.. *Psychological review* , *114* (2), 211.

Grill, J.-B., Strub, F., Altch´e, F., Tallec, C., Richemond, P., Buchatskaya, E., Doersch, C., Avila Pires, B., Guo, Z., Gheshlaghi Azar, M., et al. (2020). Bootstrap your own latent-a new approach to self-supervised learning. *Advances in neural information processing systems* , *33* , 21271–21284.

Harris, Z. S. (1954). Distributional structure. *Word* , *10* (2-3), 146–162.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., &amp; Chen, W. (2021). Lora: Low-rank adaptation of large language models..

Indurkhya, N., &amp; Damerau, F. J. (2010). *Handbook of natural language processing* . Chapman and Hall/CRC.

Ivison, H., Wang, Y., Pyatkin, V., Lambert, N., Peters, M., Dasigi, P., Jang, J., Wadden, D., Smith, N. A., Beltagy, I., et al. (2023). Camels in a changing climate: Enhancing lm adaptation with tulu 2..

Iyer, S., Dandekar, N., &amp; Csernai, K. (2017). Quora question pairs..

Jeronymo, V., Bonifacio, L., Abonizio, H., Fadaee, M., Lotufo, R., Zavrel, J., &amp; Nogueira, R. (2023). Inpars-v2: Large language models as efficient dataset generators for information retrieval..

Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., Casas, D. d. l., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., et al. (2023). Mistral 7b..

Jiang, A. Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., Chaplot, D. S., Casas, D. d. l., Hanna, E. B., Bressand, F., et al. (2024). Mixtral of experts..

Karpukhin, V., O˘guz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., &amp; Yih, W.-t. (2020a). Dense passage retrieval for open-domain question answering..

Karpukhin, V., O˘guz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., &amp; Yih, W.-t. (2020b). Dense passage retrieval for open-domain question answering..

Kashyap, A. R., Nguyen, T.-T., Schlegel, V., Winkler, S., Ng, S. K., &amp; Poria, S. (2024). A comprehensive survey of sentence representations: From the bert epoch to the chatgpt era and beyond. In *Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)* , pp. 1738–1751.

Kusupati, A., Bhatt, G., Rege, A., Wallingford, M., Sinha, A., Ramanujan, V., HowardSnyder, W., Chen, K., Kakade, S., Jain, P., et al. (2022). Matryoshka representation learning. *Advances in Neural Information Processing Systems* , *35* , 30233–30249.

Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M., Parikh, A., Alberti, C., Epstein, D., Polosukhin, I., Devlin, J., Lee, K., et al. (2019). Natural questions: a benchmark for question answering research. *Transactions of the Association for Computational Linguistics* , *7* , 453–466.

Lan, Z., Chen, M., Goodman, S., Gimpel, K., Sharma, P., &amp; Soricut, R. (2019). Albert: A lite bert for self-supervised learning of language representations..

Lee, J., Dai, Z., Ren, X., Chen, B., Cer, D., Cole, J. R., Hui, K., Boratko, M., Kapadia, R., Ding, W., et al. (2024). Gecko: Versatile text embeddings distilled from large language models..

Li, R., Zhao, X., &amp; Moens, M.-F. (2022). A brief overview of universal sentence representation methods: A linguistic view. *ACM Computing Surveys (CSUR)* , *55* (3), 1–42.

Li, X., &amp; Li, J. (2023). Angle-optimized text embeddings..

Li, X., Li, Z., Li, J., Xie, H., &amp; Li, Q. (2024). 2d matryoshka sentence embeddings..

Li, X., Li, Z., Xie, H., &amp; Li, Q. (2021). Merging statistical feature via adaptive gate for improved text classification. In *Proceedings of the AAAI conference on artificial intelligence* , Vol. 35, pp. 13288–13296.

Li, Z., Zhang, X., Zhang, Y., Long, D., Xie, P., &amp; Zhang, M. (2023). Towards general text embeddings with multi-stage contrastive learning..

Liu, Q., Kusner, M. J., &amp; Blunsom, P. (2020). A survey on contextual embeddings..

Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., &amp; Stoyanov, V. (2019a). Roberta: A robustly optimized bert pretraining approach..

Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., &amp; Stoyanov, V. (2019b). Roberta: A robustly optimized BERT pretraining approach. *CoRR* , *abs/1907.11692* .

Lo, K., Wang, L. L., Neumann, M., Kinney, R., &amp; Weld, D. S. (2019). S2orc: The semantic scholar open research corpus..

Long, D., Gao, Q., Zou, K., Xu, G., Xie, P., Guo, R., Xu, J., Jiang, G., Xing, L., &amp; Yang, P. (2022). Multi-cpr: A multi domain chinese dataset for passage retrieval. In *Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval* , pp. 3046–3056.

Manning, C. D., Raghavan, P., &amp; Schu¨tze, H. (2008). *Introduction to information retrieval* . Cambridge university press.

Mikolov, T., Chen, K., Corrado, G., &amp; Dean, J. (2013). Efficient estimation of word representations in vector space..

Moiseev, F., Abrego, G. H., Dornbach, P., Zitouni, I., Alfonseca, E., &amp; Dong, Z. (2023). Samtone: Improving contrastive loss for dual encoder retrieval models with same tower negatives..

Muennighoff, N., Su, H., Wang, L., Yang, N., Wei, F., Yu, T., Singh, A., &amp; Kiela, D. (2024). Generative representational instruction tuning..

Muennighoff, N., Tazi, N., Magne, L., &amp; Reimers, N. (2022a). Mteb: Massive text embedding benchmark..

Muennighoff, N., Wang, T., Sutawika, L., Roberts, A., Biderman, S., Scao, T. L., Bari, M. S., Shen, S., Yong, Z.-X., Schoelkopf, H., et al. (2022b). Crosslingual generalization through multitask finetuning..

Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K., &amp; Zettlemoyer, L. (2018). Deep contextualized word representations..

Ni, J., Abrego, G. H., Constant, N., Ma, J., Hall, K. B., Cer, D., &amp; Yang, Y. (2021a). Sentence-t5: Scalable sentence encoders from pre-trained text-to-text models..

Ni, J., Qu, C., Lu, J., Dai, Z., Abrego, G. H., Ma, J., Zhao, V. Y., Luan, Y., Hall, K. B.,´ Chang, M.-W., et al. (2021b). Large dual encoders are generalizable retrievers..

Nussbaum, Z., Morris, J. X., Duderstadt, B., &amp; Mulyar, A. (2024). Nomic embed: Training a reproducible long context text embedder..

Oord, A. v. d., Li, Y., &amp; Vinyals, O. (2018). Representation learning with contrastive predictive coding..

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. (2022). Training language models to follow instructions with human feedback. *Advances in neural information processing systems* , *35* , 27730–27744.

Pal, A., Umapathi, L. K., &amp; Sankarasubbu, M. (2022). Medmcqa: A large-scale multisubject multi-choice dataset for medical domain question answering. In *Conference on health, inference, and learning* , pp. 248–260. PMLR.

Patil, R., Boit, S., Gudivada, V., &amp; Nandigam, J. (2023). A survey of text representation and embedding techniques in nlp..

Pekalska, E. M. (2005). Dissimilarity representations in pattern recognition. concepts, theory and applications...

Pennington, J., Socher, R., &amp; Manning, C. D. (2014). Glove: Global vectors for word representation. In *Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP)* , pp. 1532–1543.

Peterson, J. C., Chen, D., &amp; Griffiths, T. L. (2020). Parallelograms revisited: Exploring the limitations of vector space models for simple analogies. *Cognition* , *205* , 104440.

Petukhova, A., Matos-Carvalho, J. P., &amp; Fachada, N. (2024a). Text clustering with llm embeddings..

Petukhova, A., Matos-Carvalho, J. P., &amp; Fachada, N. (2024b). Text clustering with llm embeddings..

Qiu, Y., Li, H., Qu, Y., Chen, Y., She, Q., Liu, J., Wu, H., &amp; Wang, H. (2022). Dureader retrieval: A large-scale chinese benchmark for passage retrieval from web search engine..

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. (2021). Learning transferable visual models from natural language supervision. In *International conference on machine learning* , pp. 8748–8763. PMLR.

Radford, A., Narasimhan, K., Salimans, T., Sutskever, I., et al. (2018). Improving language understanding by generative pre-training..

Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., &amp; Liu, P. J. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of machine learning research* , *21* (140), 1–67.

Rajapakse, T. C. (2023). Dense passage retrieval: Architectures and augmentation methods. In *Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval* , pp. 3494–3494.

Reimers, N., &amp; Gurevych, I. (2019). Sentence-bert: Sentence embeddings using siamese bert-networks..

Ren, R., Lv, S., Qu, Y., Liu, J., Zhao, W. X., She, Q., Wu, H., Wang, H., &amp; Wen, J.R. (2021). Pair: Leveraging passage-centric similarity relation for improving dense passage retrieval..

Rui, M., Ye, L., Shafiq Rayhan, J., Caiming, X., Yingbo, Z., &amp; Semih, Y. (2024). Sfrembedding-mistral:enhance text retrieval with transfer learning. Salesforce AI Research Blog.

Sachan, D. S., Lewis, M., Joshi, M., Aghajanyan, A., Yih, W.-t., Pineau, J., &amp; Zettlemoyer, L. (2022). Improving passage retrieval with zero-shot question generation..

Sanh, V., Debut, L., Chaumond, J., &amp; Wolf, T. (2019). Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter..

Sanh, V., Webson, A., Raffel, C., Bach, S. H., Sutawika, L., Alyafeai, Z., Chaffin, A., Stiegler, A., Scao, T. L., Raja, A., et al. (2021). Multitask prompted training enables zero-shot task generalization..

Sato, T. K. (2021). Vertex ai matching engine..

Sean, L., Aamir, S., Darius, K., &amp; Julius, L. (2024). Open source strikes bread - new fluffy embeddings model..

Selva Birunda, S., &amp; Kanniga Devi, R. (2021). A review on word embedding techniques for text classification..

Solatorio, A. V. (2024). Gistembed: Guided in-sample selection of training negatives for text embedding fine-tuning..

Springer, J. M., Kotha, S., Fried, D., Neubig, G., &amp; Raghunathan, A. (2024). Repetition improves language model embeddings..

Steck, H., Ekanadham, C., &amp; Kallus, N. (2024). Is cosine-similarity of embeddings really about similarity?..

Su, H., Shi, W., Kasai, J., Wang, Y., Hu, Y., Ostendorf, M., Yih, W.-t., Smith, N. A., Zettlemoyer, L., &amp; Yu, T. (2022). One embedder, any task: Instruction-finetuned text embeddings..

Sun, C., Shrivastava, A., Singh, S., &amp; Gupta, A. (2017). Revisiting unreasonable effectiveness of data in deep learning era. In *Proceedings of the IEEE international conference on computer vision* , pp. 843–852.

Sun, Z., Deng, Z.-H., Nie, J.-Y., &amp; Tang, J. (2019). Rotate: Knowledge graph embedding by relational rotation in complex space..

Suresh, V., &amp; Ong, D. C. (2021). Not all negatives are equal: Label-aware contrastive loss for fine-grained text classification..

Thorne, J., Vlachos, A., Christodoulopoulos, C., &amp; Mittal, A. (2018). Fever: a large-scale dataset for fact extraction and verification..

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. (2023a). Llama 2: Open foundation and fine-tuned chat models..

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. (2023b). Llama 2: Open foundation and fine-tuned chat models..

Tversky, A. (1977). Features of similarity.. *Psychological review* , *84* (4), 327.

Tversky, A., &amp; Hutchinson, J. (1986). Nearest neighbor analysis of psychological spaces.. *Psychological review* , *93* (1), 3.

Varma, M. (2019). Extreme classification. *Communications of the ACM* , *62* (11), 44–45.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L ., &amp; Polosukhin, I. (2017). Attention is all you need. *Advances in neural information processing systems* , *30* .

Wang, L., Yang, N., Huang, X., Jiao, B., Yang, L., Jiang, D., Majumder, R., &amp; Wei, F. (2022). Text embeddings by weakly-supervised contrastive pre-training..

Wang, L., Yang, N., Huang, X., Yang, L., Majumder, R., &amp; Wei, F. (2023a). Improving text embeddings with large language models..

Wang, L., Yang, N., Huang, X., Yang, L., Majumder, R., &amp; Wei, F. (2023b). Improving text embeddings with large language models..

Wang, L., Yang, N., Huang, X., Yang, L., Majumder, R., &amp; Wei, F. (2024). Multilingual e5 text embeddings: A technical report..

Wang, S., Zhou, W., &amp; Jiang, C. (2020a). A survey of word embeddings based on deep learning. *Computing* , *102* (3), 717–740.

Wang, W., Bao, H., Huang, S., Dong, L., &amp; Wei, F. (2020b). Minilmv2: Multi-head selfattention relation distillation for compressing pretrained transformers..

Wang, Y., Mishra, S., Alipoormolabashi, P., Kordi, Y., Mirzaei, A., Arunkumar, A., Ashok, A., Dhanasekaran, A. S., Naik, A., Stap, D., et al. (2022). Super-naturalinstructions: Generalization via declarative instructions on 1600+ nlp tasks..

Wei, J., Bosma, M., Zhao, V. Y., Guu, K., Yu, A. W., Lester, B., Du, N., Dai, A. M., &amp; Le, Q. V. (2021). Finetuned language models are zero-shot learners..

Williams, A., Nangia, N., &amp; Bowman, S. R. (2017). A broad-coverage challenge corpus for sentence understanding through inference..

Wu, Y., Schuster, M., Chen, Z., Le, Q. V., Norouzi, M., Macherey, W., Krikun, M., Cao, Y., Gao, Q., Macherey, K., et al. (2016). Google’s neural machine translation system: Bridging the gap between human and machine translation..

Xiao, S., Liu, Z., Zhang, P., &amp; Muennighof, N. (2023). C-pack: Packaged resources to advance general chinese embedding..

Xie, X., Dong, Q., Wang, B., Lv, F., Yao, T., Gan, W., Wu, Z., Li, X., Li, H., Liu, Y., et al. (2023). T2ranking: A large-scale chinese benchmark for passage ranking. In *Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval* , pp. 2681–2690.

Xiong, L., Xiong, C., Li, Y., Tang, K.-F., Liu, J., Bennett, P., Ahmed, J., &amp; Overwijk, A. (2020). Approximate nearest neighbor negative contrastive learning for dense text retrieval..

Xu, L., Xie, H., Li, Z., Wang, F. L., Wang, W., &amp; Li, Q. (2023). Contrastive learning models for sentence representations. *ACM Transactions on Intelligent Systems and Technology* , *14* (4), 1–34.

Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W. W., Salakhutdinov, R., &amp; Manning, C. D. (2018). Hotpotqa: A dataset for diverse, explainable multi-hop question answering..

Yue, Z., Kratzwald, B., &amp; Feuerriegel, S. (2021). Contrastive domain adaptation for question answering using limited text corpora..

Zhang, H., Li, Z., Xie, H., Lau, R. Y., Cheng, G., Li, Q., &amp; Zhang, D. (2022). Leveraging statistical information in fine-grained financial sentiment analysis. *World Wide Web* , *25* (2), 513–531.

Zhang, P., Liu, Z., Xiao, S., Shao, N., Ye, Q., &amp; Dou, Z. (2024). Soaring from 4k to 400k: Extending llm’s context with activation beacon..

Zhang, X., Ma, X., Shi, P., &amp; Lin, J. (2021). Mr. tydi: A multi-lingual benchmark for dense retrieval..

Zhang, X., Thakur, N., Ogundepo, O., Kamalloo, E., Alfonso-Hermelo, D., Li, X., Liu, Q., Rezagholizadeh, M., &amp; Lin, J. (2023a). Miracl: A multilingual retrieval dataset covering 18 diverse languages. *Transactions of the Association for Computational Linguistics* , *11* , 1114–1131.

Zhang, Y., Feng, Z., Teng, Z., Liu, Z., &amp; Li, H. (2023b). How well do text embedding models understand syntax?..

Zhu, Y., Kiros, R., Zemel, R., Salakhutdinov, R., Urtasun, R., Torralba, A., &amp; Fidler, S. (2015). Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. In *Proceedings of the IEEE international conference on computer vision* , pp. 19–27.

Zhuang, H., Qin, Z., Hui, K., Wu, J., Yan, L., Wang, X., &amp; Berdersky, M. (2023). Beyond yes and no: Improving zero-shot llm rankers via scoring fine-grained relevance labels..