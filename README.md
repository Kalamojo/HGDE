# Pinterest But Better

# Project Title: ???

# Background

# Literature Review
There is a lot of research being put into the field of multimodal embeddings such as Gemini Embedding 2. This model builds off of the multi-modal capabilities of Gemini to embed audio, video, image, and textual data in one unified representation space [1]. Another example is ImageBind, a model from Meta. This model can handel audio, video, image, text, thermal and IMU data. Each data type in training was paired with an image, showing that they did not need a convoluted and bloated dataset where every data type was paired to each other, instead each type was bound to an image [2]. These multimodal embeddings pair very well with LLMs, boast great zero-shot capabilities, and potential future work in RAG.

Research has been put into different methods of clustering as well. [3] (meta clustering) presents the idea that the optimal (most compact) clustering may not be the most accurate or useful way to cluster data. They experimented with a technique where they present many different options for clustering and let the user decide what data to cluster by. The idea of user feedback affecting the clusters found is also introduced in [4] (clustering with interactive feedback. This one was math heavy it seems so I may have missed something important). By allowing the user a certain number of merge and split requests for different clusters, they found an optimal algorithm to recluster based on these requests.

Fine tuning embeddings for specific purposes is also a research area with high demand. One popular method of fixing embeddings is Contrastive Learning Penalty (CLP) which is a number added to some loss function to help push similar data points together in an embedding space, and disimilar data points further away. [5] (CLP)

The final research area we are dealing with is the measurement of personalization. Because personalization is a subjective metric, as far as we know, there are no objective metrics from which to measure this. The closest thing to this Personalized Predictive Model where, similar to Reccomendation engines, they compare one input to other inputs similar to it in the training data to decide how to weight the various features of the input [6]. This has shown to outperform one-size-fits-all models.

Where the current research lacks is a combination of these topics and an adequate answer to personalization. Optimal, human-feedback oriented clustering has focused on single-modal data and self described features to focus on. Using these techniques on multimodal data in a self-supervised way present challenges to be solved like clustering based on feature that not all datapoints have, and clustering on features that have been implied by the user but not explicitly stated. Additionally, measuring how personalized an algorithm is for a specific person is undefined without strictly defined labels (like whether or not they liked the reccomendation).

# Project Goals

**Project Objective:** To develop a framework for semi-automatic data clustering that is receptive to change suggestions by users. The resulting product should allow for the dynamic clustering and search of desired subgroups in a number of datasets across multiple modalities (text, image, video, audio, etc.). If time permits, we also hope to develop further specialized applications utilizing this technique

**Features:**
Different Modalities
Text / PDF
Images
Audio
Videos
Adjust from Human Feedback
LLM-Generated Labels of Clusters
Hierarchial Clustering??

**Possible Models:**
Gemini Embedding 2
ImageBind


# Approach

User input (multimodial files)
Gemini Embedding 2 to get files to a general shared embedding space
Neural Network for projecting the embedding to a personalized embedding space
Contrastive Learning
Make clusters
K-Means
DBSCAN
COP-K-Means
Agglomerative for Hierarchial
Learning
Triplet Learning
Active Learning
Train the Neural Network
Recluster


Goal: Learn the users objective / what features matter to them

# Documentation
This project will progressively be documented on GitHub. Installation, testing, and any front-end application that has been developed will be documented in this repository.

# Validation Methods

# Deliverables

# Justification
A lot of research has been put into each research area seperately, multimodal embeddings, personalized clustering, fine-tuning embeddings, and personalization metrics, but research is lacking that combines these different topics together for practical uses. Combining these complicated ideas takes a knowledge and creativity. This specific idea of dynamic clustering is too novel for an LLM to do on its own or create a project for. An LLM can admit when it's wrong but because of its large size and stability, correcting one example will not make an observable difference in the LLM's reasoning or embedding space. 

# Timeline
