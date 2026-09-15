# Pinterest But Better

# Project Title: ???

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


**Methodology:**

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
