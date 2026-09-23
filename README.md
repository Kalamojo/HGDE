# Human-Guided Dynamic Embeddings

# Research Problem
When working with large datasets, oftentimes only limited subsets of the data are actually needed for a given task. If desired labels do not already exist, then it can be quite difficult to retrieve or organize by certain attributes.
Clustering is often the most simple way to create partitions of such data, but it comes with limitations. Most methods are unsupervised, and so it’s often difficult or impossible to tailor clustering towards desired objectives for a given user, outside of hard must-link and cannot-link constraints. If users have vague groupings in mind, and especially if they’re dealing with large amounts of data, cluster adaptation becomes quite a daunting task.
To address this use-case, hope to introduce a framework for human-guided dynamic embeddings that is not dependent on any specific clustering method. Rather, it will transform existing data embeddings into a space personalized by human constraints/suggestions, implicitly resulting in cluster adaptations that converge to desired groups efficiently and effectively.

# Literature Review
There is a lot of research being put into the field of multimodal embeddings such as Gemini Embedding 2 and ImageBind. The Gemini model builds off of the multi-modal capabilities of Gemini to embed audio, video, image, and textual data in one unified representation space [1]. ImageBind can handle audio, video, image, text, thermal and IMU data and it was trained by binding each example of every datatype to an image, thus the name ImageBind [2]. These multimodal embeddings pair very well with LLMs, boast great zero-shot capabilities, and potential future work in RAG.
Research has been put into different methods of clustering as well. [3] presents the idea that the optimal (most compact) clustering may not be the most accurate or useful way to cluster data. They present many different options for clustering and let the user decide what cluster is best. The idea of user feedback affecting the clusters found is also introduced in [4]. By allowing the user a certain number of merge and split requests for different clusters, they found an optimal algorithm to recluster based on these requests.
Fine tuning embeddings for specific purposes is also a research area with high demand. One popular method of fixing embeddings is Contrastive Learning Penalty (CLP) which is a number added to some loss function to help push similar data points together in an embedding space, and dissimilar data points further away. [5]
The final research area we are dealing with is the measurement of personalization. Because personalization is a subjective metric, as far as we know, there are no objective metrics from which to measure this. The closest thing to this Personalized Predictive Model where, similar to Recommendation engines, they compare one input to other inputs similar to it in the training data to decide how to weight the various features of the input [6]. This has shown to outperform one-size-fits-all models.
Where the current research lacks is a combination of these topics and an adequate answer to personalization. Optimal, human-feedback oriented clustering has focused on unimodal data and user described features to focus on. Using these techniques on multimodal data in a self-supervised way present challenges to be solved like clustering based on features that not all data points have, and clustering on features that have been implied by the user but not explicitly stated. Additionally, measuring how personalized an algorithm is for a specific person is undefined without strictly defined labels (like whether or not they liked the recommendation).

# Project Goals
Develop a model and framework for adapting content embeddings to human preferences, specifically in the context of multimodal dataset organization
Formulate cluster personalization as a mathematical online problem, where success is measured by converging to desired grouping of items in as few rounds as possible, and use this formulation to evaluate our proposed framework
Release an open source tool and/or UI that allows for the dynamic clustering of many data types using an intuitive interface for making cluster change suggestions
Design an experiment to evaluate whether our framework effectively learns human preferences on clustering. An LLM will be used as a benchmark to compare our final results to

# Project Scope
## Datasets
Bullapedia Pokedex Wiki (Scraped)
Source: https://bulbapedia.bulbagarden.net/wiki/List_of_Pok%C3%A9mon_by_National_Pok%C3%A9dex_number
Size: 1025 pairs of ~600x600 px images and 1 paragraph of text (exact size in MB to be determined)
We will scrape the images and biology descriptions from Bullapedia Pokemon encyclopedia to create image text pairs of Pokemon. This will result in image text pairs for embedding Pokemon, with which we will base our clustering evaluations on: for some Pokemon, we will use only the image embeddings, while for others only text embeddings will be used. Effective embeddings and embedding adaptations should be able to represent either in ways suitable for clustering.
## Models
Embedding Models:
Gemini Embedding 2
ImageBind
Embedding Transformation Models:
MLP
Deep Q Network
Baseline Frontier Models:
Gemini Pro
GPT 5.6
## Compute
The only model we will be training is our embedding transformation model, which should also require minimal training examples (as a requirement for the effectiveness of this framework). We estimate standard CPU use, and no more than a 6 GB GPU, is necessary for these purposes (which we currently have).
We may need a more capable GPU (like an AWS instance or a good local graphics card) to process our initial dataset through ImageBind. We already have access to an AWS account with compute credits for these purposes

# Approach
We will use Python and PyTorch for all of our machine learning code. We will implement our scraping job (to create pokemon dataset) in Python. We will use an Embedding Model to get the initial shared embeddings and then use an Embedding Transformation Model to transform into personalized embeddings. We will cluster and then learn based on human guidance until the transformation model is able to learn what how the user wants to cluster. We will experiment with the use of different models and clustering methods and evaluate to determine which combination works the best.

We will also make use of a few algorithms/learning methods/models for the scope of our research:
Contrastive Learning
To derive a loss value to train our embedding adaptation model, we will utilize some form of triplet loss to bring items in the same desired groups closer to each other, and items from different groups further. This is used to train the embedding transformation model.
Q-Learning (alternative)
As an alternative approach for contrastive learning for embedding transformation, we will also explore ways of framing the problem as a reinforcement learning setting, with each new suggestion from users supplying the reward r_t+1, the continuous options for embedding perturbations as actions a_t, the existing item embeddings and/or clusters are state s_t, and the learned Deep Q Network and its weights as the current policy pi
Clustering
We will test a number of clustering algorithms to apply on our generated embedding vectors. This will include K-Means, DBSCAN, COP-K-Means, Hierarchical, and/or Spectral Clustering.

Process: Embedding Model -> Transformation Model -> Contrastive Learning -> Clustering Algorithm -> Human-Guided Feedback -> Retrain

# GitHub Repo
https://github.com/Kalamojo/HGDE

# Validation
Unit tests will be built in blocks similar to the model itself. Firstly, we will test our basic data pre-processing and downloaded models by inputting our data and seeing the vectors and clusters it produces. Then, to test our cluster adjustment model, we will give the model vectors and measure how it changes the values, making sure it keeps the dimensionality and transforms the values in some way. Then we will make a test to input data into the embedder, find the initial cluster, then make an adjustment and view the recluster to test the functionality pipeline. To test the training, we will input similar data points that the embedder initially separated, and see the adjuster pull them closer together and vice versa with dissimilar points. Finally we will test whether the agents are able to use the tool by giving them a dataset and instructions, and evaluate the model from end to end.
How do we measure the performance of an unsupervised model? Well we can have a dataset with labels, we just never let the model see the labels, even in training. We can measure the precision and recall of the model using a formula very similar to image segmentation
Recall = Пi=classes(1-Пj=clusters(1-(number of i in j)/(total number of i)))
Precision = Пi=classes(1-Пj=clusters(1-(number of i in j)/(total number items in j)))
The key difference with the clustering is that, say we have 10 cats and 10 dogs that we want to cluster in different buckets. It does not matter if the 10 cats end up in bucket A or bucket B, it just matters that they are in a different bucket from the 10 dogs. With these formulas, the model will achieve a perfect recall of 1.0 if every object of class A is with every other object of class A, and same with the other classes, and the model will achieve a perfect precision of 1.0 if every cluster contains just one class. We will combine these terms to measure the accuracy
Accuracy = Precision*Recall
We want to achieve the highest accuracy we can with the least amount of user input, in the following, an “iteration” is one user input, followed by reclustering based on that input. The following formula decreases at an exponential rate because accuracy ∊ [0,1] and iterations >= 1.
This will be our success metric that we aim to maximize
Dynamic accuracy = (accuracy)iterations
In addition to accuracy, we want some way to measure how personalized our model is to each individual using it. This will be measured by the alteration of the embeddings. This metric will NOT be used to optimize the model but it will be monitored to see how user input affects the model.
Personalization = Σf∊F|w0f - w1f|
Where F is the set of features of the embedded vectors, w0 is the initial value and w1 is the value after the final iteration.

# Deliverables
A Python codebase containing our interactive clustering loop, including the custom projection model and baseline clustering algorithms.
An automated simulation testing suite featuring a synthetic user that generates graphs, ideally proving our projection model minimizes required manual edits.
A final research report and evaluation benchmark detailing our methodology and plotting the number of user moves against clustering accuracy.

Timeline, Milestones, and Work Split
The work is split between our 5 team members so that we can work in parallel without waiting on each other.
Makilan (Data & Embeddings): Responsible for curating the dataset and extracting the initial base embeddings using foundation models like ImageBind or Gemini.
Jay (Unsupervised Clustering): Responsible for implementing baseline algorithms like K-Means to find optimal clusters before and after user edits.
John (Projection Model): Responsible for building the PyTorch model and loss functions that warp the embedding space based on explicit "Must-Link" constraints.
Chris (Simulation & Evaluation): Responsible for writing the synthetic user script that applies hidden ground-truth rules, generates automated user moves, and tracks benchmark metrics like Adjusted Rand Index for the final convergence graphs.
Kolade (Project Lead & Integration): Responsible for overseeing project architecture, stitching the individual modules into a continuous pipeline, and providing cross-functional engineering support to all team members.

# Rough Timeline:
Weeks 1-3: Set up the code environment, define clustering metrics, finalize data contracts, and curate the initial datasets.
Weeks 4-6: Extract the real foundation embeddings, get the baseline clustering running, and draft the core PyTorch projection model.
Weeks 7-9: Stitch the pipeline together for the midterm MVP, then connect the simulator to run the clustering loop autonomously.
Weeks 10-12: Run tests to compare projection architectures and generate graphs tracking user moves versus accuracy.
Weeks 13-14: Document the final codebase, format the data tables, and write the final research report.

# Why a code agent cannot just do our project
We’ll answer this in 2 parts, the first being why an LLM cannot just do what we’re trying to do, and the latter being why a coding agent cannot just code up the project in one go.

1. Why an LLM cannot just do the clustering task
LLMs are massive and rigid. While an LLM might adjust its output for a single conversation, this does not permanently update its underlying mathematical weights. Furthermore, an LLM cannot natively warp a continuous embedding space to match a subjective human vibe. So, unlike standard models that personalize by comparing a user to millions of others or by relying on massive text prompts, our project builds a lightweight mathematical filter based purely on simple user clicks.

2. Why a coding agent cannot just code the entire project
Coding agents are great at building standard apps, but they cannot perform novel machine learning research on their own. Writing a custom projection model involves complex, high-dimensional math. If the training loop breaks and groups every item into one giant cluster, or a batch of multimodal data causes a silent tensor shape error, an AI cannot intuitively debug it. Finally, AI is highly susceptible to "circular testing." If a coding agent writes a test for its own custom loss function, it could just write a test inherently guaranteed to pass. Designing an unbiased synthetic user simulation and interpreting research graphs requires genuine human scientific intuition.

# Citations
Shanbhogue et al. Gemini Embedding 2: A Native Multimodal Embedding Model from Gemini. URL: https://arxiv.org/pdf/2605.27295
Girdhar et al. ImageBind: One Embedding Space To Bind Them All. URL: https://arxiv.org/pdf/2305.05665
Caruana et al. Meta Clustering. URL: https://www.cs.cornell.edu/~caruana/ICDM06.metaclust.caruana.pdf
Balcan and Blum. Clustering with Interactive Feedback. URL: https://www.cs.cmu.edu/~ninamf/papers/split-merge.pdf
Yu. Efficient Fine-Tuning Methodology Of Text Embedding Models For Information Retrieval: Contrastive Learning Penalty (CLIP). URL: https://arxiv.org/pdf/2412.17364
Krikella and Dubin. A General Mixture Loss Function to Optimize a Personalized Predictive Model. URL: https://arxiv.org/pdf/2601.20788
Viswanathan et al. Large Language Models Enable Few-Shot Clustering. URL: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00648/120476
