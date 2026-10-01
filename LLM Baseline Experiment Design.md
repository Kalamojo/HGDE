**LLM Baseline Experiment Design**

Models: Gemini Pro & GPT 5.6

General Principle:

- The LLMs should have the exact same information and feedback that is given to our projection model  
- Feedback comes in the form of knowing what should be in each cluster and what was clustered correctly

Given:  
The LLM will cluster according to its own decided pattern, and the “user” that is in this experiment will have different criteria by which they want to cluster by. The LLM will not know this, nor will it know what that different criterion is.

Dataset:  
We will use the exact same dataset we are gonna test our projection model on.

Steps:

1. Upload the dataset to the LLM  
2. Prompt: “Cluster the given dataset”  
3. Choose 2 items that are together under the user-defined criteria   
   

It’s worth trying 2 different experiments from here:   
One with the chosen 2 items being from the same cluster already  
Another with the chosen 2 items being from 2 different clusters

4. Prompt: “\<item a\> and \<item b\> should be clustered together, now recluster”  
5. After reclustering, go to the cluster in which \<item a\> and \<item b\> are together. Note down everything that was clustered correctly in there.  
6. Prompt: “\<item a\>, … \<item n\> are correctly clustered. \<item x\> should also be in that cluster, now recluster.”  
7. Repeat steps 5 and 6 until user is satisfied.

Edge case:

8. If the original defined cluster becomes perfectly satisfied, but other clusters are still incorrect, then randomly choose an incorrect cluster to correct  
9. Prompt: “\<name the clusters that are correct, then identify the incorrect cluster you want to correct, name what is correct in there right now, and one item that should also be in there that isn’t yet. Then ask for a reclustering.\>”