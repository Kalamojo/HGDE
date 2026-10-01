**LLM Baseline Experiment Design**

Models: Gemini Pro & GPT 5.6

General Principle:

* The LLMs should have the exact same information and feedback that is given to our projection model.


* Feedback is strictly limited to identifying what was clustered incorrectly using a single pair of items per turn. The LLM is never told what it did correctly.

Given:
The LLM will cluster according to its own decided pattern, and the "user" that is in this experiment will have different criteria by which they want to cluster by. The LLM will not know this, nor will it know what that different criterion is.

Dataset:
We will use the exact same dataset we are gonna test our projection model on.

Steps:

1. Upload the dataset to the LLM.


2. Prompt: "Cluster the given dataset".


3. Scan the LLM's output to find the first instance where two items that belong in the same group were separated into different clusters.
4. Prompt: "<item a> and <item b> should be clustered together, now recluster".


5. Evaluate the new clustered output. Find the next instance where two items that belong together are separated.
6. Prompt: "<item c> and <item d> should be clustered together, now recluster."
7. Repeat steps 5 and 6 until user is satisfied.