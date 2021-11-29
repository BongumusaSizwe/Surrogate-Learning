# Architecture 1

This architecture uses the CNN implemented in the deepmind paper. We explore the effects of different hyperparameters for this architecture in the reinforement learning space and in the supervised learning space.

In this folder
- there is a script, "trainer.sh", which trains agents, save models and generate the models to be tested on the Pong game. 

- test.py is responsible for testing the rl agent performance and thus generating the game score. The game score is determined by the reward collected over time.

The different models are found in the models folder which consists of the following:
1. Batch Size
2. Learning Rate

# Architecture Details

    - convlayer1
    - forwardlayer1(relu activation)
    - convlayer2
    - forwardlayer2(relu activation)
    - convlayer3
    - flattening layer
    - forwardlayer3(relu activation)
    - linear layer

Total number of layers: 8 layers

# Results Summary
1. Batch Size
\begin{table}[]
\begin{tabular}{|l|l|l|l|l|l|l|}
\hline
Batch Size & 4 & 8 & 16 & 32 & 64 & 128 \\ \hline
SL Score   &   &   &    &    &    &     \\ \hline
RL Score   &   &   &    &    &    &     \\ \hline
\end{tabular}
\end{table}


2. Learning Rate

\begin{table}[]
\begin{tabular}{|l|l|l|l|l|l|l|}
\hline
\begin{tabular}[c]{@{}l@{}}Learning\\ Rate\end{tabular} & 1E-1 & 1E-2 & 1E-3 & 1E-4 & 1E-5 & 1E-6 \\ \hline
SL Score                                                &      &      &      &      &      &      \\ \hline
RL Score                                                & -21  &  -21 & -21  & 12.7 & 0.9  &  -21 \\ \hline
\end{tabular}
\end{table}
