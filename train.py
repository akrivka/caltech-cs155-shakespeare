import sys
from HMM import unsupervised_HMM

# read from shakespeare_processed.txt
with open("data/shakespeare_processed.txt") as file:
    X = []
    for line in file:
        X.append([int(x) for x in line.strip().split()])

# create the HMM
# read from command line
N_states = int(sys.argv[1])
N_iters = int(sys.argv[2])
hmm = unsupervised_HMM(X, N_states, N_iters)

# export
hmm.export(f"models/hmm-{N_states}-{N_iters}.txt")
