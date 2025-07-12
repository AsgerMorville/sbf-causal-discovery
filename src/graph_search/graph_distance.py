import numpy as np
import rpy2.robjects as robjects
from cdt.metrics import SHD


def graph_to_r_mat(U):
    p = U.p
    U = U.G.tolist()
    U = [item for sublist in U for item in sublist]
    U = robjects.FloatVector(U)
    U = robjects.r['matrix'](U, nrow=p,ncol=p,byrow=True)
    return U

def struct_interv_dist(G,H):
    """
    This function calculates the structural intervention distance between graphs G and H.
    """

    # Convert Graph objects into R-matrices
    G = graph_to_r_mat(G)
    H = graph_to_r_mat(H)

    # Define the R-function from the SID-library
    sid_R = robjects.r('''
    library(SID)
    func <- function(x,y){return(structIntervDist(x,y)$sid)}
    func
    ''')
    return np.asarray(sid_R(G,H))[0]

def struct_hamming_dist(G, H):
    return SHD(G.G,H.G)