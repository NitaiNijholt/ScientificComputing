import numpy as np
from numba import njit

@njit
def is_point_in_bounds(c,i,j):
    imax,jmax = c.shape
    return bool(i >= 0 and i < imax and j >= 0 and j < jmax)

def get_neighboors(c,i,j,N):
    # Get neighboors if possible
    # Later on additional conditions will be added here for sinks and so on...

    # Connect values in columns!!!

    if j==0:
        # Left column
        # value for j-1 is now the last column (j==N-1)!
        c1 = c[i+1,j] if is_point_in_bounds(c,i+1,j) else 0
        c2 = c[i-1,j] if is_point_in_bounds(c,i-1,j) else 0
        c3 = c[i,j+1] if is_point_in_bounds(c,i,j+1) else 0
        c4 = c[i,N-1]
    elif j==(N-1):
        # Right column
        # Value for j+1 is now the first column (j=0)
        c1 = c[i+1,j] if is_point_in_bounds(c,i+1,j) else 0
        c2 = c[i-1,j] if is_point_in_bounds(c,i-1,j) else 0
        c3 = c[i,0]
        c4 = c[i,j-1] if is_point_in_bounds(c,i,j-1) else 0
    else:
        c1 = c[i+1,j] if is_point_in_bounds(c,i+1,j) else 0
        c2 = c[i-1,j] if is_point_in_bounds(c,i-1,j) else 0
        c3 = c[i,j+1] if is_point_in_bounds(c,i,j+1) else 0
        c4 = c[i,j-1] if is_point_in_bounds(c,i,j-1) else 0

    return c1,c2,c3,c4


def are_neighboors_insulating(insulating_mask,i,j,N):
    # insulating_mask
    are_they_insulating = np.zeros(4)

    # First neighboor, in i+1,j
    are_they_insulating[0] = insulating_mask[i+1,j] if is_point_in_bounds(insulating_mask,i+1,j) else insulating_mask[0,j]
    # Second, in i-1,j
    are_they_insulating[1] = insulating_mask[i-1,j] if is_point_in_bounds(insulating_mask,i-1,j) else insulating_mask[N-1,j]
    # Third, in i,j+1
    are_they_insulating[2] = insulating_mask[i,j+1] if is_point_in_bounds(insulating_mask,i,j+1) else insulating_mask[i,0]
    # Fourth, in i,j-1
    are_they_insulating[3] = insulating_mask[i,j-1] if is_point_in_bounds(insulating_mask,i,j-1) else insulating_mask[i,N-1]

    return are_they_insulating

@njit
def insulated_contributions(c1,c2,c3,c4,insulating_mask,c,c_k,i,j):
    n1 = c_k[i+1,j] if c1 != 0 else 0
    n2 = c[i,j]  if c2 != 0 else 0
    n3 = c_k[i,j] if c3 != 0 else 0
    n4 = c[i,j] if c4 != 0 else 0
    return n1,n2,n3,n4


# Update the SOR function to have sinks and insulating materials

def update_sor(c,insulating_mask,N,w):
    c_k = c.copy()

    for i,j in np.ndindex(c.shape):
        # If point is in sink or insulating material, it has zero
        if insulating_mask[i,j] == 1:
            c[i,j] = 0
        elif i==0:
            c[i,j] = 1
        elif i==(N-1):             
            c[i,j] = 0
        else:
            # Other cases
            c1,_,c3,_ = get_neighboors(c_k,i,j,N)
            # Values at i-1 and j-1 are used as soon as they are calculated
            # So we take them from c already!
            _,c2,_,c4 = get_neighboors(c,i,j,N)
            # Get how many neighboors are insulating
            insulated_neighboors = np.sum(are_neighboors_insulating(insulating_mask,i,j,N))
            
            # Insulated terms
            insulated_contributions = insulated_neighboors*c[i,j]
           
            c[i,j] = (w/4)*(c1+c2+c3+c4+insulated_contributions) + (1-w)*c_k[i,j]
    return c


@njit
def update_sor_opt(c,insulating_mask,N,w):
    c_k = c.copy()
    

    for i,j in np.ndindex(c.shape):
        # If point is in sink or insulating material, it has zero
        if insulating_mask[i,j] == 1:
            c[i,j] = 0
        elif i==0:
            c[i,j] = 1
        elif i==(N-1):             
            c[i,j] = 0
        else:
            # Other cases

            imax,jmax = c.shape

            # GET NEIGHBORS
            #c1,_,c3,_ = get_neighboors(c_k,i,j,N)
            # Values at i-1 and j-1 are used as soon as they are calculated
            # So we take them from c already!
            #_,c2,_,c4 = get_neighboors(c,i,j,N)
            # Get how many neighboors are insulating
            if j==0:
                # Left column
                # value for j-1 is now the last column (j==N-1)!
                            
                c1 = c[i+1,j] if bool(i+1 >= 0 and i+1 < imax and j >= 0 and j < jmax) else 0
                #c2 = c[i-1,j] if is_point_in_bounds(c,i-1,j) else 0
                c3 = c[i,j+1] if bool(i >= 0 and i < imax and j+1 >= 0 and j+1 < jmax) else 0
                #c4 = c[i,N-1]
            elif j==(N-1):
                # Right column
                # Value for j+1 is now the first column (j=0)
                c1 = c[i+1,j] if bool(i+1 >= 0 and i+1 < imax and j >= 0 and j < jmax) else 0
                #c2 = c[i-1,j] if is_point_in_bounds(c,i-1,j) else 0
                c3 = c[i,0]
                #c4 = c[i,j-1] if is_point_in_bounds(c,i,j-1) else 0
            else:
                c1 = c[i+1,j] if bool(i+1 >= 0 and i+1 < imax and j >= 0 and j < jmax) else 0
                #c2 = c[i-1,j] if is_point_in_bounds(c,i-1,j) else 0
                c3 = c[i,j+1] if bool(i >= 0 and i < imax and j+1 >= 0 and j+1 < jmax) else 0
                #c4 = c[i,j-1] if is_point_in_bounds(c,i,j-1) else 0

            #c1,c2,c3,c4

            if j==0:
                # Left column
                # value for j-1 is now the last column (j==N-1)!
                #c1 = c[i+1,j] if is_point_in_bounds(c,i+1,j) else 0
                c2 = c[i-1,j] if  bool(i+1 >= 0 and i+1 < imax and j >= 0 and j < jmax) else 0
                #c3 = c[i,j+1] if is_point_in_bounds(c,i,j+1) else 0
                c4 = c[i,N-1]
            elif j==(N-1):
                # Right column
                # Value for j+1 is now the first column (j=0)
                #c1 = c[i+1,j] if is_point_in_bounds(c,i+1,j) else 0
                c2 = c[i-1,j] if bool(i+1 >= 0 and i+1 < imax and j >= 0 and j < jmax) else 0
                #c3 = c[i,0]
                c4 = c[i,j-1] if bool(i >= 0 and i < imax and j-1 >= 0 and j-1 < jmax) else 0
            else:
                #c1 = c[i+1,j] if is_point_in_bounds(c,i+1,j) else 0
                c2 = c[i-1,j] if bool(i-1 >= 0 and i-1 < imax and j >= 0 and j < jmax)  else 0
                #c3 = c[i,j+1] if is_point_in_bounds(c,i,j+1) else 0
                c4 = c[i,j-1] if bool(i >= 0 and i < imax and j-1 >= 0 and j-1 < jmax) else 0


            # get if neighbors insulating
            are_they_insulating = np.zeros(4)

            # First neighboor, in i+1,j
            imax,jmax = insulating_mask.shape

            # bool(i >= 0 and i < imax and j-1 >= 0 and j-1 < jmax)

            are_they_insulating[0] = insulating_mask[i+1,j] if bool(i+1 >= 0 and i+1 < imax and j >= 0 and j < jmax) else insulating_mask[0,j]
            # Second, in i-1,j
            are_they_insulating[1] = insulating_mask[i-1,j] if bool(i-1 >= 0 and i-1 < imax and j >= 0 and j < jmax)  else insulating_mask[N-1,j]
            # Third, in i,j+1
            are_they_insulating[2] = insulating_mask[i,j+1] if bool(i >= 0 and i < imax and j+1 >= 0 and j+1 < jmax)  else insulating_mask[i,0]
            # Fourth, in i,j-1
            are_they_insulating[3] = insulating_mask[i,j-1] if bool(i >= 0 and i < imax and j-1 >= 0 and j-1 < jmax)  else insulating_mask[i,N-1]


            insulated_neighboors = np.sum(are_they_insulating)
            
            # Insulated terms
            insulated_contributions = insulated_neighboors*c[i,j]
           
            c[i,j] = (w/4)*(c1+c2+c3+c4+insulated_contributions) + (1-w)*c_k[i,j]
    return c
