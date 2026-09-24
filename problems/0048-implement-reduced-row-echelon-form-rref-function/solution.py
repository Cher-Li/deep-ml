import numpy as np

# mostly from 68
def rref(A):
    A = A.astype(np.float32)
    n, m = A.shape

    pivot_row = 0  # fix 1: Track pivot row independently from column i
    for i in range(m):  # Loop through columns (m), not row count (n)
        if pivot_row >= n:
            break

        # Find non-zero element in column i starting from pivot_row
        nonzero = np.nonzero(A[pivot_row:, i])[0] + pivot_row
        if len(nonzero) == 0:
            continue  # fix 2: Skip division when no pivot exists in this column

        # Swap row containing pivot to pivot_row position
        A[[pivot_row, nonzero[0]]] = A[[nonzero[0], pivot_row]]

        # Normalize pivot row
        A[pivot_row] = A[pivot_row] / A[pivot_row, i]

        # Eliminate column i entries in other rows
        for j in range(n):
            if j != pivot_row:
                A[j] -= A[pivot_row] * A[j, i]

        pivot_row += 1  # Advance to next row for the next pivot

    return A