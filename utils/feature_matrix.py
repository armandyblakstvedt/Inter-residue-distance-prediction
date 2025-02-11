import torch


def get_feature_matrix(protein_sequence, distance_matrix, dimension):
    """
    Converts a one-hot encoded protein sequence into a feature matrix.
    Feature matrix: A tensor of shape (2 * num_unique_amino_acids, L, L)

    Args:
    protein_sequence: List of amino acids in the protein.
    distance_matrix: Distance matrix of the protein.
    dimension: The dimension of an amino acid in the one-hot encoded vector.
    """
    # Calculate protein length based on one_hot_encoded vector size.
    L = protein_sequence.numel() // dimension
    # Reshape flat one-hot vector into (L, num_unique_amino_acids)
    one_hot_matrix = protein_sequence.view(L, dimension)
    device = one_hot_matrix.device
    ones = torch.ones(L, device=device)

    A = torch.einsum('ik,j->ijk', one_hot_matrix, ones)   # shape: (L, L, num_unique_amino_acids)
    B = torch.einsum('i,jk->ijk', ones, one_hot_matrix)      # shape: (L, L, num_unique_amino_acids)

    # Concatenate along the last dimension and permute to get channels-first
    feature_matrix = torch.cat((A, B), dim=2).permute(2, 0, 1)

    return feature_matrix, distance_matrix
