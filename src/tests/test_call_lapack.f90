program main
  use numeric_kinds, only: dp
  use test_utils, only:  write_matrix, write_vector
  use davidson, only: generate_diagonal_dominant, lapack_generalized_eigensolver, lapack_qr
  
  implicit none

  integer, parameter :: dim = 50
  real(dp), dimension(dim) :: eigenvalues
  real(dp), dimension(dim, dim) :: eigenvectors
  real(dp), dimension(dim, dim) :: copy, mtx, stx

  ! mtx = read_matrix("tests/matrix.txt", 100)
  mtx = generate_diagonal_dominant(dim, 1d-3)
  copy = mtx
  stx = generate_diagonal_dominant(dim, 1d-3)
  call write_matrix("test_lapack_matrix.txt", mtx)  
  call write_matrix("test_lapack_stx.txt", stx)  

  ! standard eigenvalue problem
  call lapack_generalized_eigensolver(copy, eigenvalues, eigenvectors)

  call write_vector("test_lapack_eigenvalues.txt",eigenvalues)
  call write_matrix("test_lapack_eigenvectors.txt",eigenvectors)

  ! General eigenvalue problem
  call lapack_generalized_eigensolver(copy, eigenvalues, eigenvectors, stx)
  call write_vector("test_lapack_eigenvalues_gen.txt",eigenvalues)
  call write_matrix("test_lapack_eigenvectors_gen.txt",eigenvectors)

  ! Lapack orthonormalization
  call lapack_qr(mtx)
  call write_matrix("test_lapack_qr.txt", mtx)
  
end program main
