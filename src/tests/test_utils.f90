
module test_utils
  use numeric_kinds, only: dp

  use matrix_free, only: free_matmul
  use array_utils, only: generate_diagonal_dominant

  implicit none
  
contains

  function apply_mtx_to_vect(input_vect) result (output_vect)
    !> \brief Function to compute the optional mtx on the fly
    !> \param[in] i column/row to compute from mtx
    !> \param vec column/row from mtx
    
    complex (dp), dimension(:,:), intent(in) :: input_vect
    complex (dp), dimension(size(input_vect,1),size(input_vect,2)) :: output_vect

    output_vect = free_matmul(compute_matrix_on_the_fly,input_vect)

  end function apply_mtx_to_vect

  function apply_stx_to_vect(input_vect) result (output_vect)
    !> \brief Function to compute the optional mtx on the fly
    !> \param[in] i column/row to compute from mtx
    !> \param vec column/row from mtx
    
    complex (dp), dimension(:,:), intent(in) :: input_vect
    complex (dp), dimension(size(input_vect,1),size(input_vect,2)) :: output_vect

    output_vect = free_matmul(compute_stx_on_the_fly,input_vect)

  end function apply_stx_to_vect


  function compute_matrix_on_the_fly(i, dim) result (vector)
    !> \param[in] i index of the i-th column
    !> \param[in] dim dimension of the resulting column
    !> \return the i-th column of a square matrix of dimension dim
    
    integer, intent(in) :: i, dim
    complex(dp), dimension(dim) :: vector

    ! call expensive function
    vector = expensive_function_1(i, dim)

    ! set the diagonal value
    vector(i) = vector(i) + i

  end function compute_matrix_on_the_fly


  function compute_stx_on_the_fly(i, dim) result (vector)
    !> \param[in] i index of the i-th column
    !> \param[in] dim dimension of the resulting column
    !> \return the i-th column of a square matrix of dimension dim

    integer, intent(in) :: i, dim
    complex(dp), dimension(dim) :: vector

    ! call expensive function
    vector = expensive_function_2(i, dim)

    ! Set diagonal value equal to 1
    vector(i) = 1d0
    
  end function compute_stx_on_the_fly



  function expensive_function_1(i, dim) result (vector)
    !> expensive function to test matrix free version

    integer, intent(in) :: i, dim
    complex(dp), dimension(dim) :: vector
    
    ! local variable
    integer :: j
    real(dp) :: x, y
    
    x = exp(real(i)/real(dim))
    
    do j=1,dim
       y = exp(real(j)/real(dim))
       if (j >= i) then
          vector(j) = cos(log(sqrt(atan2(x, y)))) * 1e-4
       else
          vector(j) = cos(log(sqrt(atan2(y, x)))) * 1e-4
       endif
    end do
    
  end function expensive_function_1

  function expensive_function_2(i, dim) result (vector)
    !> expensive function to test matrix free version
    
    integer, intent(in) :: i, dim
    complex(dp), dimension(dim) :: vector
    
    ! local variable
    integer :: j
    real(dp) :: x, y
    
    x = exp(real(i)/real(dim))
    
    do j=1,dim
       y = exp(real(j)/real(dim))
       if (j >= i) then
          vector(j) = sin(log(sqrt(atan2(x, y)))) * 1e-4
       else
          vector(j) = sin(log(sqrt(atan2(y, x)))) * 1e-4
       endif
    end do
    
  end function expensive_function_2

  function read_matrix(path_file, dim) result(mtx)
    !> read a row-major square matrix from a file
    !> \param path_file: path to the file
    !> \param dim: dimension of the square matrix
    !> \return matrix 

    character(len=*), intent(in) :: path_file
    integer, intent(in) :: dim
    complex(dp), dimension(dim, dim) :: mtx
    real(dp), dimension(dim, dim) :: r_mtx
    integer :: i
    
    open(unit=3541, file=path_file, status="OLD")
    do i=1,dim
       read(3541, *) r_mtx(i, :)
       mtx(i, :) = r_mtx(i, :)
    end do
    close(3541)
    
  end function read_matrix

  
  subroutine write_vector(path_file, vector)
    !> Write vector to path_file
    character(len=*), intent(in) :: path_file
    real(dp), dimension(:), intent(in) :: vector
    integer :: i

    open(unit=314, file=path_file, status="REPLACE")
    do i=1,size(vector)
       write(314, *) vector(i)
    end do
    close(314)
    
  end subroutine write_vector


  subroutine write_matrix(path_file, mtx)
    !> Write matrix to path_file
    character(len=*), intent(in) :: path_file
    complex(dp), dimension(:, :), intent(in) :: mtx
    integer :: i, j

    open(unit=314, file=path_file, status="REPLACE")
    do i=1,size(mtx, 1)
       do j=1,size(mtx, 2)
          write(314, *) mtx(i, j)
       end do
    end do
    close(314)
    
  end subroutine write_matrix
  
end module test_utils

