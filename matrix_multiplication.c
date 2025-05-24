#include <stdlib.h>
#include <stdio.h>

int* matrix_generator(int width, int height) {
	int size = width * height;
	int* arr = (int*) malloc(size * sizeof(int));
	if (arr == NULL) {
		return 0;
	}
	for (int i = 0; i < size; i++) {
		arr[i] = 0;
	}
	return arr;
}

int* sample_matrix_generator(int width, int height) {
	int size = width * height;
	int* arr = (int*) malloc(size * sizeof(int));
	if (arr == NULL) {
		return 0;
	}
	for (int i = 0; i < size; i++) {
		arr[i] = i;
	}
	return arr;
}

int* matrix_multiplication(int mat1[], int mat2[], int mat1_width, int mat1_height, int mat2_width, int mat2_height) {
	if (mat1_width != mat2_height) {
		return 0;
	}
	
	int* result = matrix_generator(mat1_height, mat2_width);
	int size = mat1_height * mat2_width;

	for (int i = 0; i < mat1_height; i++) {
		for (int j = 0; j < mat2_width; j++) {
			for (int k = 0; k < mat1_width; k++) {
				result[(i * mat2_width) + j] += mat1[(i * mat1_width) + k] * mat2[(k * mat2_width) + j];
			}
		}
	}

	return result;
}

void print_matrix(int* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main() {
	int mat1_width = 4;
	int mat1_height = 2;
	int mat2_width = 5;
	int mat2_height = 4;
	
	int* mat1 = sample_matrix_generator(mat1_width, mat1_height);
	int* mat2 = sample_matrix_generator(mat2_width, mat2_height);
	if (!mat1 || !mat2) {
        	printf("Memory allocation failed!\n");
        	free(mat1);
        	free(mat2);
        	return 1;
    	}
	
    	printf("Original Matrix 1 (%dx%d):\n", mat1_height, mat1_width);
    	print_matrix(mat1, mat1_height, mat1_width);

    	printf("\nOriginal Matrix 2 (%dx%d):\n", mat2_height, mat2_width);
    	print_matrix(mat2, mat2_height, mat2_width);
	
	int* result = matrix_multiplication(mat1, mat2, mat1_width, mat1_height, mat2_width, mat2_height);
	printf("\nResultant matrix memory address: %p\n", result);

    	if (!result) {
        	printf("\nMultiplication failed: Dimension mismatch!\n");
    	} else {
        	printf("\nResult Matrix (%dx%d):\n", mat1_height, mat2_width);
        	print_matrix(result, mat1_height, mat2_width);
    	}
	
	free(mat1);
	free(mat2);
	free(result);
	return 0;
}

