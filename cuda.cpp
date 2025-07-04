#include <stdio.h>
#include <cuda_runtime.h>  // Incluye las funciones de CUDA

#define N 1048576         // 1M elementos
#define THREADS_PER_BLOCK 256  // Tamaño típico de bloque

// Kernel CUDA que se ejecuta en la GPU
__global__ void add_vectors(float *A, float *B, float *C) {
    // Cálculo del índice global único del thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Verificación de límites para evitar accesos fuera del array
    if (i < N) {
        C[i] = A[i] + B[i];  // Cada thread calcula un elemento
    }
}

int main() {
    // Punteros para memoria en CPU (host)
    float *A, *B, *C;
    // Punteros para memoria en GPU (device)
    float *d_A, *d_B, *d_C;
    
    // Reserva de memoria en CPU
    A = (float*)malloc(N * sizeof(float));
    B = (float*)malloc(N * sizeof(float));
    C = (float*)malloc(N * sizeof(float));
    
    // Inicialización de vectores
    for (int i = 0; i < N; i++) {
        A[i] = (float)i;
        B[i] = (float)(i * 2);
    }
    
    // 1. Reserva de memoria en GPU
    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));
    cudaMalloc((void**)&d_C, N * sizeof(float));
    
    // 2. Transferencia de datos CPU → GPU
    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // 3. Configuración de la ejecución del kernel
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // Creación de eventos CUDA para medir tiempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 4. Ejecución del kernel con temporización
    cudaEventRecord(start);
    add_vectors<<<blocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C);
    cudaEventRecord(stop);
    
    // 5. Transferencia de resultados GPU → CPU
    cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Cálculo del tiempo de ejecución
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Tiempo GPU (CUDA): %.6f segundos\n", milliseconds / 1000);
    
    // Verificación de resultados
    printf("Primeros 10 elementos del resultado:\n");
    for (int i = 0; i < 10; i++) {
        printf("C[%d] = %.1f + %.1f = %.1f\n", i, A[i], B[i], C[i]);
    }
    
    // 6. Liberación de memoria
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);
    
    return 0;
}