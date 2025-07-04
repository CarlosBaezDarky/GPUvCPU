#include <stdio.h>
#include <stdlib.h>
#include <omp.h>  // Incluye la biblioteca OpenMP

#define N 1048576 // 1M elementos (2^20)

// Función que suma dos vectores usando paralelismo con OpenMP
void vector_add_cpu(float *A, float *B, float *C) {
    // Directiva OpenMP para paralelizar el bucle
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];  // Operación elemental de suma
    }
}

int main() {
    // Reserva de memoria en el heap para los vectores
    float *A = (float*)malloc(N * sizeof(float));
    float *B = (float*)malloc(N * sizeof(float));
    float *C = (float*)malloc(N * sizeof(float));

    // Inicialización de vectores con valores de prueba
    for (int i = 0; i < N; i++) {
        A[i] = (float)i;       // Vector A contiene 0, 1, 2, ..., N-1
        B[i] = (float)(i * 2); // Vector B contiene 0, 2, 4, ..., 2*(N-1)
    }

    // Medición del tiempo con función OpenMP
    double start = omp_get_wtime();  // Obtiene tiempo de reloj de alta precisión
    vector_add_cpu(A, B, C);         // Llama a la función paralelizada
    double end = omp_get_wtime();    // Detiene el cronómetro

    printf("Tiempo CPU (OpenMP): %f segundos\n", end - start);

    // Verificación de resultados (solo muestra los primeros 10 elementos)
    printf("Primeros 10 elementos del resultado:\n");
    for (int i = 0; i < 10; i++) {
        printf("C[%d] = %.1f + %.1f = %.1f\n", i, A[i], B[i], C[i]);
    }

    // Liberación de memoria
    free(A);
    free(B);
    free(C);

    return 0;
}
