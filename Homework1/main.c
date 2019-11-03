#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int c_conv(int in_channel, int o_channel, int k_size, int stride);

int main(void) {
    
    int o_channel;
    int i;
    int total_time;
    int in_channel = 3;
    int k_size = 3;
    int stride = 1;
    
    
    
    for (i =0; i<8; i++) {
        o_channel = (int) pow(2,i);
        total_time = c_conv(in_channel,o_channel, k_size, stride);

    }
    
    return 0;
}


int c_conv(int in_channel, int o_channel, int k_size, int stride){
    int i,j,k,l,m,n;
    double pr;
    int count;
    
    time_t start_t, end_t, total_t;
    start_t = clock();
    
    /* creat the random kernel*/
    double ***kernel;
    kernel = (double ***)malloc((k_size)*sizeof(double **));
    for (i =0;i<k_size; i++){
        kernel[i] = (double **)malloc((k_size)*sizeof(double *));
        for (j=0;j<k_size;j++){
            kernel[i][j] = (double *)malloc(sizeof(double *) * o_channel);
            for (k=0;k<o_channel; k++){
                kernel[i][j][k] = rand();
            }
        }
    }
    
    
    /* image array*/
    int row = 1080;//720;//
    int col = 1920;//1280;//
    int hgt = 3;
    
    double ***image;
    image = (double ***)malloc(row*sizeof(double **));
    
    for (i = 0 ;  i < row; i++) {
        image[i] = (double **)malloc(col*sizeof(double *));
        
        for (j = 0; j < col; j++) {
            image[i][j] = (double *)malloc(hgt*sizeof(double));
            
            for (k = 0; k < hgt; k++){
               image[i][j][k] = rand() % 255 + 0;
            }
        }
    }
    

    
    double ***output_image = (double ***)malloc(row*sizeof(double **));
    
    for (i = 0 ;  i < row-k_size+1; i++) {
        output_image[i] = (double **)malloc(col*sizeof(double *));

        for (j = 0; j < col-k_size+1; j++) {
            output_image[i][j] = (double *)malloc(o_channel*sizeof(double));
        }
    }
    
    
    /* put the convolution in here */
    
    count = 0;
    for (l=0; l<o_channel;l++){                  //through the layers of the kernel
        for (i=0;i<row-k_size+1;i=i+stride){     // through the rows of the picture
            for (j=0;j<col-k_size+1;j=j+stride){ // through the columns of the picture
                double temp_sum =0;
                for (k=0;k<hgt;k++){             // through the layers of the picture
                    for (m=0;m<k_size;m++){      //through rows of the kernel
                        for (n=0;n<k_size;n++){  // through columns of the kernel
                            pr = image[i+m][j+n][k]*kernel[m][n][l];
                            temp_sum += pr;
                        }
                    }
                }
                output_image[i/stride][j/stride][l] = temp_sum;
            }
        }
    }
    
    
    
    end_t = clock();
    total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
    
    
    
    for (i=0;i<row;i++){
        for(j=0;j<col;j++){
            free(image[i][j]);
        }
        free(image[i]);
    }
    free(image);
    
    
    for (i=0;i<k_size;i++){
        for(j=0;j<k_size;j++){
            free(kernel[i][j]);
        }
        free(kernel[i]);
    }
    free(kernel);
    
    for (i=0;i<row-k_size+1;i++){
        for (j=0; j<col-k_size+1;j++){
            free(output_image[i][j]);
        }
        free(output_image[i]);
    }
    free(output_image);
    
    
    
    
    return total_t;
}
