#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define PI 3.141592
#define uc unsigned char

#define WIDTH 512
#define HEIGHT 512

#define PATH "../Raw_Image/lena.img"
#define OUTPUT_PATH "./output.img"

void rotate_degree(uc* input, uc* output, int degree);
void rotate_90(uc* input, uc* output);
void rotate_180(uc* input, uc* output);
void rotate_270(uc* input, uc* output);
void flip_v(uc* input, uc* output);
void flip_h(uc* input, uc* output);

void userInput(uc* input, uc* output); 

void load_image(uc* input); 
void save_image(uc* output); 

int main() {
    uc* img;
    uc* outputimg;
    img = (uc*)malloc(sizeof(uc) * WIDTH * HEIGHT);
    outputimg = (uc*)malloc(sizeof(uc) * WIDTH * HEIGHT);

    load_image(img);
    userInput(img, outputimg);
    save_image(outputimg);

    free(img);
    free(outputimg);
    return 0;
}

/** 사용자 입력을 처리하는 함수*/
void userInput(uc* input, uc* output) {
    int i;
    printf("__________________\n");
    printf("0 : 오른쪽으로 90도 회전\n");
    printf("1 : 오른쪽으로 180도 회전\n");
    printf("2 : 오른쪽으로 270도 회전\n");
    printf("3 : 오른쪽으로 임의 각도 회전\n");
    printf("4 : 좌우 반전\n");
    printf("5 : 상하 반전\n");
    printf("__________________\n");

    do
    {
        printf("입력(0~5) : ");
        scanf("%d", &i);
    } while (i < 0 || 5 < i);

    if (i == 0)
        rotate_90(input, output);
    else if (i == 1)
        rotate_180(input, output);
    else if (i == 2)
        rotate_270(input, output);
    else if (i == 3) {
        int degree = -1;
        for (;;) {
            printf("각도 입력(0~360) : ");
            scanf("%d", &degree);
            if (0 <= degree && degree <= 360) {
                break;
            }
        }
        rotate_degree(input, output, degree);
    }
    else if (i == 4)
        flip_v(input, output);
    else if (i == 5)
        flip_h(input, output);
}

/** 이미지 파일을 여는 함수*/
void load_image(uc* input) {
    FILE* inFile = fopen(PATH, "rb");
    //이미지 열기

    if (inFile == NULL) {
        printf("%s 파일을 열 수 없습니다.\n", PATH);
        exit(1);
    }

    fread(input, sizeof(uc), WIDTH * HEIGHT, inFile);
    printf("%s 파일을 열었습니다.\n", PATH);

    // 파일 닫기
    fclose(inFile);

}

/** 이미지를 저장하는 함수*/
void save_image(uc* output) {
    FILE* outFile = fopen(OUTPUT_PATH, "wb");
    //이미지 열기

    if (outFile == NULL) {
        printf("파일을 저장할 수 없습니다. %s\n", OUTPUT_PATH);
        exit(1);
    }

    fwrite(output, sizeof(uc), WIDTH * HEIGHT, outFile);
    printf("%s에 저장 완료.\n", OUTPUT_PATH);

    // 파일 닫기
    fclose(outFile);
}

/** 이미지를 90도 회전하는 함수*/
void rotate_90(uc* input, uc* output) {
    for (int i = 0; i < HEIGHT; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            *(output + j * HEIGHT + (HEIGHT - i - 1)) = *(input + i * WIDTH + j);
        }
    }
    printf("이미지 90도 회전 완료.\n");
}

/** 이미지를 180도 회전하는 함수*/
void rotate_180(uc* input, uc* output) {
    for (int i = 0; i < HEIGHT; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            *(output + (HEIGHT - i - 1) * WIDTH + (WIDTH - j - 1)) = *(input + i * WIDTH + j);
        }
    }
    printf("이미지 180도 회전 완료.\n");
}

/** 이미지를 270도 회전하는 함수*/
void rotate_270(uc* input, uc* output) {
    for (int i = 0; i < HEIGHT; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            *(output + (WIDTH - j - 1) * HEIGHT + i) = *(input + i * WIDTH + j);
        }
    }
    printf("이미지 270도 회전 완료.\n");
}

/** 이미지를 degree만큼 회전하는 함수*/
void rotate_degree(uc* input, uc* output, int degree) {
    float radians = degree * PI / 180.0;
    float cos_theta = cos(radians);
    float sin_theta = sin(radians);

    int centerX = WIDTH / 2;
    int centerY = HEIGHT / 2;

    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            // 역방향 사상
            int x = (int)(cos_theta * (j - centerX) + sin_theta * (i - centerY)) + centerX;
            int y = (int)(-sin_theta * (j - centerX) + cos_theta * (i - centerY)) + centerY;

            if (x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT) {
                *(output + i * WIDTH + j) = *(input + y * WIDTH + x);
            }
            else {
                *(output + i * WIDTH + j) = 0; // 범위를 벗어나면 검은색으로
            }
        }
    }
    printf("이미지 %d도 회전 완료.\n", degree);
}

/** 이미지를 좌우로 뒤집는 함수*/
void flip_v(uc* input, uc* output) {
    for (int i = 0; i < HEIGHT; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            *(output + i * WIDTH + (WIDTH - j - 1)) = *(input + i * WIDTH + j);
        }
    }
    printf("이미지 좌우반전 완료.\n");
}

/** 이미지를 상하로 뒤집는 함수*/
void flip_h(uc* input, uc* output) {
    for (int i = 0; i < HEIGHT; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            *(output + (HEIGHT - i - 1) * WIDTH + j) = *(input + i * WIDTH + j);
        }
    }
    printf("이미지 상하반전 완료.\n");
}
