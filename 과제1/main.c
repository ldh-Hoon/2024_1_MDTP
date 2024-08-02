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

/** ����� �Է��� ó���ϴ� �Լ�*/
void userInput(uc* input, uc* output) {
    int i;
    printf("__________________\n");
    printf("0 : ���������� 90�� ȸ��\n");
    printf("1 : ���������� 180�� ȸ��\n");
    printf("2 : ���������� 270�� ȸ��\n");
    printf("3 : ���������� ���� ���� ȸ��\n");
    printf("4 : �¿� ����\n");
    printf("5 : ���� ����\n");
    printf("__________________\n");

    do
    {
        printf("�Է�(0~5) : ");
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
            printf("���� �Է�(0~360) : ");
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

/** �̹��� ������ ���� �Լ�*/
void load_image(uc* input) {
    FILE* inFile = fopen(PATH, "rb");
    //�̹��� ����

    if (inFile == NULL) {
        printf("%s ������ �� �� �����ϴ�.\n", PATH);
        exit(1);
    }

    fread(input, sizeof(uc), WIDTH * HEIGHT, inFile);
    printf("%s ������ �������ϴ�.\n", PATH);

    // ���� �ݱ�
    fclose(inFile);

}

/** �̹����� �����ϴ� �Լ�*/
void save_image(uc* output) {
    FILE* outFile = fopen(OUTPUT_PATH, "wb");
    //�̹��� ����

    if (outFile == NULL) {
        printf("������ ������ �� �����ϴ�. %s\n", OUTPUT_PATH);
        exit(1);
    }

    fwrite(output, sizeof(uc), WIDTH * HEIGHT, outFile);
    printf("%s�� ���� �Ϸ�.\n", OUTPUT_PATH);

    // ���� �ݱ�
    fclose(outFile);
}

/** �̹����� 90�� ȸ���ϴ� �Լ�*/
void rotate_90(uc* input, uc* output) {
    for (int i = 0; i < HEIGHT; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            *(output + j * HEIGHT + (HEIGHT - i - 1)) = *(input + i * WIDTH + j);
        }
    }
    printf("�̹��� 90�� ȸ�� �Ϸ�.\n");
}

/** �̹����� 180�� ȸ���ϴ� �Լ�*/
void rotate_180(uc* input, uc* output) {
    for (int i = 0; i < HEIGHT; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            *(output + (HEIGHT - i - 1) * WIDTH + (WIDTH - j - 1)) = *(input + i * WIDTH + j);
        }
    }
    printf("�̹��� 180�� ȸ�� �Ϸ�.\n");
}

/** �̹����� 270�� ȸ���ϴ� �Լ�*/
void rotate_270(uc* input, uc* output) {
    for (int i = 0; i < HEIGHT; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            *(output + (WIDTH - j - 1) * HEIGHT + i) = *(input + i * WIDTH + j);
        }
    }
    printf("�̹��� 270�� ȸ�� �Ϸ�.\n");
}

/** �̹����� degree��ŭ ȸ���ϴ� �Լ�*/
void rotate_degree(uc* input, uc* output, int degree) {
    float radians = degree * PI / 180.0;
    float cos_theta = cos(radians);
    float sin_theta = sin(radians);

    int centerX = WIDTH / 2;
    int centerY = HEIGHT / 2;

    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            // ������ ���
            int x = (int)(cos_theta * (j - centerX) + sin_theta * (i - centerY)) + centerX;
            int y = (int)(-sin_theta * (j - centerX) + cos_theta * (i - centerY)) + centerY;

            if (x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT) {
                *(output + i * WIDTH + j) = *(input + y * WIDTH + x);
            }
            else {
                *(output + i * WIDTH + j) = 0; // ������ ����� ����������
            }
        }
    }
    printf("�̹��� %d�� ȸ�� �Ϸ�.\n", degree);
}

/** �̹����� �¿�� ������ �Լ�*/
void flip_v(uc* input, uc* output) {
    for (int i = 0; i < HEIGHT; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            *(output + i * WIDTH + (WIDTH - j - 1)) = *(input + i * WIDTH + j);
        }
    }
    printf("�̹��� �¿���� �Ϸ�.\n");
}

/** �̹����� ���Ϸ� ������ �Լ�*/
void flip_h(uc* input, uc* output) {
    for (int i = 0; i < HEIGHT; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            *(output + (HEIGHT - i - 1) * WIDTH + j) = *(input + i * WIDTH + j);
        }
    }
    printf("�̹��� ���Ϲ��� �Ϸ�.\n");
}
