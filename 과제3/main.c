#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <windows.h>
#include <io.h>

// model 관련
#define INPUT_SIZE 784 // train data 입력 크기
#define OUTPUT_SIZE 10 // train data 출력 크기

#define HIDDEN_LAYER_NUM 3
#define LAYER_DATA 256, 128, 32 //히든 레이어 노드 개수

#define LAYER_NUM HIDDEN_LAYER_NUM+1 //출력층 포함 레이어 개수

// dataset 관련
#define TRAIN_DATA_PATH "../training"

#define DATA_ADD 1 // 1이면 데이터 증강 6만개(랜덤으로 +- 1 상하이동, +- 10도 회전)
#define TRAIN_DATA_SIZE (60000 + DATA_ADD*60000) // 기본 6만, 데이터 증강 시 12만

// train 관련 
#define EPOCH 30 // 전체 데이터 반복 횟수
#define BATCH_SIZE 16 // batch size
#define LR 0.04 //learning rate
double linear_lr = 0.01;

#define DROPOUT_RATE 0.1 //dropout 

#define MOMENTUM 0.9 //update 가속도
#define NAG 1 //1이면 NAG, 0이면 모멘텀

#define RELU 0 // 1이면 ReLU 사용
#define SOFTMAX RELU //출력층 softmax 적용 여부

// 기타 
#define PI 3.141592
#define Sigmoid(x) 1.0 / (1.0 + exp(-x))

// 레이어 구조체
typedef struct {
    int id;
    int input_nodes;
    int output_nodes;
    double* inputs;
    double* outputs;
    double* weights;
    double* delta_weights;
    double* batch_weights;
    double* biases;
    double* delta_biases;
    double* batch_biases;
    double* g;
    double dropout_rate;
    void (*activation_function)(double*, double*, int);
    void (*activation_function_d)(double*, double*, int);
} Layer;

//신경망 구조체
typedef struct {
    Layer* layers;
    int layer_num;
    bool eval;
} Network;

void sigmoid(double* input, double* output, int length);
void sigmoid_d(double* input, double* output, int length);

void ReLU(double* input, double* output, int length);
void ReLU_d(double* input, double* output, int length);

double cross_entropy_loss(double* y_true, double* y_pred);
void cross_entropy_loss_d(double* y_true, double* y_pred, double* output);

double mse(double* y_true, double* y_pred);
void mse_d(double* y_true, double* y_pred, double* output);

void Softmax(double* input, double* output, int length);
void Softmax_d(double* input, double* output, int length);

// 정규분포 난수 생성 함수

double normal_distribution(double mean, double std);

// 신경망 초기화 함수들

void initialize_layer(Layer* layer, int input_nodes, int output_nodes, int id, double dropout_rate, void (*activation_function)(double*, double*, int), void (*activation_function_d)(double*, double*, int));
void initialize_network(Network* network, int layer_num, int* num, double dropout_rate,
    void (*activation_function)(double*, double*, int), void (*activation_function_d)(double*, double*, int));

// 순전파 함수들

void forward_layer(Layer* layer, double* input, bool is_eval);
void forward(Network* network, double* input);

// 역전파 함수들

void backward_layer(Layer* layer, double* next_g);
void backward(Network* network, double* next_g);


// 데이터셋 관련

int data_count(const char* baseFolder);
void prepare_data(const char* data_path,
    int data_size, double*** features, double*** targets);
void load_data(const char* baseFolder, double** features, double** targets, int max);
void load_image(const char* imagePath, double* features, double* features2);
void data_augmentation(unsigned char* input, unsigned char* output);

// 데이터 전처리

void mix(double** features, double** target, int size);

void one_hot_encode(int label, double* output);

// 학습

void train(Network* network, double** train_features, double** train_targets);

void save_model(Network* network);

// 메모리 해제

void free_layer(Layer* layer);
void free_network(Network* network);

void memory_clear(
    double** features, double** targets,
    int size);

int main() {
    // 학습 전

    // 신경망 초기화
    Network network;
    int layer_data[LAYER_NUM + 1] = { INPUT_SIZE, LAYER_DATA, OUTPUT_SIZE }; //입력층, 출력층 포함 노드 개수
    //초기화 및 dropout, 활성함수 설정
    if (RELU == 0) {
        initialize_network(&network, LAYER_NUM, layer_data, DROPOUT_RATE, sigmoid, sigmoid_d);
    }
    else if (RELU == 1) {
        initialize_network(&network, LAYER_NUM, layer_data, DROPOUT_RATE, ReLU, ReLU_d);
    }

    // 데이터 준비
    int train_data_size = data_count(TRAIN_DATA_PATH);

    double** train_features = NULL;
    double** train_targets = NULL;

    prepare_data(TRAIN_DATA_PATH, train_data_size, &train_features, &train_targets);

    //학습
    train(&network, train_features, train_targets);

    save_model(&network);

    memory_clear(train_features, train_targets, TRAIN_DATA_SIZE);

    free_network(&network);

    return 0;
}

void Softmax(double* input, double* output, int length) {
    double max = *input;
    double sum = 0.0;

    for (int i = 1; i < length; i++) {
        if (*(input + i) > max) {
            max = *(input + i);
        }
    }

    for (int i = 0; i < length; i++) {
        *(output + i) = exp(*(input + i) - max);//오버플로우 방지
        sum += *(output + i);
    }

    for (int i = 0; i < length; i++) {
        *(output + i) /= sum;
    }
}

void Softmax_d(double* input, double* output, int length) {
    for (int i = 0; i < length; i++) {
        for (int j = 0; j < length; j++) {
            if (i == j)
                *(output + i * length + j) = *(input + i) * (1 - *(input + j));
            else
                *(output + i * length + j) = -*(input + i) * *(input + j);
        }
    }
}

void sigmoid(double* input, double* output, int length) {
    for (int i = 0; i < length; i++) {
        *(output + i) = Sigmoid(*(input + i));
    }
}

void sigmoid_d(double* input, double* output, int length) {
    for (int i = 0; i < length; i++) {
        *(output + i) = *(input + i) * (1 - *(input + i));
    }
}

void ReLU(double* input, double* output, int length) {
    for (int i = 0; i < length; i++) {
        *(output + i) = *(input + i) > 0 ? *(input + i) : 0.0;
    }
}

void ReLU_d(double* input, double* output, int length) {
    for (int i = 0; i < length; i++) {
        *(output + i) = *(input + i) > 0 ? 1.0 : 0.0;
    }
}


double cross_entropy_loss(double* y_true, double* y_pred) {
    double sum = 0.0;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        sum += *(y_true + i) * log(*(y_pred + i) + 1e-15);
    }
    return -sum / OUTPUT_SIZE;
}

void cross_entropy_loss_d(double* y_true, double* y_pred, double* output) {
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        *(output + i) = *(y_pred + i) - *(y_true + i);
    }
}

double mse(double* y_true, double* y_pred) {
    double sum = 0.0;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        double diff = *(y_true + i) - *(y_pred + i);
        sum += 0.5 * diff * diff;
    }
    return sum / OUTPUT_SIZE;
}

void mse_d(double* y_true, double* y_pred, double* output) {
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        *(output + i) = *(y_pred + i) - *(y_true + i);
    }
}

/*정규분포 난수 생성 함수*/
double normal_distribution(double mean, double std) {
    double U1 = (double)rand() / RAND_MAX;
    double U2 = (double)rand() / RAND_MAX;
    return mean + std * sqrt(-2 * log(U1)) * cos(2 * PI * U2);
}

/*각 레이어 초기화 함수*/
void initialize_layer(Layer* layer, int input_nodes, int output_nodes, int id, double dropout_rate,
    void (*activation_function)(double*, double*, int), void (*activation_function_d)(double*, double*, int)) {
    layer->id = id;
    layer->input_nodes = input_nodes;
    layer->output_nodes = output_nodes;
    layer->dropout_rate = dropout_rate;

    layer->activation_function = activation_function;
    layer->activation_function_d = activation_function_d;

    layer->weights = (double*)malloc(input_nodes * output_nodes * sizeof(double));
    layer->delta_weights = (double*)malloc(input_nodes * output_nodes * sizeof(double));
    layer->batch_weights = (double*)malloc(input_nodes * output_nodes * sizeof(double));
    if (layer->weights == NULL) {
        fprintf(stderr, "layer->weights 메모리 할당 실패\n");
        exit(1);
    }
    if (layer->delta_weights == NULL) {
        fprintf(stderr, "layer->delta_weights 메모리 할당 실패\n");
        exit(1);
    }
    if (layer->batch_weights == NULL) {
        fprintf(stderr, "layer->batch_weights 메모리 할당 실패\n");
        exit(1);
    }
    double a = sqrt(2.0 / (layer->input_nodes + layer->output_nodes)); //Xavier 초기화
    for (int i = 0; i < input_nodes; i++) {
        for (int j = 0; j < output_nodes; j++) {
            *(layer->weights + i * output_nodes + j) = normal_distribution(0, a);  // 표준편차가 a인 정규분포 값
            if (*(layer->weights + i * output_nodes + j) < -99) {
                *(layer->weights + i * output_nodes + j) = 0;
            }
            if (*(layer->weights + i * output_nodes + j) > 99) {
                *(layer->weights + i * output_nodes + j) = 0;
            }
            // 정규분포 이상치 처리(log의 음의 무한대 값)
            *(layer->delta_weights + i * output_nodes + j) = 0.0;
            *(layer->batch_weights + i * output_nodes + j) = 0.0;
        }
    }

    layer->g = (double*)malloc(input_nodes * sizeof(double));
    layer->inputs = (double*)malloc(input_nodes * sizeof(double));
    layer->biases = (double*)malloc(output_nodes * sizeof(double));
    layer->batch_biases = (double*)malloc(output_nodes * sizeof(double));
    layer->delta_biases = (double*)malloc(output_nodes * sizeof(double));
    layer->outputs = (double*)malloc(output_nodes * sizeof(double));
    if (layer->g == NULL) {
        fprintf(stderr, "layer->g 메모리 할당 실패\n");
        exit(1);
    }
    if (layer->inputs == NULL) {
        fprintf(stderr, "layer->inputs 메모리 할당 실패\n");
        exit(1);
    }
    if (layer->biases == NULL) {
        fprintf(stderr, "layer->biases 메모리 할당 실패\n");
        exit(1);
    }
    if (layer->batch_biases == NULL) {
        fprintf(stderr, "layer->batch_biases 메모리 할당 실패\n");
        exit(1);
    }
    if (layer->delta_biases == NULL) {
        fprintf(stderr, "layer->delta_biases 메모리 할당 실패\n");
        exit(1);
    }
    if (layer->outputs == NULL) {
        fprintf(stderr, "layer->outputs 메모리 할당 실패\n");
        exit(1);
    }
    for (int i = 0; i < input_nodes; i++) {
        *(layer->g + i) = 0.0;
        *(layer->inputs + i) = 0.0;
    }
    for (int i = 0; i < output_nodes; i++) {
        *(layer->biases + i) = normal_distribution(0, a);  // 표준편차가 a인 정규분포 값
        if (*(layer->biases + i) < -99) {
            *(layer->biases + i) = 0;
        }
        if (*(layer->biases + i) > 99) {
            *(layer->biases + i) = 0;
        }
        // 정규분포 이상치 처리
        *(layer->outputs + i) = 0.0;
        *(layer->batch_biases + i) = 0.0;
        *(layer->delta_biases + i) = 0.0;
    }
}
/*신경망 초기화 함수*/
void initialize_network(Network* network, int layer_num, int* num, double dropout_rate,
    void (*activation_function)(double*, double*, int), void (*activation_function_d)(double*, double*, int)) {
    printf("_____init_____\n");
    printf("hidden layer 개수 : %d\n", layer_num - 1);
    network->layer_num = layer_num;
    network->layers = (Layer*)malloc(sizeof(Layer) * layer_num);
    if (network->layers == NULL) {
        printf("network->layers 메모리 할당 실패\n");
        exit(1);
    }
    printf("%d", *(num + 0));
    for (int j = 0; j < layer_num; j++) {
        printf(">%d", *(num + j + 1));
        initialize_layer((network->layers + j), *(num + j), *(num + j + 1), j, dropout_rate, activation_function, activation_function_d);
    }
    printf("\n\n");
    network->eval = false;
}

/*순전파 함수*/
void forward(Network* network, double* input) {
    double* current_input = input;
    // 각 레이어를 순회하며 forward_layer 호출
    for (int i = 0; i < network->layer_num; i++) {
        forward_layer((network->layers + i), current_input, network->eval);
        current_input = (*(network->layers + i)).outputs;
    }
}

/*각 레이어의 순전파를 계산하는 함수*/
void forward_layer(Layer* layer, double* input, bool is_eval) {
    for (int i = 0; i < layer->input_nodes; i++) {
        *(layer->inputs + i) = *(input + i);
    }

    if ((!is_eval) && (layer->dropout_rate > 0.0)) {
        for (int i = 0; i < layer->input_nodes; i++) {
            if ((double)rand() / RAND_MAX < layer->dropout_rate) {
                *(layer->inputs + i) = 0.0; // 드롭아웃 비율만큼 노드를 비활성화
            }
        }
    }

    // weight와 input을 곱하고, bias를 더한 후, 활성화 함수 적용
    for (int j = 0; j < layer->output_nodes; j++) {
        *(layer->outputs + j) = 0.0;
        for (int i = 0; i < layer->input_nodes; i++) {
            *(layer->outputs + j) += (*(layer->weights + i * layer->output_nodes + j)
                + NAG * MOMENTUM * *(layer->delta_weights + i * layer->output_nodes + j)) * *(layer->inputs + i);
        }
        *(layer->outputs + j) += *(layer->biases + j) + NAG * MOMENTUM * *(layer->delta_biases + j);
        if (isnan(*(layer->outputs + j))) {
            *(layer->outputs + j) = DBL_MAX;
        }
    }

    layer->activation_function(layer->outputs, layer->outputs, layer->output_nodes);

    if (SOFTMAX == 1 && layer->id == HIDDEN_LAYER_NUM) {
        Softmax(layer->outputs, layer->outputs, layer->output_nodes);
    }
}

/*각 레이어의 역전파를 진행하는 함수*/
void update_layer(Layer* layer) {
    int i, j;
    // 배치 가중치 업데이트
    for (i = 0; i < layer->input_nodes; i++) {
        for (j = 0; j < layer->output_nodes; j++) {
            *(layer->batch_weights + i * layer->output_nodes + j) = *(layer->batch_weights + i * layer->output_nodes + j) / BATCH_SIZE;
            *(layer->delta_weights + i * layer->output_nodes + j) = MOMENTUM * *(layer->delta_weights + i * layer->output_nodes + j)
                + *(layer->batch_weights + i * layer->output_nodes + j);
            *(layer->weights + i * layer->output_nodes + j) += *(layer->delta_weights + i * layer->output_nodes + j);
            *(layer->batch_weights + i * layer->output_nodes + j) = 0.0;
        }
    }

    // 바이어스 업데이트
    for (j = 0; j < layer->output_nodes; j++) {
        *(layer->batch_biases + j) = *(layer->batch_biases + j) / BATCH_SIZE;
        *(layer->delta_biases + j) = MOMENTUM * *(layer->delta_biases + j) + *(layer->batch_biases + j);
        *(layer->biases + j) += *(layer->delta_biases + j);
        *(layer->batch_biases + j) = 0.0;
    }
}
void backward_update(Network* network) {
    for (int i = network->layer_num - 1; i >= 0; i--) {
        update_layer(network->layers + i);
    }
}

void backward_layer(Layer* layer, double* next_g) {
    int i, j;
    // 업데이트 계산
    for (i = 0; i < layer->input_nodes; i++) {
        for (j = 0; j < layer->output_nodes; j++) {
            *(layer->batch_weights + i * layer->output_nodes + j) -= (linear_lr * LR) * (*(next_g + j) * *(layer->inputs + i));
        }
    }
    // 바이어스 업데이트 계산
    for (j = 0; j < layer->output_nodes; j++) {
        *(layer->batch_biases + j) -= (linear_lr * LR) * (*(next_g + j));
    }
    layer->activation_function_d(layer->inputs, layer->inputs, layer->input_nodes);

    // 그래디언트 계산 후 이전 레이어로 전달
    for (i = 0; i < layer->input_nodes; i++) {
        *(layer->g + i) = 0.0;
        for (j = 0; j < layer->output_nodes; j++) {
            *(layer->g + i) += (*(layer->weights + i * layer->output_nodes + j) + NAG * (MOMENTUM * *(layer->delta_weights + i * layer->output_nodes + j))) * *(next_g + j);
        }
        *(layer->g + i) *= *(layer->inputs + i);
    }
}

/*역전파 함수*/
void backward(Network* network, double* next_g) {
    double* layer_d_output = next_g;

    for (int i = network->layer_num - 1; i >= 0; i--) {
        backward_layer(network->layers + i, layer_d_output);

        layer_d_output = (*(network->layers + i)).g;
    }
}

/*dataset을 섞는 함수*/
void mix(double** features, double** target, int size) {
    srand(time(NULL));
    double temp;
    int rn;
    for (int i = 0; i < size - 1; i++) {
        rn = rand() % (size - i) + i;
        for (int k = 0; k < INPUT_SIZE; k++) {
            temp = *(*(features + i) + k);
            *(*(features + i) + k) = *(*(features + rn) + k);
            *(*(features + rn) + k) = temp;
        }
        for (int k = 0; k < OUTPUT_SIZE; k++) {
            temp = *(*(target + i) + k);
            *(*(target + i) + k) = *(*(target + rn) + k);
            *(*(target + rn) + k) = temp;
        }
    }
}

/*신경망 저장 함수, 히든레이어개수.bin 파일*/
void save_model(Network* network) {
    char save_path[1024];

    sprintf(save_path, "./%d.bin", HIDDEN_LAYER_NUM);
    FILE* file = fopen(save_path, "wb");

    int relu = RELU;
    fwrite(&(network->layer_num), sizeof(int), 1, file);
    fwrite(&relu, sizeof(int), 1, file);
    fwrite(&((*(network->layers)).input_nodes), sizeof(int), 1, file);
    for (int i = 0; i < network->layer_num; i++) {
        fwrite(&((*(network->layers + i)).output_nodes), sizeof(int), 1, file);
    }
    for (int i = 0; i < network->layer_num; i++) {
        fwrite(((*(network->layers + i)).weights), sizeof(double),
            (*(network->layers + i)).input_nodes * (*(network->layers + i)).output_nodes, file);

        fwrite(((*(network->layers + i)).biases), sizeof(double),
            (*(network->layers + i)).output_nodes, file);
    }
    fclose(file);
    printf("저장 완료\n");
}

/*조건대로 학습을 진행하는 함수*/
void train(Network* network, double** train_features, double** train_targets) {
    time_t start = time(NULL);
    printf("\n______train_____\n");
    for (int epoch = 0; epoch < EPOCH; epoch++) {
        double total_loss = 0.0;
        printf("Epoch %d start\n", epoch + 1);
        for (int i = 0; i < TRAIN_DATA_SIZE; i += BATCH_SIZE) {
            double batch_loss = 0.0;

            for (int j = i; j < i + BATCH_SIZE; j++) {
                forward(network, *(train_features + j));

                double loss = mse(*(train_targets + j), (*(network->layers + network->layer_num - 1)).outputs);
                batch_loss += loss;

                mse_d(*(train_targets + j), (*(network->layers + network->layer_num - 1)).outputs, (*(network->layers + network->layer_num - 1)).outputs);

                backward(network, (*(network->layers + network->layer_num - 1)).outputs);
            }
            backward_update(network);
            batch_loss /= BATCH_SIZE;
            total_loss += batch_loss;
        }
        total_loss /= TRAIN_DATA_SIZE / (double)(BATCH_SIZE);
        printf("Epoch %d, Total train loss: %f\n", epoch + 1, total_loss);
        linear_lr = (1.0 + linear_lr) / 2;
    }
    time_t end = time(NULL);
    printf("학습 소요시간: %lf\n", (double)(end - start));
}

/*one-hot encoding*/
void one_hot_encode(int label, double* output) {
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        *(output + i) = (i == label) ? 1.0 : 0.0;
    }
}


/*데이터 폴더의 이미지 개수를 세는 함수*/
int data_count(const char* baseFolder) {
    int count = 0;
    struct _finddata_t fileinfo;
    intptr_t hFile;
    char path[1024];

    for (int folder = 0; folder <= 9; folder++) {
        sprintf(path, "%s\\%d\\*.raw", baseFolder, folder);

        if ((hFile = _findfirst(path, &fileinfo)) == -1L) {
            printf("폴더 접근 불가: '%s'\n", path);
            continue;
        }
        else {
            do {
                if (!(fileinfo.attrib & _A_SUBDIR)) {
                    count++;
                }
            } while (_findnext(hFile, &fileinfo) == 0);
            _findclose(hFile);
        }
    }
    return count;
}

/*데이터 증강 함수*/
void data_augmentation(unsigned char* input, unsigned char* output) {
    int px = rand() % 3 - 1;
    int py = rand() % 3 - 1;
    int degree = rand() % 21 - 10;

    float radians = degree * PI / 180.0;
    float cos_theta = cos(radians);
    float sin_theta = sin(radians);

    int WIDTH = 28;
    int HEIGHT = 28;
    int centerX = WIDTH / 2;
    int centerY = HEIGHT / 2;

    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            // 역방향 사상
            int x = (int)(cos_theta * (j - centerX) + sin_theta * (i - centerY)) + centerX + px;
            int y = (int)(-sin_theta * (j - centerX) + cos_theta * (i - centerY)) + centerY + py;

            if (x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT) {
                *(output + i * WIDTH + j) = *(input + y * WIDTH + x);
            }
            else {
                *(output + i * WIDTH + j) = 0; // 범위를 벗어나면 검은색으로
            }
        }
    }
}

/*이미지를 읽어오는 함수*/
void load_image(const char* imagePath, double* features, double* features2) {
    FILE* file = fopen(imagePath, "rb");
    if (file == NULL) {
        printf("%s 파일 없음!", imagePath);
        return;
    }
    unsigned char image[INPUT_SIZE];
    unsigned char aug_image[INPUT_SIZE];

    fread(image, sizeof(unsigned char), INPUT_SIZE, file);
    fclose(file);
    if (DATA_ADD == 1) {
        data_augmentation(image, aug_image);
    }

    for (int i = 0; i < INPUT_SIZE; i++) {
        *(features + i) = (double)(*(image + i)) / 255.0;
        if (DATA_ADD == 1) {
            *(features2 + i) = (double)(*(aug_image + i)) / 255.0;
        }
    }
}

/*데이터셋을 로드하는 함수*/
void load_data(const char* baseFolder, double** features, double** targets, int max) {
    struct _finddata_t fileinfo;
    intptr_t hFile;
    char path[1024];
    int index = 0;
    printf("이미지 로드 중...\n");
    for (int folder = 0; folder <= 9; folder++) {
        int count = 0;
        sprintf(path, "%s\\%d\\*.raw", baseFolder, folder);
        // path 위치의 모든 raw 확장자 파일 목록을 반환
        if ((hFile = _findfirst(path, &fileinfo)) == -1L) {
            printf("폴더 접근 불가: '%s'\n", path);
            continue;
        }
        else {
            do {
                if (!(fileinfo.attrib & _A_SUBDIR)) {
                    char filePath[1024];
                    sprintf(filePath, "%s\\%d\\%s", baseFolder, folder, fileinfo.name);

                    if (DATA_ADD == 1) {
                        load_image(filePath, *(features + index), *(features + index + max));
                        one_hot_encode(folder, *(targets + index));
                        one_hot_encode(folder, *(targets + index + max));
                    }
                    else {
                        load_image(filePath, *(features + index), *(features + index));
                        one_hot_encode(folder, *(targets + index));
                    }
                    index++;
                    if (index > max) {
                        return;
                    }
                    count += 1;
                }
            } while (_findnext(hFile, &fileinfo) == 0);
            _findclose(hFile);
            printf("숫자 %d : %d개\n", folder, count);
        }
    }
    if (DATA_ADD) {
        printf("총 개수 %d개, 데이터 증강 %d개\n", index, index);
    }
    else {
        printf("총 개수 %d개\n", index);
    }
}

/*dataset 준비 및 전처리 함수*/
void prepare_data(const char* data_path,
    int data_size, double*** features, double*** targets) {

    time_t start = time(NULL);
    printf("______data load/augmentation_____\n");

    *features = (double**)malloc(sizeof(double*) * TRAIN_DATA_SIZE);
    *targets = (double**)malloc(sizeof(double*) * TRAIN_DATA_SIZE);
    if (*features == NULL || *targets == NULL) {
        printf("dataset 메모리 할당 실패");
        exit(1);
        return;
    }
    for (int i = 0; i < TRAIN_DATA_SIZE; i++) {
        *(*features + i) = (double*)malloc(sizeof(double) * INPUT_SIZE);
        *(*targets + i) = (double*)malloc(sizeof(double) * OUTPUT_SIZE);
    }
    load_data(data_path, *features, *targets, data_size);
    for (int i = 0; i < 5; i++) {
        mix(*features, *targets, TRAIN_DATA_SIZE);
    }

    time_t end = time(NULL);
    printf("데이터셋 로드 및 전처리 소요시간: %lf\n\n", (double)(end - start));
}

/*dataset 관련 메모리 해제*/
void memory_clear(
    double** features, double** targets,
    int size)
{
    for (int i = 0; i < size; i++) {
        free(*(features + i));
        free(*(targets + i));
    }
    free(features);
    free(targets);
}


/*각 레이어 메모리를 해제하는 함수*/
void free_layer(Layer* layer) {
    free(layer->weights);
    free(layer->batch_weights);
    free(layer->g);
    free(layer->inputs);
    free(layer->biases);
    free(layer->outputs);
    layer->activation_function = NULL;
    layer->activation_function_d = NULL;
}

/*신경망 메모리를 해제하는 함수*/
void free_network(Network* network) {
    for (int i = network->layer_num - 1; i >= 0; i--) {
        free_layer(network->layers + i);
    }
    free(network->layers);
}