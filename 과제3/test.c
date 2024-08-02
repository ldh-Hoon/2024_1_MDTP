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

#define HIDDEN_LAYER_NUM 3 // 로드 할 신경망의 hidden layer 개수

#define MODEL_BASE_PATH "../Project_Train" // 모델 저장 폴더(파일명은 히든레이어개수.bin)

// dataset 관련
#define TEST_DATA_PATH "../testing"

/* 
숫자 0 : 980개
숫자 1 : 1135개
숫자 2 : 1032개
숫자 3 : 1010개
숫자 4 : 982개
숫자 5 : 892개
숫자 6 : 958개
숫자 7 : 1028개
숫자 8 : 974개
숫자 9 : 1009개
총 개수 10000개
*/
#define TEST_DATA_SIZE 10000

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
    double* batch_weights;
    double* biases;
    double* batch_biases;
    double* g;
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

// 신경망 초기화 함수들

void initialize_layer(Layer* layer, int input_nodes, int output_nodes, int id, void (*activation_function)(double*, double*, int), void (*activation_function_d)(double*, double*, int));
void initialize_network(Network* network, int layer_num, int* num,
    void (*activation_function)(double*, double*, int), void (*activation_function_d)(double*, double*, int));

// 신경망 로드

void load_model(Network* network, int num);

// 순전파 함수들

void forward_layer(Layer* layer, double* input, bool is_eval);
void forward(Network* network, double* input);

// 검증 함수
void eval(Network* network, double** features, double** targets, int size);

// 데이터셋 관련

int data_count(const char* baseFolder);
void prepare_data(const char* data_path,
    int data_size, double*** features, double*** targets);
void load_data(const char* baseFolder, double** features, double** targets, int max);
void load_image(const char* imagePath, double* features);

// 데이터 전처리

void one_hot_encode(int label, double* output);

// 메모리 해제

void free_layer(Layer* layer);
void free_network(Network* network);

void memory_clear(
    double** features, double** targets,
    int size);

int main() {
    // 데이터 준비
    int test_data_size = data_count(TEST_DATA_PATH);

    double** test_feature = NULL;
    double** test_targets = NULL;

    prepare_data(TEST_DATA_PATH, test_data_size, &test_feature, &test_targets);

    Network network;

    // 신경망 불러오기
    load_model(&network, HIDDEN_LAYER_NUM);

    //검증
    eval(&network, test_feature, test_targets, TEST_DATA_SIZE);

    free_network(&network);
    
    memory_clear(test_feature, test_targets, TEST_DATA_SIZE);

    return 0;
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


/*각 레이어 초기화 함수*/
void initialize_layer(Layer* layer, int input_nodes, int output_nodes, int id,
    void (*activation_function)(double*, double*, int), void (*activation_function_d)(double*, double*, int)) {
    layer->id = id;
    layer->input_nodes = input_nodes;
    layer->output_nodes = output_nodes;

    layer->activation_function = activation_function;
    layer->activation_function_d = activation_function_d;

    layer->weights = (double*)malloc(input_nodes * output_nodes * sizeof(double));
    layer->batch_weights = (double*)malloc(input_nodes * output_nodes * sizeof(double));
    if (layer->weights == NULL) {
        printf("layer->weights 메모리 할당 실패\n");
        exit(1);
    }

    if (layer->batch_weights == NULL) {
        printf("layer->batch_weights 메모리 할당 실패\n");
        exit(1);
    }
    for (int i = 0; i < input_nodes; i++) {
        for (int j = 0; j < output_nodes; j++) {
            *(layer->weights + i * output_nodes + j) = 0;
            *(layer->batch_weights + i * output_nodes + j) = 0.0;
        }
    }

    layer->g = (double*)malloc(input_nodes * sizeof(double));
    layer->inputs = (double*)malloc(input_nodes * sizeof(double));
    layer->biases = (double*)malloc(output_nodes * sizeof(double));
    layer->batch_biases = (double*)malloc(output_nodes * sizeof(double));
    layer->outputs = (double*)malloc(output_nodes * sizeof(double));
    if (layer->g == NULL) {
        printf("layer->g 메모리 할당 실패\n");
        exit(1);
    }
    if (layer->inputs == NULL) {
        printf("layer->inputs 메모리 할당 실패\n");
        exit(1);
    }
    if (layer->biases == NULL) {
        printf("layer->biases 메모리 할당 실패\n");
        exit(1);
    }
    if (layer->batch_biases == NULL) {
        printf("layer->batch_biases 메모리 할당 실패\n");
        exit(1);
    }
    if (layer->outputs == NULL) {
        printf("layer->outputs 메모리 할당 실패\n");
        exit(1);
    }
    for (int i = 0; i < input_nodes; i++) {
        *(layer->g + i) = 0.0;
        *(layer->inputs + i) = 0.0;
    }
    for (int i = 0; i < output_nodes; i++) {
        *(layer->biases + i) = 0;
        *(layer->outputs + i) = 0.0;
        *(layer->batch_biases + i) = 0.0;
    }
}

/*신경망 초기화 함수*/
void initialize_network(Network* network, int layer_num, int* num,
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
        initialize_layer((network->layers + j), *(num + j), *(num + j + 1), j, activation_function, activation_function_d);
    }
    printf("\n");
    network->eval = false;
}

/*신경망 로드*/
void load_model(Network* network, int num) {
    char load_path[1024];
    sprintf(load_path, "%s/%d.bin", MODEL_BASE_PATH, num);
    printf("______model load______\n");
    printf("%s 를 불러옵니다.\n", load_path);
    FILE* file = fopen(load_path, "rb");
    if (file == NULL) {
        fprintf(stderr, "%s 파일을 열 수 없습니다.\n", load_path);
        exit(1);
    }
    int layer_num = 0;
    int* layer_data = NULL;
    fread(&layer_num, sizeof(int), 1, file);
    printf("히든 레이어 : %d개\n", layer_num - 1);
    layer_data = malloc(sizeof(int) * (layer_num + 1));
    fread(layer_data, sizeof(int), layer_num + 1, file);
    for (int i = 1; i < layer_num; i++) {
        printf("%dth layer : %d개\n", i, *(layer_data + i));
    }
    printf("\n");

    initialize_network(network, layer_num, layer_data, sigmoid, sigmoid_d);

    for (int i = 0; i < network->layer_num; i++) {
        fread(((*(network->layers + i)).weights), sizeof(double),
            (*(network->layers + i)).input_nodes * (*(network->layers + i)).output_nodes, file);

        fread(((*(network->layers + i)).biases), sizeof(double),
            (*(network->layers + i)).output_nodes, file);
    }
    fclose(file);
    printf("로드 완료!\n\n");
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

    // weight와 input을 곱하고, bias를 더한 후, 활성화 함수 적용
    for (int j = 0; j < layer->output_nodes; j++) {
        *(layer->outputs + j) = 0.0;
        for (int i = 0; i < layer->input_nodes; i++) {
            *(layer->outputs + j) += *(layer->weights + i * layer->output_nodes + j) * *(layer->inputs + i);
        }
        *(layer->outputs + j) += *(layer->biases + j);
        if (isnan(*(layer->outputs + j))) {
            *(layer->outputs + j) = DBL_MAX;
        }
    }

    layer->activation_function(layer->outputs, layer->outputs, layer->output_nodes);
}

/*성능 검증 함수*/
void eval(Network* network, double** features, double** targets, int size) {
    network->eval = true;
    printf("\n______eval_____\n");

    double accuracy = 0.0;
    int count[OUTPUT_SIZE] = { 0, };
    double accuracy_arr[OUTPUT_SIZE] = { 0,  };

    double* last_output = (*(network->layers + network->layer_num - 1)).outputs;

    for (int i = 0; i < size; i++) {
        forward(network, *(features + i)); // 순전파

        int true_label = 0, predicted_label = 0;
        double max_true_val = **(targets + i), max_pred_val = *(last_output + 0);

        // 실제 label과 예측 label
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (*(*(targets + i) + j) > max_true_val) {
                max_true_val = *(*(targets + i) + j);
                true_label = j;
            }
            if (*(last_output + j) > max_pred_val) {
                max_pred_val = *(last_output + j);
                predicted_label = j;
            }
        }

        count[true_label] += 1;
        if (predicted_label == true_label) {
            accuracy_arr[predicted_label] += 1.0;
            accuracy += 1;
        }
    }
    printf("Total Accuracy: %.2f%\n\n", accuracy/TEST_DATA_SIZE * 100);
    accuracy = 0.0;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        printf("acc of %d : %.2f%\n", i, accuracy_arr[i]/count[i] * 100);
        accuracy += accuracy_arr[i] / count[i];
    }
    accuracy /= OUTPUT_SIZE;

    // 결과 출력
    printf("Mean Accuracy: %.2f%\n\n", accuracy * 100);
    network->eval = false;
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

/*이미지를 읽어오는 함수*/
void load_image(const char* imagePath, double* features) {
    FILE* file = fopen(imagePath, "rb");
    if (file == NULL) {
        printf("%s 파일 없음!", imagePath);
        return;
    }
    unsigned char image[INPUT_SIZE];

    fread(image, sizeof(unsigned char), INPUT_SIZE, file);
    fclose(file);

    for (int i = 0; i < INPUT_SIZE; i++) {
        *(features + i) = (double)(*(image + i)) / 255.0;
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

                    load_image(filePath, *(features + index));
                    one_hot_encode(folder, *(targets + index));
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
    printf("총 개수 %d개\n", index);
}

/*dataset 준비 및 전처리 함수*/
void prepare_data(const char* data_path,
    int data_size, double*** features, double*** targets) {

    time_t start = time(NULL);
    printf("______data load_____\n");

    *features = (double**)malloc(sizeof(double*) * data_size);
    *targets = (double**)malloc(sizeof(double*) * data_size);
    if (*features == NULL || *targets == NULL) {
        printf("dataset 메모리 할당 실패");
        exit(1);
        return;
    }
    for (int i = 0; i < data_size; i++) {
        *(*features + i) = (double*)malloc(sizeof(double) * INPUT_SIZE);
        *(*targets + i) = (double*)malloc(sizeof(double) * OUTPUT_SIZE);
    }

    load_data(data_path, *features, *targets, data_size);

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