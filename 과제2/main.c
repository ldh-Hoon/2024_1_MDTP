#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <windows.h>
#include <io.h>

#define PI 3.141592

// model 관련
#define INPUT_SIZE 784 // train data 입력 크기
#define OUTPUT_SIZE 10 // train data 출력 크기

#define LAYER_NUM 3
int layer_data[LAYER_NUM + 1] = { 784, 128, 16, 10 };

// dataset 관련
#define TRAIN_DATA_PATH "../5주차 과제/training"

// 5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949
#define TRAIN_DATA_SIZE 60000

// train 관련 
#define EPOCH 3 // 전체 데이터 반복 횟수
#define BATCH_SIZE 16 // batch size
#define LR 0.02 //learning rate

#define WEIGHT_CLIP 5
#define BIAS_CLIP 2


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
    double dropout_rate;
    void (*activation_function)(double*, double*, int);
    void (*activation_function_d)(double*, double*, int);
} Layer;

//신경망 구조체
typedef struct {
    Layer* layers;
    bool eval;
} Network;

void save_model(const char* save_path, Network network) {
    FILE* file = fopen(save_path, "w"); // 파일을 쓰기 모드로 열기

    if (file == NULL) {
        printf("파일을 열 수 없습니다.\n");
        return;
    }

    // 네트워크의 레이어 개수 저장
    fprintf(file, "%d\n", LAYER_NUM);

    for (int i = 0; i < LAYER_NUM; i++) {
        Layer layer = network.layers[i];

        // 각 레이어의 노드 개수 정보 저장
        fprintf(file, "%d %d\n", layer.input_nodes, layer.output_nodes);

        // 가중치 저장
        for (int j = 0; j < layer.input_nodes; j++) {
            for (int k = 0; k < layer.output_nodes; k++) {
                fprintf(file, "%lf ", layer.weights[j * layer.output_nodes + k]);
            }
            fprintf(file, "\n");
        }

        // 바이어스 저장
        for (int j = 0; j < layer.output_nodes; j++) {
            fprintf(file, "%lf ", layer.biases[j]);
        }
        fprintf(file, "\n");
    }

    fclose(file); // 파일 닫기
    printf("모델 저장 완료: %s\n", save_path);
}

// 활성화 함수 및 미분 함수 등

void sigmoid(double* input, double* output, int length);
void sigmoid_d(double* input, double* output, int length);

double cross_entropy_loss(double* y_true, double* y_pred);
void cross_entropy_loss_d(double* y_true, double* y_pred, double* output);

double mse(double* y_true, double* y_pred);
void mse_d(double* y_true, double* y_pred, double* output);

// 정규분포 난수 생성 함수

double normal_distribution(double mean, double std);

// 신경망 초기화 함수들

void initialize_layer(Layer* layer, int input_nodes, int output_nodes, int id, double dropout_rate, void (*activation_function)(double*, double*, int), void (*activation_function_d)(double*, double*, int));
void initialize_network(Network* network, int* num, double dropout_rate,
    void (*activation_function)(double*, double*, int), void (*activation_function_d)(double*, double*, int));

// 순방향 전파 함수들

void forward_layer(Layer* layer, double* input, bool is_eval);
void forward(Network* network, double* input);

// 역전파 함수들

void backward_layer(Layer* layer, double* next_g);
void backward(Network* network, double* next_g);

// 가중치 및 레이어 정보 출력 확인

void print_weights(Network* network);

// 데이터셋 관련

int data_count(const char* baseFolder);
void prepare_data(const char* train_data_path,
    int train_data_size,
    double*** train_features, double*** train_targets);
void load_data(const char* baseFolder, double** features, double** targets, int max);
void load_image(const char* imagePath, double* features);

// 데이터 전처리

void mix(double** features, double** target, int size);

void one_hot_encode(int label, double* output);

// 학습

void train(Network* network, double** train_features, double** train_targets);

// 메모리 반납

void free_layer(Layer* layer);
void free_network(Network* network);

void memory_clear(
    double** train_features, double** train_targets,
    int size);

int main() {
    //srand(time(NULL));  // 난수 생성기 초기화

    // 학습 전

    // 신경망 초기화
    Network network;    

    //초기화 및 dropout, 활성함수 설정
    initialize_network(&network, layer_data, 0.1, sigmoid, sigmoid_d);

    // 데이터 준비
    int train_data_size = data_count(TRAIN_DATA_PATH);

    double** train_features = NULL;
    double** train_targets = NULL;

    prepare_data(TRAIN_DATA_PATH, train_data_size, &train_features, &train_targets);

    //학습
    train(&network, train_features, train_targets);

    memory_clear(train_features, train_targets, TRAIN_DATA_SIZE);

    free_network(&network);

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
        sum += diff * diff;
    }
    return sum / OUTPUT_SIZE;
}

void mse_d(double* y_true, double* y_pred, double* output) {
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        *(output + i) = 2.0 * (*(y_pred + i) - *(y_true + i)) / OUTPUT_SIZE;
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
    layer->batch_weights = (double*)malloc(input_nodes * output_nodes * sizeof(double));
    if (layer->weights == NULL) {
        fprintf(stderr, "layer->weights 메모리 할당 실패\n");
        exit(1);
    }

    if (layer->batch_weights == NULL) {
        fprintf(stderr, "layer->batch_weights 메모리 할당 실패\n");
        exit(1);
    }
    double a = sqrt(1.0 / (layer->input_nodes + layer->output_nodes)); //Sigmoid 초기화
    for (int i = 0; i < input_nodes; i++) {
        for (int j = 0; j < output_nodes; j++) {
            *(layer->weights + i * output_nodes + j) = normal_distribution(0, a);  // 표준편차가 a인 정규분포 값
            *(layer->batch_weights + i * output_nodes + j) = 0.0;
        }
    }

    layer->g = (double*)malloc(input_nodes * sizeof(double));
    layer->inputs = (double*)malloc(input_nodes * sizeof(double));
    layer->biases = (double*)malloc(output_nodes * sizeof(double));
    layer->batch_biases = (double*)malloc(output_nodes * sizeof(double));
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
    if (layer->outputs == NULL) {
        fprintf(stderr, "layer->outputs 메모리 할당 실패\n");
        exit(1);
    }
    for (int i = 0; i < input_nodes; i++) {
        double a = sqrt(2.0 / layer->input_nodes);
        *(layer->g + i) = 0.0;
        *(layer->inputs + i) = 0.0;
    }
    for (int i = 0; i < output_nodes; i++) {
        double a = sqrt(2.0 / layer->input_nodes);
        *(layer->biases + i) = normal_distribution(0, a);  // 표준편차가 a인 정규분포 값
        *(layer->outputs + i) = 0.0;
        *(layer->batch_biases + i) = 0.0;
    }
}

/*신경망 초기화 함수*/
void initialize_network(Network* network, int* num, double dropout_rate,
    void (*activation_function)(double*, double*, int), void (*activation_function_d)(double*, double*, int)) {
    network->layers = (Layer*)malloc(sizeof(Layer) * LAYER_NUM);
    if (network->layers == NULL) {
        fprintf(stderr, "network->layers 메모리 할당 실패\n");
        exit(1);
    }
    for (int j = 0; j < LAYER_NUM; j++) {
        printf("layer %d(%d>%d)\n", j, *(num + j), *(num + j + 1));
        initialize_layer((network->layers + j), *(num + j), *(num + j + 1), j, dropout_rate, activation_function, activation_function_d);
    }
    network->eval = false;
}

/*순전파 함수*/
void forward(Network* network, double* input) {
    double* current_input = input;
    // 각 레이어를 순회하며 forward_layer 호출
    for (int i = 0; i < LAYER_NUM; i++) {
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

/*각 레이어의 역전파를 진행하는 함수*/
void update_layer(Layer* layer) {
    int i, j;
    // 배치 가중치 업데이트
    for (i = 0; i < layer->input_nodes; i++) {
        for (j = 0; j < layer->output_nodes; j++) {
            *(layer->batch_weights + i * layer->output_nodes + j) = *(layer->batch_weights + i * layer->output_nodes + j) / BATCH_SIZE;
            *(layer->weights + i * layer->output_nodes + j) += *(layer->batch_weights + i * layer->output_nodes + j);
            *(layer->batch_weights + i * layer->output_nodes + j) = 0.0;
        }
    }

    // 바이어스 업데이트
    for (j = 0; j < layer->output_nodes; j++) {
        *(layer->batch_biases + j) = *(layer->batch_biases + j) / BATCH_SIZE;
        *(layer->biases + j) += *(layer->batch_biases + j);
        *(layer->batch_biases + j) = 0.0;
    }
}
void backward_update(Network* network) {
    for (int i = LAYER_NUM - 1; i >= 0; i--) {
        update_layer(network->layers + i);
    }
}

void backward_layer(Layer* layer, double* next_g) {
    int i, j;
    // 업데이트 계산
    for (i = 0; i < layer->input_nodes; i++) {
        for (j = 0; j < layer->output_nodes; j++) {
            *(layer->batch_weights + i * layer->output_nodes + j) -= LR * *(next_g + j) * *(layer->inputs + i);
        }
    }
    // 바이어스 업데이트 계산
    for (j = 0; j < layer->output_nodes; j++) {
        *(layer->batch_biases + j) -= LR * *(next_g + j);
    }
    layer->activation_function_d(layer->inputs, layer->inputs, layer->input_nodes);

    // 그래디언트 계산 후 이전 레이어로 전달
    for (i = 0; i < layer->input_nodes; i++) {
        *(layer->g + i) = 0.0;
        for (j = 0; j < layer->output_nodes; j++) {
            *(layer->g + i) += *(layer->weights + i * layer->output_nodes + j) * *(next_g + j);
        }
        *(layer->g + i) *= *(layer->inputs + i);
    }
}

/*역전파 함수*/
void backward(Network* network, double* next_g) {
    double* layer_d_output = next_g;

    for (int i = LAYER_NUM - 1; i >= 0; i--) {
        backward_layer(network->layers + i, layer_d_output);

        layer_d_output = (*(network->layers + i)).g;
    }
}

/*신경망의 weights를 출력하는 함수*/
void print_weights(Network* network) {
    for (int i = 0; i < LAYER_NUM; i++) {
        printf("<layer %d>\n", i);
        printf("%d -> %d\n", (*(network->layers + i)).input_nodes, (*(network->layers + i)).output_nodes);
        printf("weights : \n");
        for (int j = 0; j < (*(network->layers + i)).input_nodes; j++) {
            for (int k = 0; k < (*(network->layers + i)).output_nodes; k++) {
                printf("%f, ", *((*(network->layers + i)).weights + j * (*(network->layers + i)).output_nodes + k));
            }
            printf("\n");
        }
        printf("bias : \n");
        for (int j = 0; j < (*(network->layers + i)).output_nodes; j++) {
            printf("%f, ", *((*(network->layers + i)).biases + j));
        }
        printf("\ninputs : \n");
        for (int j = 0; j < (*(network->layers + i)).input_nodes; j++) {
            printf("%f, ", *((*(network->layers + i)).inputs + j));
        }
        printf("\noutputs : \n");
        for (int j = 0; j < (*(network->layers + i)).output_nodes; j++) {
            printf("%f, ", *((*(network->layers + i)).outputs + j));
        }
        printf("\n_________\n");
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


/*성능 검증 함수*/
void eval(Network* network, double** features, double** targets, int size) {
    network->eval = true;
    printf("\n______eval_____\n");

    double accuracy = 0.;

    double* last_output = (*(network->layers + LAYER_NUM - 1)).outputs;

    for (int i = 0; i < size; i++) {
        forward(network, *(features + i)); // 순방향 전파

        int true_label = 0, predicted_label = 0;
        double max_true_val = **(targets + i), max_pred_val = *(last_output + 0);

        // 실제 레이블과 예측 레이블 찾기
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

        // 행렬 업데이트
        if (predicted_label == true_label) {
            accuracy += 1.0;
        }
    }

    accuracy /= size;

    // 결과 출력
    printf("Accuracy: %.2f%\n", accuracy*10);
    network->eval = false;

    // 동적 할당된 메모리 해제
}

/*조건대로 학습을 진행하는 함수*/
void train(Network* network, double** train_features, double** train_targets) {
    time_t start = time(NULL);
    printf("\n______train_____\n");
    for (int epoch = 0; epoch < EPOCH; epoch++) {
        double total_loss = 0.0;
        printf("Epoch %d start\n", epoch);
        for (int i = 0; i < TRAIN_DATA_SIZE; i += BATCH_SIZE) {
            double batch_loss = 0.0;

            for (int j = i; j < i + BATCH_SIZE; j++) {
                forward(network, *(train_features + j));

                double loss = cross_entropy_loss(*(train_targets + j), (*(network->layers + LAYER_NUM - 1)).outputs);
                batch_loss += loss;

                cross_entropy_loss_d(*(train_targets + j), (*(network->layers + LAYER_NUM - 1)).outputs, (*(network->layers + LAYER_NUM - 1)).outputs);

                backward(network, (*(network->layers + LAYER_NUM - 1)).outputs);
            }
            backward_update(network);
            batch_loss /= BATCH_SIZE;
            total_loss += batch_loss;
            if (i % 5000 == 0 && i != 0) {
                printf("  %d step loss: %f\n", i, batch_loss);
            }
        }
        total_loss /= TRAIN_DATA_SIZE / (double)(BATCH_SIZE);

        printf("Epoch %d, Total train loss: %f\n", epoch, total_loss);
        
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
        sprintf(path, "%s\\%d\\*", baseFolder, folder);

        if ((hFile = _findfirst(path, &fileinfo)) == -1L) {
            printf("폴더 접근 불가: '%s'\n", path);
            continue;
        }
        else {
            do {
                if (!(fileinfo.attrib & _A_SUBDIR)) { // 디렉토리가 아닌 경우에만 카운트
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
        *(features + i) = (double)(*(image + i))/255.0;
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
        sprintf(path, "%s\\%d\\*", baseFolder, folder);

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
                    if (index%5000==0) {
                        printf("이미지 로드 중...%d개\n", index);
                    }
                    if (index >= max) {
                        return;
                    }
                }
            } while (_findnext(hFile, &fileinfo) == 0);
            _findclose(hFile);
        }
    }
}

/*dataset 전처리 함수*/
void prepare_data(const char* train_data_path,
    int train_data_size, double*** train_features, double*** train_targets) {

    time_t start = time(NULL);
    printf("______data load_____\ntrain:%d개\n", train_data_size);

    *train_features = (double**)malloc(sizeof(double*) * train_data_size);
    *train_targets = (double**)malloc(sizeof(double*) * train_data_size);

    for (int i = 0; i < train_data_size; i++) {
        *(*train_features + i) = (double*)malloc(sizeof(double) * INPUT_SIZE);
        *(*train_targets + i) = (double*)malloc(sizeof(double) * OUTPUT_SIZE);
    }
    
    load_data(train_data_path, *train_features, *train_targets, TRAIN_DATA_SIZE);
    for (int i = 0; i < 5; i++) {
        mix(*train_features, *train_targets, train_data_size);
    }

    time_t end = time(NULL);
    printf("데이터셋 로드 및 전처리 소요시간: %lf\n", (double)(end - start));
}

/*dataset 관련 메모리 해제*/
void memory_clear(
    double** train_features, double** train_targets,
    int size)
{
    for (int i = 0; i < size; i++) {
        free(*(train_features + i));
        free(*(train_targets + i));
    }
    free(train_features);
    free(train_targets);
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
    for (int i = LAYER_NUM - 1; i >= 0; i--) {
        free_layer(network->layers + i);
    }
    free(network->layers);
}
