#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <stdbool.h>
#include <float.h>

#define MAX_LINE_LENGTH 1024

// 상수 관련
#define PI 3.141592
#define e 2.7182818

// model 관련
#define INPUT_WIDTH 28
#define INPUT_HEIGHT 28
#define INPUT_SIZE 784

#define CONV_LAYER_NUM 3
#define LAYER_NUM 2

#define OUTPUT_SIZE 10 // train data 출력 개수


// dataset 관련
#define IMAGEPATH "../mnist/train-images.idx3-ubyte"
#define LABELPATH "../mnist/train-labels.idx1-ubyte"

#define TOTAL_DATA_SIZE 100
#define TRAIN_DATA_SIZE 90
#define TEST_DATA_SIZE TOTAL_DATA_SIZE-TRAIN_DATA_SIZE


// train 관련 
#define EPOCH 2 // 전체 데이터 반복 횟수
#define BATCH 10// batch size
#define LR 0.0004

#define WEIGHT_CLIP 10
#define BIAS_CLIP 10

// fc layer
typedef struct {
    int id;
    int type;
    int input_nodes;
    int output_nodes;
    double dropout_rate;
    void (*activation_function)(double*, double*, int);
    void (*activation_function_d)(double*, double*, int);
    double** weights;
    double** delta_weights;
    double* biases;
    double* outputs;
    double* inputs;
    double* g;
} Layer;

// CNN layer
typedef struct {
    int type;
    int filter_size;
    int stride;
    int padding;
    int input_depth;
    int input_width;
    int output_depth;
    int output_width;
    void (*activation_function)(double*, double*, int);
    void (*activation_function_d)(double*, double*, int);
    double**** filters; 
    double* biases; 
    double* outputs;
    double* inputs;
    double dropout_rate;
} ConvLayer;

// pooling layer
typedef struct {
    int type;
    double* outputs;
    double* inputs;
    void (*pooling_function)(double*, double*, int, int, int, int, int);
    int pool_size;
    int stride;
    int padding;
    int input_width;
    int input_depth;
    int output_width;
} PoolingLayer;

typedef struct {
    int type;
    double* outputs;
    double* inputs;
    int input_depth;
    int input_width;
    int input_height;
    int output_size;
} FlattenLayer;

// 네트워크 구조체
typedef struct {
    ConvLayer* conv_layers;
    Layer* layers;
    PoolingLayer* pool_layers;
    FlattenLayer* flatten;
    bool eval;
} Network;

// 활성화 함수 및 미분 함수 등
void Softmax(double* input, double* output, int length);
void Softmax_d(double* input, double* output, int length);

void sigmoid(double* input, double* output, int length);
void sigmoid_d(double* input, double* output, int length);

void ReLU(double* input, double* output, int length);
void ReLU_d(double* input, double* output, int length);

double cross_entropy_loss(double* y_true, double* y_pred);
void cross_entropy_loss_d(double* y_true, double* y_pred, double* output);

double mse(double* y_true, double* y_pred);
void mse_d(double* y_true, double* y_pred, double* output);

// 정규분포 난수 생성 함수
double normal_distribution(double mean, double std);

void clipping(double* input, int clip);

// 메모리 반납
void free_layer(Layer* layer);
void free_network(Network* network);

// 신경망 초기화 함수들
void initialize_layer(Layer* layer, int input_nodes, int output_nodes, int id, double dropout_rate, void (*activation_function)(double*, double*, int), void (*activation_function_d)(double*, double*, int));
void initialize_network(Network* network);
void make_network(Network* network);

void initialize_flatten_layer(FlattenLayer* layer, int input_depth, int input_width, int input_height) {
    layer->type = 3;
    layer->input_depth = input_depth;
    layer->input_width = input_width;
    layer->input_height = input_height;
    layer->output_size = input_depth * input_width * input_height;
    layer->outputs = (double*)malloc(layer->output_size * sizeof(double));
}

void initialize_pooling_layer(PoolingLayer* layer, int input_depth, int input_width, int pool_size, int stride, int padding) {
    layer->type = 2; 
    layer->pool_size = pool_size;
    layer->stride = stride;
    layer->padding = padding;
    layer->input_width = input_width;
    layer->input_depth = input_depth;
    layer->output_width = (int)((input_width + 2 * padding - pool_size) / stride) + 1;

    int output_height = layer->output_width; 
    layer->outputs = (double*)malloc(layer->input_depth * layer->output_width * layer->output_width * sizeof(double));
}

// CNN
void initialize_conv_layer(ConvLayer* layer, int input_depth, int input_width, int output_depth, int filter_size, int stride, int padding,
    double dropout_rate, void (*activation_function)(double*, double*, int), void (*activation_function_d)(double*, double*, int)) {
    layer->type = 1;
    layer->activation_function = activation_function;
    layer->activation_function_d = activation_function_d;
    layer->input_depth = input_depth;
    layer->input_width = input_width;
    layer->output_depth = output_depth;
    layer->output_width = (int)((1 + (layer->input_width + 2 * layer->padding - layer->filter_size) / layer->stride));
    layer->filter_size = filter_size;
    layer->stride = stride;
    layer->padding = padding;
    layer->outputs = (double*)malloc(layer->output_depth * layer->output_width * layer->output_width * sizeof(double));
    layer->dropout_rate = dropout_rate;

    double a = sqrt(2.0 / 700);
    layer->filters = (double****)malloc(output_depth * sizeof(double***));
    for (int d = 0; d < output_depth; d++) {
        *(layer->filters + d) = (double***)malloc(input_depth * sizeof(double**));
        for (int i = 0; i < input_depth; i++) {
            *(*(layer->filters + d) + i) = (double**)malloc(filter_size * sizeof(double*));
            for (int j = 0; j < filter_size; j++) {
                *(*(*(layer->filters + d) + i) + j) = (double*)malloc(filter_size * sizeof(double));
                for (int k = 0; k < filter_size; k++) {
                    *(*(*(*(layer->filters + d) + i) + j) + k) = normal_distribution(0, a);
                }
            }
        }
    }

    layer->biases = (double*)malloc(output_depth * sizeof(double));
    for (int i = 0; i < output_depth; i++) {
        *(layer->biases + i) = normal_distribution(0, a);
    }
}

void forward_conv_layer(ConvLayer* layer, double* input, bool is_eval);
void forward_flatten_layer(FlattenLayer* layer, double* input);
void forward_pooling_layer(PoolingLayer* layer, double* input);
// 순방향 전파 함수들
void forward_layer(Layer* layer, double* input, bool is_eval);
void forward(Network* network, double* input);

// 역전파 함수들
void backward_layer(Layer* layer, double* d_output, double learning_rate);
void backward(Network* network, double* d_output, double learning_rate);

// 가중치 및 레이어 정보 출력 확인
void print_weights(Network* network);

// 데이터셋 관련
void load_data(const char* imagepath, const char* labelpath, double** total_features, double** total_targets);

void prepare_data(const char* imagepath, const char* labelpath,
    double*** total_features, double*** total_targets,
    double*** train_features, double*** train_targets,
    double*** test_features, double*** test_targets);

void print_data(double* image, double* label);

// 데이터 전처리
void mix(double** features, double** target);

void split_train_test(double** total_features, double** total_target,
    double** train_features, double** train_target,
    double** test_features, double** test_target);

void one_hot_encode(int label, double* output);


// 검증
void eval(Network* network, double** test_feature, double** test_target);

// 학습
void train(Network* network, double** train_features, double** train_targets);

// 메모리 해제
void memory_clear(
    double** total_features, double** total_targets,
    double** train_features, double** train_targets,
    double** test_features, double** test_targets);

int main() {
    srand(time(NULL));  // 난수 생성기 초기화

    // 학습 전

    // 데이터 준비
    double** total_features = NULL;
    double** total_targets = NULL;

    double** train_features = NULL;
    double** train_targets = NULL;

    double** test_features = NULL;
    double** test_targets = NULL;

    prepare_data(IMAGEPATH, LABELPATH, &total_features, &total_targets, &train_features, &train_targets, &test_features, &test_targets);

    //하나 확인
    print_data(total_features[0], total_targets[0]);
    print_data(total_features[10], total_targets[10]);

    // 신경망 초기화
    Network network; 
    make_network(&network);
    eval(&network, test_features, test_targets);

    print_weights(&network);

    return 0;
    //학습
    train(&network, train_features, train_targets);

    //검증
    eval(&network, test_features, test_targets);

    memory_clear(total_features, total_targets,
        train_features, train_targets,
        test_features, test_targets);

    free_network(&network);

    return 0;
}

void memory_clear(
    double** total_features, double** total_targets,
    double** train_features, double** train_targets,
    double** test_features, double** test_targets)
{
    free(total_features);
    free(total_targets);
    free(train_features);
    free(train_targets);
    free(test_features);
    free(test_targets);
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
        *(output + i) = exp(*(input + i) - max);
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
        *(output + i) = 1.0 / (1.0 + exp(-*(input + i)));
    }
}

void sigmoid_d(double* input, double* output, int length) {
    for (int i = 0; i < length; i++) {
        double sigmoid = 1.0 / (1.0 + exp(-*(input + i)));
        *(output + i) = sigmoid * (1 - sigmoid);
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
        sum += diff * diff;
    }
    return sum / OUTPUT_SIZE;
}

void mse_d(double* y_true, double* y_pred, double* output) {
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        *(output + i) = 2.0 * (*(y_pred + i) - *(y_true + i)) / OUTPUT_SIZE;
    }
}

double normal_distribution(double mean, double std) {
    double U1 = (double)rand() / RAND_MAX;
    double U2 = (double)rand() / RAND_MAX;
    return mean + std * sqrt(-2 * log(U1)) * cos(2 * PI * U2);
}

void clipping(double* input, int clip) {
    if (*input > clip) {
        *input = clip;
    }
    else if (*input < -clip) {
        *input = -clip;
    }
}

void initialize_layer(Layer* layer, int input_nodes, int output_nodes, int id, double dropout_rate,
    void (*activation_function)(double*, double*, int), void (*activation_function_d)(double*, double*, int)) {
    layer->type = 1;
    layer->id = id;
    layer->input_nodes = input_nodes;
    layer->output_nodes = output_nodes;
    layer->dropout_rate = dropout_rate;

    layer->activation_function = activation_function;
    layer->activation_function_d = activation_function_d;

    layer->weights = (double**)malloc(input_nodes * sizeof(double*));
    layer->delta_weights = (double**)malloc(input_nodes * sizeof(double*));

    if (layer->weights == NULL) {
        fprintf(stderr, "layer->weights 메모리 할당 실패\n");
        exit(1);
    }
    if (layer->delta_weights == NULL) {
        fprintf(stderr, "layer->delta_weights 메모리 할당 실패\n");
        exit(1);
    }
    for (int i = 0; i < input_nodes; i++) {
        double a = sqrt(2.0 / layer->input_nodes);

        *(layer->weights + i) = (double*)malloc(output_nodes * sizeof(double));
        *(layer->delta_weights + i) = (double*)malloc(output_nodes * sizeof(double));
        if (*(layer->weights + i) == NULL) {
            fprintf(stderr, "*(layer->weights + i) 메모리 할당 실패\n");
            exit(1);
        }
        if (*(layer->delta_weights + i) == NULL) {
            fprintf(stderr, "*(layer->delta_weights + i) 메모리 할당 실패\n");
            exit(1);
        }
        for (int j = 0; j < output_nodes; j++) {
            *(*(layer->weights + i) + j) = normal_distribution(0, a);  // 표준편차가 a인 정규분포 값
            *(*(layer->delta_weights + i) + j) = 0.0;
        }
    }

    layer->g = (double*)malloc(input_nodes * sizeof(double));
    layer->inputs = (double*)malloc(input_nodes * sizeof(double));
    layer->biases = (double*)malloc(output_nodes * sizeof(double));
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
    }
}


void initialize_network(Network* network) {
    network->conv_layers = (ConvLayer*)malloc(sizeof(ConvLayer) * CONV_LAYER_NUM);
    network->layers = (Layer*)malloc(sizeof(Layer) * LAYER_NUM);
    network->pool_layers = (PoolingLayer*)malloc(sizeof(PoolingLayer) * CONV_LAYER_NUM);
    network->flatten = (FlattenLayer*)malloc(sizeof(FlattenLayer));
    if ((network->layers == NULL) || (network->conv_layers == NULL)) {
        fprintf(stderr, "network->layers 메모리 할당 실패\n");
        exit(1);
    }
    initialize_conv_layer((network->conv_layers + 0), 1, 28, 8, 3, 1, 1, 0.01, ReLU, ReLU_d);
    initialize_pooling_layer((network->pool_layers + 0), 8, 28, 2, 2, 0);
    initialize_conv_layer((network->conv_layers + 1), 8, 14, 16, 3, 1, 1, 0.01, ReLU, ReLU_d);
    initialize_pooling_layer((network->pool_layers + 1), 16, 14, 2, 2, 0);
    initialize_conv_layer((network->conv_layers + 2), 16, 7, 32, 3, 1, 1, 0.01, ReLU, ReLU_d);
    initialize_pooling_layer((network->pool_layers + 2), 32, 7, 7, 1, 0);

    initialize_flatten_layer((network->flatten), 32, 1, 1);

    initialize_layer((network->layers + 0), 32, 16, 0, 0.01, ReLU, ReLU_d);
    initialize_layer((network->layers + 1), 16, 10, 1, 0.0, ReLU, ReLU_d);
    network->eval = false;
}

void make_network(Network* network) {
    initialize_network(network);
}

void forward(Network* network, double* input) {
    double* current_input = input;
    int conv_layer_index = 0;
    int pool_layer_index = 0;

    // Convolutional Layer와 Pooling Layer를 포함하여 forward 연산 수행
    for (int i = 0; i < CONV_LAYER_NUM; i++) {
        forward_conv_layer((network->conv_layers + conv_layer_index), current_input, network->eval);
        current_input = (network->conv_layers + conv_layer_index)->outputs;
        conv_layer_index++;
        forward_pooling_layer((network->pool_layers + pool_layer_index), current_input);
        current_input = (network->pool_layers + pool_layer_index)->outputs;
        pool_layer_index++;
    }

    // Flatten layer 처리
    forward_flatten_layer(network->flatten, current_input);
    current_input = network->flatten->outputs;

    // 일반 레이어에 대한 forward 연산 수행
    for (int i = 0; i < LAYER_NUM; i++) {
        forward_layer((network->layers + i), current_input, network->eval);
        current_input = (network->layers + i)->outputs;
    }
}


void forward_layer(Layer* layer, double* input, bool is_eval) {
    for (int i = 0; i < layer->input_nodes; i++) {
        *(layer->inputs + i) = *(input + i);
    }

    if ((!is_eval) && (layer->dropout_rate > 0.0)) {
        for (int i = 0; i < layer->input_nodes; i++) {
            if ((double)rand() / RAND_MAX < layer->dropout_rate) {
                *(layer->inputs + i) = 0.0; // 드롭아웃 비율만큼 뉴런을 비활성화
            }
        }
    }
    // weight와 input을 곱하고, bias를 더한 후, 활성화 함수 적용
    for (int j = 0; j < layer->output_nodes; j++) {
        *(layer->outputs + j) = 0.0;
        for (int i = 0; i < layer->input_nodes; i++) {
            *(layer->outputs + j) += *(*(layer->weights + i) + j) * *(layer->inputs + i);
        }
        *(layer->outputs + j) += *(layer->biases + j);
    }

    layer->activation_function(layer->outputs, layer->outputs, layer->output_nodes);
    if (layer->id == LAYER_NUM - 1) {
        Softmax(layer->outputs, layer->outputs, layer->output_nodes);
    }
}

void forward_conv_layer(ConvLayer* layer, double* input, bool is_eval) {
    for (int d = 0; d < layer->output_depth; d++) {
        for (int oh = 0; oh < layer->output_width; oh++) {
            for (int ow = 0; ow < layer->output_width; ow++) {
                double sum = 0.0;
                for (int fd = 0; fd < layer->input_depth; fd++) {
                    for (int fh = 0; fh < layer->filter_size; fh++) {
                        for (int fw = 0; fw < layer->filter_size; fw++) {
                            int ih = oh * layer->stride + fh - layer->padding;
                            int iw = ow * layer->stride + fw - layer->padding;
                            if (ih >= 0 && ih < layer->input_width && iw >= 0 && iw < layer->input_width) {
                                sum += *(*(*(*(layer->filters + d) + fd) + fh) + fw) * *(input + fd * layer->input_width * layer->input_width + ih * layer->input_width + iw);
                            }
                        }
                    }
                }
                sum += *(layer->biases + d);
                *(layer->outputs + d * layer->output_width * layer->output_width + oh * layer->output_width + ow) = sum;
            }
        }
    }
    layer->activation_function(layer->outputs, layer->outputs, layer->output_depth * layer->output_width * layer->output_width);
}

void forward_pooling_layer(PoolingLayer* layer, double* input) {
    int output_width = (layer->input_width - layer->pool_size) / layer->stride + 1;

    for (int d = 0; d < layer->input_depth; d++) {
        for (int ph = 0; ph < output_width; ph++) {
            for (int pw = 0; pw < output_width; pw++) {
                double max_value = -INFINITY;
                for (int fh = 0; fh < layer->pool_size; fh++) {
                    for (int fw = 0; fw < layer->pool_size; fw++) {
                        int ih = ph * layer->stride + fh;
                        int iw = pw * layer->stride + fw;
                        if (ih < layer->input_width && iw < layer->input_width) {
                            double current_value = *(input + d * layer->input_width * layer->input_width + ih * layer->input_width + iw);
                            if (current_value > max_value) {
                                max_value = current_value;
                            }
                        }
                    }
                }
                *(layer->outputs + d * layer->output_width * layer->output_width + ph * layer->output_width + pw) = max_value;
            }
        }
    }
}

void forward_flatten_layer(FlattenLayer* layer, double* input) {
    int input_index = 0;
    for (int i = 0; i < layer->output_size; i++) {
        *(layer->outputs + i) = *(input + input_index);
        input_index++;
    }
}

void backward_layer(Layer* layer, double* d_output, double learning_rate) {
    int i, j;
    // 가중치 업데이트
    for (i = 0; i < layer->input_nodes; i++) {
        for (j = 0; j < layer->output_nodes; j++) {
            *(*(layer->delta_weights + i) + j) = 0.11 * *(*(layer->delta_weights + i) + j) - learning_rate * *(d_output + j) * *(layer->inputs + i);
            *(*(layer->weights + i) + j) += *(*(layer->delta_weights + i) + j);
            clipping((*(layer->weights + i) + j), WEIGHT_CLIP);
        }
    }
    // 편향 업데이트
    for (j = 0; j < layer->output_nodes; j++) {
        *(layer->biases + j) -= learning_rate * *(d_output + j);
        clipping((layer->biases + j), BIAS_CLIP);
    }
    layer->activation_function_d(layer->inputs, layer->inputs, layer->input_nodes);
    // 그래디언트 계산
    for (i = 0; i < layer->input_nodes; i++) {
        *(layer->g + i) = 0.0;
        for (j = 0; j < layer->output_nodes; j++) {
            *(layer->g + i) += *(*(layer->weights + i) + j) * *(d_output + j);
        }
        *(layer->g + i) *= *(layer->inputs + i);
    }
}

void backward(Network* network, double* d_output, double learning_rate) {
    double* layer_d_output = d_output;

    for (int i = LAYER_NUM - 1; i >= 0; i--) {
        backward_layer(network->layers + i, layer_d_output, learning_rate);

        layer_d_output = (*(network->layers + i)).g;
    }
}

void print_weights(Network* network) {
    printf("______________________________\n");
    for (int i = 0; i < CONV_LAYER_NUM; i++) {
        printf("<conv layer %d>\n", i);
        printf("input depth: %d, input width: %d, padding: %d, stride: %d \n output depth: %d, output width: %d, \n", 
            (*(network->conv_layers + i)).input_depth, (*(network->conv_layers + i)).input_width,
            (*(network->conv_layers + i)).padding, (*(network->conv_layers + i)).stride,
            (*(network->conv_layers + i)).output_depth, (*(network->conv_layers + i)).output_width);
        printf("weights : \n");
        for (int d = 0; d < 1; d++) {
            for (int id = 0; id < 1; id++) {
                for (int h = 0; h < (*(network->conv_layers + i)).filter_size; h++) {
                    for (int w = 0; w < (*(network->conv_layers + i)).filter_size; w++) {
                        printf("%f ", *(*(*(*((*(network->conv_layers + i)).filters + d) + id) + h) + w));
                    }printf("\n");
                }printf("_______\n");
            }printf("_________________\n");
        }
        printf("output:\n");
        for (int d = 0; d < (*(network->conv_layers + i)).output_depth; d++) {
            for (int h = 0; h < (*(network->conv_layers + i)).output_width; h++) {
                for (int w = 0; w < (*(network->conv_layers + i)).output_width; w++) {
                    printf("%f ", *((network->conv_layers + i)->outputs
                        + d * (*(network->conv_layers + i)).output_width * (*(network->conv_layers + i)).output_width
                        + h * (*(network->conv_layers + i)).output_width + w));
                }printf("\n");
            }printf("_______\n");
        }

        printf("<maxpooling layer %d>\n", i);
        printf("output:\n");
        for (int d = 0; d < (*(network->pool_layers + i)).input_depth; d++) {
            for (int h = 0; h < (*(network->pool_layers + i)).output_width; h++) {
                for (int w = 0; w < (*(network->pool_layers + i)).output_width; w++) {
                    printf("%f ", *((network->pool_layers + i)->outputs 
                        + d * (*(network->pool_layers + i)).output_width* (*(network->pool_layers + i)).output_width 
                        + h * (*(network->pool_layers + i)).output_width + w));
                }printf("\n");
            }printf("_______\n");
        }
    }

    // Flatten layer 처리
    printf("<flatten>\n");
    for (int i = 0; i < (*(network->flatten)).output_size; i++) {
        printf("%f\n", *((*(network->flatten)).outputs + i));
    }

    for (int i = 0; i < LAYER_NUM; i++) {
        printf("<layer %d>\n", i);
        printf("%d -> %d\n", (*(network->layers + i)).input_nodes, (*(network->layers + i)).output_nodes);
        printf("weights : \n");
        for (int j = 0; j < (*(network->layers + i)).input_nodes; j++) {
            for (int k = 0; k < (*(network->layers + i)).output_nodes; k++) {
                printf("%f, ", *(*((*(network->layers + i)).weights + j) + k));
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

void mix(double** features, double** target) {
    double temp;
    int rn;
    for (int i = 0; i < TOTAL_DATA_SIZE - 1; i++) {
        rn = rand() % (TOTAL_DATA_SIZE - i) + i;
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


void split_train_test(double** total_features, double** total_targets,
    double** train_features, double** train_targets,
    double** test_features, double** test_targets) {
    for (int i = 0; i < TRAIN_DATA_SIZE; i++) {
        for (int k = 0; k < INPUT_SIZE; k++) {
            *(*(train_features + i) + k) = *(*(total_features + i) + k);
        }
        for (int k = 0; k < OUTPUT_SIZE; k++) {
            *(*(train_targets + i) + k) = *(*(total_targets + i) + k);
        }
    }
    for (int i = 0; i < TEST_DATA_SIZE; i++) {
        for (int k = 0; k < INPUT_SIZE; k++) {
            *(*(test_features + i) + k) = *(*(total_features + i + TRAIN_DATA_SIZE) + k);
        }
        for (int k = 0; k < OUTPUT_SIZE; k++) {
            *(*(test_targets + i) + k) = *(*(total_targets + i + TRAIN_DATA_SIZE) + k);
        }
    }
}

void eval(Network* network, double** test_features, double** test_targets) {
    network->eval = true;
    printf("\n______eval_____\n");

    int* true_positive = (int*)calloc(OUTPUT_SIZE, sizeof(int));
    int* false_positive = (int*)calloc(OUTPUT_SIZE, sizeof(int));
    int* false_negative = (int*)calloc(OUTPUT_SIZE, sizeof(int));
    double* precision = (double*)calloc(OUTPUT_SIZE, sizeof(double));
    double* recall = (double*)calloc(OUTPUT_SIZE, sizeof(double));
    double* f1 = (double*)calloc(OUTPUT_SIZE, sizeof(double));
    double macro_f1 = 0., macro_precision = 0., macro_recall = 0., accuracy = 0.;

    double* last_output = (*(network->layers + LAYER_NUM - 1)).outputs;

    for (int i = 0; i < TEST_DATA_SIZE; i++) {
        forward(network, *(test_features + i)); // 순방향 전파

        int true_label = 0, predicted_label = 0;
        double max_true_val = **(test_targets + i), max_pred_val = *(last_output + 0);

        // 실제 레이블과 예측 레이블 찾기
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (*(*(test_targets + i) + j) > max_true_val) {
                max_true_val = *(*(test_targets + i) + j);
                true_label = j;
            }
            if (*(last_output + j) > max_pred_val) {
                max_pred_val = *(last_output + j);
                predicted_label = j;
            }
        }

        // 행렬 업데이트
        if (predicted_label == true_label) {
            *(true_positive + true_label) += 1;
            accuracy += 1.0;
        }
        else {
            *(false_positive + predicted_label) += 1;
            *(false_negative + true_label) += 1;
        }
    }

    // 정밀도, 재현율, F1 점수 계산
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        if (*(true_positive + i) + *(false_positive + i) == 0) {
            *(precision + i) = 0;
        }
        else {
            *(precision + i) = (double)*(true_positive + i) / (double)(*(true_positive + i) + *(false_positive + i));
        }

        if (*(true_positive + i) + *(false_negative + i) == 0) {
            *(recall + i) = 0;
        }
        else {
            *(recall + i) = (double)*(true_positive + i) / (double)(*(true_positive + i) + *(false_negative + i));
        }

        if (*(precision + i) + *(recall + i) == 0) {
            *(f1 + i) = 0;
        }
        else {
            *(f1 + i) = 2 * (*(precision + i)) * (*(recall + i)) / (*(precision + i) + *(recall + i));
        }

        macro_precision += *(precision + i);
        macro_recall += *(recall + i);
        macro_f1 += *(f1 + i);
    }
    macro_precision /= OUTPUT_SIZE;
    macro_recall /= OUTPUT_SIZE;
    macro_f1 /= OUTPUT_SIZE;
    accuracy /= TEST_DATA_SIZE;

    // 결과 출력
    printf("Macro Precision: %.2f\n", macro_precision);
    printf("Macro Recall: %.2f\n", macro_recall);
    printf("Macro F1 Score: %.2f\n", macro_f1);
    printf("Accuracy: %.2f\n", accuracy);
    network->eval = false;

    // 동적 할당된 메모리 해제

    free(true_positive);
    free(false_positive);
    free(false_negative);
    free(precision);
    free(recall);
    free(f1);
}

void train(Network* network, double** train_features, double** train_targets) {
    printf("\n______train_____\n");
    for (int epoch = 0; epoch < EPOCH; epoch++) {
        double total_loss = 0.0;
        for (int i = 0; i < TRAIN_DATA_SIZE; i += BATCH) {
            double batch_loss = 0.0;

            for (int j = i; j < i + BATCH; j++) {
                forward(network, *(train_features + j));

                double loss = cross_entropy_loss(*(train_targets + j), (*(network->layers + LAYER_NUM - 1)).outputs);
                batch_loss += loss;

                cross_entropy_loss_d(*(train_targets + j), (*(network->layers + LAYER_NUM - 1)).outputs, (*(network->layers + LAYER_NUM - 1)).outputs);

                backward(network, (*(network->layers + LAYER_NUM - 1)).outputs, LR);
            }

            batch_loss /= BATCH;
            total_loss += batch_loss;
            if (i % 2000 == 0) {
                printf("  %d step loss: %f\n", i, batch_loss);
            }
        }
        total_loss /= TRAIN_DATA_SIZE / (double)(BATCH);

        if (epoch % 1 == 0) {
            printf("Epoch %d, Loss: %f\n", epoch, total_loss);
        }
    }
}

void one_hot_encode(int label, double* output) {
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = (i == label) ? 1.0 : 0.0;
    }
}

void print_data(double* image, double* label) {
    int k = 0;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        if (*(label + i) == 1) {
            k = i;
        }
    }
    printf("\n____%d____\n", k);
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            if (*(image + i * 28 + j) > 0.9) {
                printf("##");
            }
            else if (*(image + i * 28 + j) > 0.7) {
                printf("oo");
            }
            else if (*(image + i * 28 + j) > 0.6) {
                printf("vv");
            }
            else if (*(image + i * 28 + j) > 0.4) {
                printf(". ");
            }
            else {
                printf("  ");
            }
        }
        printf("\n");
    }
    printf("\n_________\n");
}

void load_data(const char* imagepath, const char* labelpath, double** total_features, double** total_targets) {
    FILE* train_image_file = fopen(imagepath, "rb");
    FILE* train_label_file = fopen(labelpath, "rb");
    if (train_image_file == NULL || train_label_file == NULL) {
        fprintf(stderr, "파일을 열 수 없습니다.\n");
        exit(1);
    }
    // 헤더 정보 스킵
    fseek(train_image_file, 16, SEEK_SET);
    fseek(train_label_file, 8, SEEK_SET);
    unsigned char* image_buffer = NULL;
    double* image_values = NULL;
    image_buffer = (unsigned char*)malloc(INPUT_SIZE * sizeof(unsigned char));
    image_values = (double*)malloc(INPUT_SIZE * sizeof(double));
    if (image_buffer == NULL || image_values == NULL) {
        printf("할당 실패!");
    }
    for (int i = 0; i < TOTAL_DATA_SIZE; i++) {
        fread(image_buffer, sizeof(unsigned char), INPUT_SIZE, train_image_file);
        for (int j = 0; j < INPUT_SIZE; j++) {
            *(image_values + j) = (int)(*(image_buffer + j)) / 255.0;
        }
        for (int j = 0; j < INPUT_SIZE; j++) {
            *(*(total_features + i) + j) = *(image_values + j);
        }

        unsigned char label_buffer;
        fread(&label_buffer, sizeof(unsigned char), 1, train_label_file);
        int label = (int)label_buffer;
        one_hot_encode(label, *(total_targets + i));
    }
    free(image_buffer);
    free(image_values);
}


void prepare_data(const char* imagepath, const char* labelpath,
    double*** total_features, double*** total_targets,
    double*** train_features, double*** train_targets,
    double*** test_features, double*** test_targets) {

    *total_features = (double**)malloc(sizeof(double*) * TOTAL_DATA_SIZE);
    *total_targets = (double**)malloc(sizeof(double*) * TOTAL_DATA_SIZE);

    *train_features = (double**)malloc(sizeof(double*) * TRAIN_DATA_SIZE);
    *train_targets = (double**)malloc(sizeof(double*) * TRAIN_DATA_SIZE);

    *test_features = (double**)malloc(sizeof(double*) * TEST_DATA_SIZE);
    *test_targets = (double**)malloc(sizeof(double*) * TEST_DATA_SIZE);
    if (*total_features == NULL || *total_targets == NULL ||
        *train_features == NULL || *train_targets == NULL ||
        *test_features == NULL || *test_targets == NULL) {
        fprintf(stderr, "dataset 메모리 할당 실패\n");
        exit(1);
    }
    for (int i = 0; i < TOTAL_DATA_SIZE; i++) {
        *(*total_features + i) = (double*)malloc(sizeof(double) * INPUT_SIZE);
        *(*total_targets + i) = (double*)malloc(sizeof(double) * OUTPUT_SIZE);
    }
    for (int i = 0; i < TRAIN_DATA_SIZE; i++) {
        *(*train_features + i) = (double*)malloc(sizeof(double) * INPUT_SIZE);
        *(*train_targets + i) = (double*)malloc(sizeof(double) * OUTPUT_SIZE);
    }
    for (int i = 0; i < TEST_DATA_SIZE; i++) {
        *(*test_features + i) = (double*)malloc(sizeof(double) * INPUT_SIZE);
        *(*test_targets + i) = (double*)malloc(sizeof(double) * OUTPUT_SIZE);
    }

    load_data(imagepath, labelpath, *total_features, *total_targets);

    mix(*total_features, *total_targets);

    split_train_test(*total_features, *total_targets, *train_features, *train_targets, *test_features, *test_targets);
}

void freeConvLayer(ConvLayer* layer) {
    if (layer->filters != NULL) {
        for (int i = 0; i < layer->output_depth; i++) {
            for (int j = 0; j < layer->input_depth; j++) {
                for (int k = 0; k < layer->filter_size; k++) {
                    free(*(*(*(layer->filters + i) + j) + k));
                }
                free(*(*(layer->filters + i) + j));
            }
            free(*(layer->filters + i));
        }
        free(layer->filters);
    }

    if (layer->biases != NULL) {
        free(layer->biases);
    }

    if (layer->outputs != NULL) {
        free(layer->outputs);
    }

    if (layer->inputs != NULL) {
        free(layer->inputs);
    }
}

void freePoolingLayer(PoolingLayer* layer) {
    if (layer->outputs != NULL) {
        free(layer->outputs);
    }

    if (layer->inputs != NULL) {
        free(layer->inputs);
    }
}


void free_layer(Layer* layer) {
    for (int i = layer->input_nodes - 1; i > 0; i--) {
        free(*(layer->weights + i));
        free(*(layer->delta_weights + i));
    }
    free(layer->g);
    free(layer->inputs);
    free(layer->biases);
    free(layer->outputs);
    layer->activation_function = NULL;
    layer->activation_function_d = NULL;
}
void freeFlattenLayer(FlattenLayer* layer) {
    if (layer->outputs != NULL) {
        free(layer->outputs);
    }
}
void free_network(Network* network) {
    for (int i = LAYER_NUM - 1; i >= 0; i--) {
        free_layer(network->layers + i);
    }
    free(network->layers);
}

