#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <windows.h>
#include <io.h>

// model ����
#define INPUT_SIZE 784 // train data �Է� ũ��
#define OUTPUT_SIZE 10 // train data ��� ũ��

#define HIDDEN_LAYER_NUM 3 // �ε� �� �Ű���� hidden layer ����

#define MODEL_BASE_PATH "../Project_Train" // �� ���� ����(���ϸ��� ���緹�̾��.bin)

// dataset ����
#define TEST_DATA_PATH "../testing"

/* 
���� 0 : 980��
���� 1 : 1135��
���� 2 : 1032��
���� 3 : 1010��
���� 4 : 982��
���� 5 : 892��
���� 6 : 958��
���� 7 : 1028��
���� 8 : 974��
���� 9 : 1009��
�� ���� 10000��
*/
#define TEST_DATA_SIZE 10000

// ��Ÿ 
#define PI 3.141592
#define Sigmoid(x) 1.0 / (1.0 + exp(-x))

// ���̾� ����ü
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

//�Ű�� ����ü
typedef struct {
    Layer* layers;
    int layer_num;
    bool eval;
} Network;

void sigmoid(double* input, double* output, int length);
void sigmoid_d(double* input, double* output, int length);

// �Ű�� �ʱ�ȭ �Լ���

void initialize_layer(Layer* layer, int input_nodes, int output_nodes, int id, void (*activation_function)(double*, double*, int), void (*activation_function_d)(double*, double*, int));
void initialize_network(Network* network, int layer_num, int* num,
    void (*activation_function)(double*, double*, int), void (*activation_function_d)(double*, double*, int));

// �Ű�� �ε�

void load_model(Network* network, int num);

// ������ �Լ���

void forward_layer(Layer* layer, double* input, bool is_eval);
void forward(Network* network, double* input);

// ���� �Լ�
void eval(Network* network, double** features, double** targets, int size);

// �����ͼ� ����

int data_count(const char* baseFolder);
void prepare_data(const char* data_path,
    int data_size, double*** features, double*** targets);
void load_data(const char* baseFolder, double** features, double** targets, int max);
void load_image(const char* imagePath, double* features);

// ������ ��ó��

void one_hot_encode(int label, double* output);

// �޸� ����

void free_layer(Layer* layer);
void free_network(Network* network);

void memory_clear(
    double** features, double** targets,
    int size);

int main() {
    // ������ �غ�
    int test_data_size = data_count(TEST_DATA_PATH);

    double** test_feature = NULL;
    double** test_targets = NULL;

    prepare_data(TEST_DATA_PATH, test_data_size, &test_feature, &test_targets);

    Network network;

    // �Ű�� �ҷ�����
    load_model(&network, HIDDEN_LAYER_NUM);

    //����
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


/*�� ���̾� �ʱ�ȭ �Լ�*/
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
        printf("layer->weights �޸� �Ҵ� ����\n");
        exit(1);
    }

    if (layer->batch_weights == NULL) {
        printf("layer->batch_weights �޸� �Ҵ� ����\n");
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
        printf("layer->g �޸� �Ҵ� ����\n");
        exit(1);
    }
    if (layer->inputs == NULL) {
        printf("layer->inputs �޸� �Ҵ� ����\n");
        exit(1);
    }
    if (layer->biases == NULL) {
        printf("layer->biases �޸� �Ҵ� ����\n");
        exit(1);
    }
    if (layer->batch_biases == NULL) {
        printf("layer->batch_biases �޸� �Ҵ� ����\n");
        exit(1);
    }
    if (layer->outputs == NULL) {
        printf("layer->outputs �޸� �Ҵ� ����\n");
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

/*�Ű�� �ʱ�ȭ �Լ�*/
void initialize_network(Network* network, int layer_num, int* num,
    void (*activation_function)(double*, double*, int), void (*activation_function_d)(double*, double*, int)) {
    printf("_____init_____\n");
    printf("hidden layer ���� : %d\n", layer_num - 1);
    network->layer_num = layer_num;
    network->layers = (Layer*)malloc(sizeof(Layer) * layer_num);
    if (network->layers == NULL) {
        printf("network->layers �޸� �Ҵ� ����\n");
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

/*�Ű�� �ε�*/
void load_model(Network* network, int num) {
    char load_path[1024];
    sprintf(load_path, "%s/%d.bin", MODEL_BASE_PATH, num);
    printf("______model load______\n");
    printf("%s �� �ҷ��ɴϴ�.\n", load_path);
    FILE* file = fopen(load_path, "rb");
    if (file == NULL) {
        fprintf(stderr, "%s ������ �� �� �����ϴ�.\n", load_path);
        exit(1);
    }
    int layer_num = 0;
    int* layer_data = NULL;
    fread(&layer_num, sizeof(int), 1, file);
    printf("���� ���̾� : %d��\n", layer_num - 1);
    layer_data = malloc(sizeof(int) * (layer_num + 1));
    fread(layer_data, sizeof(int), layer_num + 1, file);
    for (int i = 1; i < layer_num; i++) {
        printf("%dth layer : %d��\n", i, *(layer_data + i));
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
    printf("�ε� �Ϸ�!\n\n");
}

/*������ �Լ�*/
void forward(Network* network, double* input) {
    double* current_input = input;
    // �� ���̾ ��ȸ�ϸ� forward_layer ȣ��
    for (int i = 0; i < network->layer_num; i++) {
        forward_layer((network->layers + i), current_input, network->eval);
        current_input = (*(network->layers + i)).outputs;
    }
}

/*�� ���̾��� �����ĸ� ����ϴ� �Լ�*/
void forward_layer(Layer* layer, double* input, bool is_eval) {
    for (int i = 0; i < layer->input_nodes; i++) {
        *(layer->inputs + i) = *(input + i);
    }

    // weight�� input�� ���ϰ�, bias�� ���� ��, Ȱ��ȭ �Լ� ����
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

/*���� ���� �Լ�*/
void eval(Network* network, double** features, double** targets, int size) {
    network->eval = true;
    printf("\n______eval_____\n");

    double accuracy = 0.0;
    int count[OUTPUT_SIZE] = { 0, };
    double accuracy_arr[OUTPUT_SIZE] = { 0,  };

    double* last_output = (*(network->layers + network->layer_num - 1)).outputs;

    for (int i = 0; i < size; i++) {
        forward(network, *(features + i)); // ������

        int true_label = 0, predicted_label = 0;
        double max_true_val = **(targets + i), max_pred_val = *(last_output + 0);

        // ���� label�� ���� label
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

    // ��� ���
    printf("Mean Accuracy: %.2f%\n\n", accuracy * 100);
    network->eval = false;
}

/*one-hot encoding*/
void one_hot_encode(int label, double* output) {
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        *(output + i) = (i == label) ? 1.0 : 0.0;
    }
}


/*������ ������ �̹��� ������ ���� �Լ�*/
int data_count(const char* baseFolder) {
    int count = 0;
    struct _finddata_t fileinfo;
    intptr_t hFile;
    char path[1024];

    for (int folder = 0; folder <= 9; folder++) {
        sprintf(path, "%s\\%d\\*.raw", baseFolder, folder);

        if ((hFile = _findfirst(path, &fileinfo)) == -1L) {
            printf("���� ���� �Ұ�: '%s'\n", path);
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

/*�̹����� �о���� �Լ�*/
void load_image(const char* imagePath, double* features) {
    FILE* file = fopen(imagePath, "rb");
    if (file == NULL) {
        printf("%s ���� ����!", imagePath);
        return;
    }
    unsigned char image[INPUT_SIZE];

    fread(image, sizeof(unsigned char), INPUT_SIZE, file);
    fclose(file);

    for (int i = 0; i < INPUT_SIZE; i++) {
        *(features + i) = (double)(*(image + i)) / 255.0;
    }
}

/*�����ͼ��� �ε��ϴ� �Լ�*/
void load_data(const char* baseFolder, double** features, double** targets, int max) {
    struct _finddata_t fileinfo;
    intptr_t hFile;
    char path[1024];
    int index = 0;

    printf("�̹��� �ε� ��...\n");
    for (int folder = 0; folder <= 9; folder++) {
        int count = 0;
        sprintf(path, "%s\\%d\\*.raw", baseFolder, folder);
        // path ��ġ�� ��� raw Ȯ���� ���� ����� ��ȯ
        if ((hFile = _findfirst(path, &fileinfo)) == -1L) {
            printf("���� ���� �Ұ�: '%s'\n", path);
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
            printf("���� %d : %d��\n", folder, count);
        }
    }
    printf("�� ���� %d��\n", index);
}

/*dataset �غ� �� ��ó�� �Լ�*/
void prepare_data(const char* data_path,
    int data_size, double*** features, double*** targets) {

    time_t start = time(NULL);
    printf("______data load_____\n");

    *features = (double**)malloc(sizeof(double*) * data_size);
    *targets = (double**)malloc(sizeof(double*) * data_size);
    if (*features == NULL || *targets == NULL) {
        printf("dataset �޸� �Ҵ� ����");
        exit(1);
        return;
    }
    for (int i = 0; i < data_size; i++) {
        *(*features + i) = (double*)malloc(sizeof(double) * INPUT_SIZE);
        *(*targets + i) = (double*)malloc(sizeof(double) * OUTPUT_SIZE);
    }

    load_data(data_path, *features, *targets, data_size);

    time_t end = time(NULL);
    printf("�����ͼ� �ε� �� ��ó�� �ҿ�ð�: %lf\n\n", (double)(end - start));
}

/*dataset ���� �޸� ����*/
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

/*�� ���̾� �޸𸮸� �����ϴ� �Լ�*/
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

/*�Ű�� �޸𸮸� �����ϴ� �Լ�*/
void free_network(Network* network) {
    for (int i = network->layer_num - 1; i >= 0; i--) {
        free_layer(network->layers + i);
    }
    free(network->layers);
}