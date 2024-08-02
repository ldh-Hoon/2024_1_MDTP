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

#define HIDDEN_LAYER_NUM 3
#define LAYER_DATA 256, 128, 32 //���� ���̾� ��� ����

#define LAYER_NUM HIDDEN_LAYER_NUM+1 //����� ���� ���̾� ����

// dataset ����
#define TRAIN_DATA_PATH "../training"

#define DATA_ADD 1 // 1�̸� ������ ���� 6����(�������� +- 1 �����̵�, +- 10�� ȸ��)
#define TRAIN_DATA_SIZE (60000 + DATA_ADD*60000) // �⺻ 6��, ������ ���� �� 12��

// train ���� 
#define EPOCH 30 // ��ü ������ �ݺ� Ƚ��
#define BATCH_SIZE 16 // batch size
#define LR 0.04 //learning rate
double linear_lr = 0.01;

#define DROPOUT_RATE 0.1 //dropout 

#define MOMENTUM 0.9 //update ���ӵ�
#define NAG 1 //1�̸� NAG, 0�̸� �����

#define RELU 0 // 1�̸� ReLU ���
#define SOFTMAX RELU //����� softmax ���� ����

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

//�Ű�� ����ü
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

// ���Ժ��� ���� ���� �Լ�

double normal_distribution(double mean, double std);

// �Ű�� �ʱ�ȭ �Լ���

void initialize_layer(Layer* layer, int input_nodes, int output_nodes, int id, double dropout_rate, void (*activation_function)(double*, double*, int), void (*activation_function_d)(double*, double*, int));
void initialize_network(Network* network, int layer_num, int* num, double dropout_rate,
    void (*activation_function)(double*, double*, int), void (*activation_function_d)(double*, double*, int));

// ������ �Լ���

void forward_layer(Layer* layer, double* input, bool is_eval);
void forward(Network* network, double* input);

// ������ �Լ���

void backward_layer(Layer* layer, double* next_g);
void backward(Network* network, double* next_g);


// �����ͼ� ����

int data_count(const char* baseFolder);
void prepare_data(const char* data_path,
    int data_size, double*** features, double*** targets);
void load_data(const char* baseFolder, double** features, double** targets, int max);
void load_image(const char* imagePath, double* features, double* features2);
void data_augmentation(unsigned char* input, unsigned char* output);

// ������ ��ó��

void mix(double** features, double** target, int size);

void one_hot_encode(int label, double* output);

// �н�

void train(Network* network, double** train_features, double** train_targets);

void save_model(Network* network);

// �޸� ����

void free_layer(Layer* layer);
void free_network(Network* network);

void memory_clear(
    double** features, double** targets,
    int size);

int main() {
    // �н� ��

    // �Ű�� �ʱ�ȭ
    Network network;
    int layer_data[LAYER_NUM + 1] = { INPUT_SIZE, LAYER_DATA, OUTPUT_SIZE }; //�Է���, ����� ���� ��� ����
    //�ʱ�ȭ �� dropout, Ȱ���Լ� ����
    if (RELU == 0) {
        initialize_network(&network, LAYER_NUM, layer_data, DROPOUT_RATE, sigmoid, sigmoid_d);
    }
    else if (RELU == 1) {
        initialize_network(&network, LAYER_NUM, layer_data, DROPOUT_RATE, ReLU, ReLU_d);
    }

    // ������ �غ�
    int train_data_size = data_count(TRAIN_DATA_PATH);

    double** train_features = NULL;
    double** train_targets = NULL;

    prepare_data(TRAIN_DATA_PATH, train_data_size, &train_features, &train_targets);

    //�н�
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
        *(output + i) = exp(*(input + i) - max);//�����÷ο� ����
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

/*���Ժ��� ���� ���� �Լ�*/
double normal_distribution(double mean, double std) {
    double U1 = (double)rand() / RAND_MAX;
    double U2 = (double)rand() / RAND_MAX;
    return mean + std * sqrt(-2 * log(U1)) * cos(2 * PI * U2);
}

/*�� ���̾� �ʱ�ȭ �Լ�*/
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
        fprintf(stderr, "layer->weights �޸� �Ҵ� ����\n");
        exit(1);
    }
    if (layer->delta_weights == NULL) {
        fprintf(stderr, "layer->delta_weights �޸� �Ҵ� ����\n");
        exit(1);
    }
    if (layer->batch_weights == NULL) {
        fprintf(stderr, "layer->batch_weights �޸� �Ҵ� ����\n");
        exit(1);
    }
    double a = sqrt(2.0 / (layer->input_nodes + layer->output_nodes)); //Xavier �ʱ�ȭ
    for (int i = 0; i < input_nodes; i++) {
        for (int j = 0; j < output_nodes; j++) {
            *(layer->weights + i * output_nodes + j) = normal_distribution(0, a);  // ǥ�������� a�� ���Ժ��� ��
            if (*(layer->weights + i * output_nodes + j) < -99) {
                *(layer->weights + i * output_nodes + j) = 0;
            }
            if (*(layer->weights + i * output_nodes + j) > 99) {
                *(layer->weights + i * output_nodes + j) = 0;
            }
            // ���Ժ��� �̻�ġ ó��(log�� ���� ���Ѵ� ��)
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
        fprintf(stderr, "layer->g �޸� �Ҵ� ����\n");
        exit(1);
    }
    if (layer->inputs == NULL) {
        fprintf(stderr, "layer->inputs �޸� �Ҵ� ����\n");
        exit(1);
    }
    if (layer->biases == NULL) {
        fprintf(stderr, "layer->biases �޸� �Ҵ� ����\n");
        exit(1);
    }
    if (layer->batch_biases == NULL) {
        fprintf(stderr, "layer->batch_biases �޸� �Ҵ� ����\n");
        exit(1);
    }
    if (layer->delta_biases == NULL) {
        fprintf(stderr, "layer->delta_biases �޸� �Ҵ� ����\n");
        exit(1);
    }
    if (layer->outputs == NULL) {
        fprintf(stderr, "layer->outputs �޸� �Ҵ� ����\n");
        exit(1);
    }
    for (int i = 0; i < input_nodes; i++) {
        *(layer->g + i) = 0.0;
        *(layer->inputs + i) = 0.0;
    }
    for (int i = 0; i < output_nodes; i++) {
        *(layer->biases + i) = normal_distribution(0, a);  // ǥ�������� a�� ���Ժ��� ��
        if (*(layer->biases + i) < -99) {
            *(layer->biases + i) = 0;
        }
        if (*(layer->biases + i) > 99) {
            *(layer->biases + i) = 0;
        }
        // ���Ժ��� �̻�ġ ó��
        *(layer->outputs + i) = 0.0;
        *(layer->batch_biases + i) = 0.0;
        *(layer->delta_biases + i) = 0.0;
    }
}
/*�Ű�� �ʱ�ȭ �Լ�*/
void initialize_network(Network* network, int layer_num, int* num, double dropout_rate,
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
        initialize_layer((network->layers + j), *(num + j), *(num + j + 1), j, dropout_rate, activation_function, activation_function_d);
    }
    printf("\n\n");
    network->eval = false;
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

    if ((!is_eval) && (layer->dropout_rate > 0.0)) {
        for (int i = 0; i < layer->input_nodes; i++) {
            if ((double)rand() / RAND_MAX < layer->dropout_rate) {
                *(layer->inputs + i) = 0.0; // ��Ӿƿ� ������ŭ ��带 ��Ȱ��ȭ
            }
        }
    }

    // weight�� input�� ���ϰ�, bias�� ���� ��, Ȱ��ȭ �Լ� ����
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

/*�� ���̾��� �����ĸ� �����ϴ� �Լ�*/
void update_layer(Layer* layer) {
    int i, j;
    // ��ġ ����ġ ������Ʈ
    for (i = 0; i < layer->input_nodes; i++) {
        for (j = 0; j < layer->output_nodes; j++) {
            *(layer->batch_weights + i * layer->output_nodes + j) = *(layer->batch_weights + i * layer->output_nodes + j) / BATCH_SIZE;
            *(layer->delta_weights + i * layer->output_nodes + j) = MOMENTUM * *(layer->delta_weights + i * layer->output_nodes + j)
                + *(layer->batch_weights + i * layer->output_nodes + j);
            *(layer->weights + i * layer->output_nodes + j) += *(layer->delta_weights + i * layer->output_nodes + j);
            *(layer->batch_weights + i * layer->output_nodes + j) = 0.0;
        }
    }

    // ���̾ ������Ʈ
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
    // ������Ʈ ���
    for (i = 0; i < layer->input_nodes; i++) {
        for (j = 0; j < layer->output_nodes; j++) {
            *(layer->batch_weights + i * layer->output_nodes + j) -= (linear_lr * LR) * (*(next_g + j) * *(layer->inputs + i));
        }
    }
    // ���̾ ������Ʈ ���
    for (j = 0; j < layer->output_nodes; j++) {
        *(layer->batch_biases + j) -= (linear_lr * LR) * (*(next_g + j));
    }
    layer->activation_function_d(layer->inputs, layer->inputs, layer->input_nodes);

    // �׷����Ʈ ��� �� ���� ���̾�� ����
    for (i = 0; i < layer->input_nodes; i++) {
        *(layer->g + i) = 0.0;
        for (j = 0; j < layer->output_nodes; j++) {
            *(layer->g + i) += (*(layer->weights + i * layer->output_nodes + j) + NAG * (MOMENTUM * *(layer->delta_weights + i * layer->output_nodes + j))) * *(next_g + j);
        }
        *(layer->g + i) *= *(layer->inputs + i);
    }
}

/*������ �Լ�*/
void backward(Network* network, double* next_g) {
    double* layer_d_output = next_g;

    for (int i = network->layer_num - 1; i >= 0; i--) {
        backward_layer(network->layers + i, layer_d_output);

        layer_d_output = (*(network->layers + i)).g;
    }
}

/*dataset�� ���� �Լ�*/
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

/*�Ű�� ���� �Լ�, ���緹�̾��.bin ����*/
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
    printf("���� �Ϸ�\n");
}

/*���Ǵ�� �н��� �����ϴ� �Լ�*/
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
    printf("�н� �ҿ�ð�: %lf\n", (double)(end - start));
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

/*������ ���� �Լ�*/
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
            // ������ ���
            int x = (int)(cos_theta * (j - centerX) + sin_theta * (i - centerY)) + centerX + px;
            int y = (int)(-sin_theta * (j - centerX) + cos_theta * (i - centerY)) + centerY + py;

            if (x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT) {
                *(output + i * WIDTH + j) = *(input + y * WIDTH + x);
            }
            else {
                *(output + i * WIDTH + j) = 0; // ������ ����� ����������
            }
        }
    }
}

/*�̹����� �о���� �Լ�*/
void load_image(const char* imagePath, double* features, double* features2) {
    FILE* file = fopen(imagePath, "rb");
    if (file == NULL) {
        printf("%s ���� ����!", imagePath);
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
            printf("���� %d : %d��\n", folder, count);
        }
    }
    if (DATA_ADD) {
        printf("�� ���� %d��, ������ ���� %d��\n", index, index);
    }
    else {
        printf("�� ���� %d��\n", index);
    }
}

/*dataset �غ� �� ��ó�� �Լ�*/
void prepare_data(const char* data_path,
    int data_size, double*** features, double*** targets) {

    time_t start = time(NULL);
    printf("______data load/augmentation_____\n");

    *features = (double**)malloc(sizeof(double*) * TRAIN_DATA_SIZE);
    *targets = (double**)malloc(sizeof(double*) * TRAIN_DATA_SIZE);
    if (*features == NULL || *targets == NULL) {
        printf("dataset �޸� �Ҵ� ����");
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