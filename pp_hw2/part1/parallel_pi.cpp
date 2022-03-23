#include <iostream>
#include <pthread.h>
#include <random>

using namespace std;

typedef struct thread_data {
    long long tosses, result;
} thread_data;

void *thread_pi(void *data) {
    random_device rd;
    default_random_engine engine(rd());
    uniform_real_distribution<float> ran(0.0, 1.0);

    thread_data *d = (thread_data*) data;
    int cnt = 0;
    for(int i = 0; i < d->tosses; i++) {
        double x = ran(engine), y = ran(engine);
        double distance = x * x + y * y;
        if(distance <= 1)
            cnt++;
    }
    d->result = cnt;
    return data;
}

int main(int argc, char* argv[]) {
    if(argc != 3) {
        cerr << "Execution with command \"./pi.out <number of threads> <number of tosses>\"\n";
        return -1;
    }
    long long num_threads, num_tosses, tosses_each;
    int rem;
    num_threads = atoi(argv[1]);
    num_tosses = atoi(argv[2]);
    tosses_each = num_tosses / num_threads;
    rem = num_tosses % num_threads;

    pthread_t *ids = (pthread_t*) malloc(sizeof(pthread_t) * num_threads);
    thread_data *datas = (thread_data*) malloc(sizeof(thread_data) * num_threads);

    for(int i = 0; i < num_threads; i++) {
        datas[i].tosses = tosses_each + (i < rem);
        pthread_create(&ids[i], NULL, thread_pi, (void*) &datas[i]);
    }
    for(int i = 0; i < num_threads; i++)
        pthread_join(ids[i], NULL);

    int res = 0;
    for(int i = 0; i < num_threads; i++)
        res += datas[i].result;
    
    float pi = 4 * (double) res / (double) num_tosses;

    cout << pi << endl;
    return 0;
}