#include <iostream>
#include <pthread.h>
#include <random>


#define ran \
    (2.0 - as_double(0x3FF0000000000000ULL | (rand64() >> 12)))

using namespace std;

typedef struct thread_data {
    long long tosses, result;
} thread_data;

static unsigned long x=123456789, y=362436069, z=521288629;

double as_double(uint64_t i)
{
    union
    {
        uint64_t i;
        double f;
    } pun = { i };
    return pun.f;
}

static unsigned long rand64(void) {
    unsigned long t;
    x ^= x << 16;
    x ^= x >> 5;
    x ^= x << 1;

    t = x;
    x = y;
    y = z;
    z = t ^ x ^ y;

    return z;
}

void *thread_pi(void *data) {

    thread_data *d = (thread_data*) data;
    long long cnt = 0;
    for(int i = 0; i < d->tosses; i++) {
        double x = ran, y = ran;
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
    num_threads = atoll(argv[1]);
    num_tosses = atoll(argv[2]);
    tosses_each = num_tosses / num_threads;
    rem = num_tosses % num_threads;

    pthread_t *ids = (pthread_t*) malloc(sizeof(pthread_t) * num_threads);
    thread_data *datas = (thread_data*) malloc(sizeof(thread_data) * num_threads);

    for(long long i = 0; i < num_threads; i++) {
        datas[i].tosses = tosses_each + (i < rem);
        pthread_create(&ids[i], NULL, thread_pi, (void*) &datas[i]);
    }
    for(int i = 0; i < num_threads; i++)
        pthread_join(ids[i], NULL);

    long long res = 0;
    for(long long i = 0; i < num_threads; i++)
        res += datas[i].result;
    
    float pi = 4 * (double) res / (double) num_tosses;

    printf("%.6f\n", pi);
    return 0;
}