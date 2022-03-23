#include <iostream>

#define ran \
    -1 + static_cast<float> (rand()) / static_cast<float> (RAND_MAX / 2)

using namespace std;

int main(int argc, char* argv[]) {
    if(argc != 2) {
        cerr << "Execute with command \"./serial <number of tosses>\"\n";
        return -1;
    }
    int number_in_circle = 0;
    int number_of_tosses = atoi(argv[1]);
    for (int toss = 0; toss < number_of_tosses; toss++) {
        double x = ran, y = ran;
        double distance_squared = x * x + y * y;
        if ( distance_squared <= 1)
            number_in_circle++;
    }
    float pi_estimate = 4 * number_in_circle /(( double ) number_of_tosses);
    cout << pi_estimate << endl;
    return 0;
}