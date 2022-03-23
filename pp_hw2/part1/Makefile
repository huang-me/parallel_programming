CXX := clang++

TARGET := serial pi.out

OBJ := serial.o parallel_pi.o

CXXFLAGS := -O3 -Wall

all: $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

serial: %:%.o
	$(CXX) $(CXXFLAGS) $< -o $@

pi.out: parallel_pi.o
	$(CXX) $(CXXFLAGS) $< -o $@ -lpthread 

clean:
	rm -f $(TARGET) $(OBJ)