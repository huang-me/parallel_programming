CXX := clang++

TARGET := serial

OBJ := serial.o

CXXFLAGS := -O3 -Wall

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) $(OBJ) -o $@

clean:
	rm $(TARGET) $(OBJ)