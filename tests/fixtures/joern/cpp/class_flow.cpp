#include <vector>

class Accumulator {
 public:
  int positive_sum(const std::vector<int>& values) {
    int result = 0;
    for (int value : values) {
      if (value > 0) result += value;
    }
    return result;
  }
};
