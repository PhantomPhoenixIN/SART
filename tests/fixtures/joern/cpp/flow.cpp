int total(const int* values, int size) {
  int result = 0;
  for (int i = 0; i < size; ++i) {
    if (values[i] > 0) result += values[i];
  }
  return result;
}
