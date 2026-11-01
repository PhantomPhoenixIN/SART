class Flow {
  static int total(int[] values) {
    int result = 0;
    for (int value : values) {
      if (value > 0) result += value;
    }
    return result;
  }
}
