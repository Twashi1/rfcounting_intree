int addmul(int a, int b) {
  if (a > b) {
    return a + b;
  }
  
  return a * b;
}

int main() {
  int res = addmul(5, 3);
  int x = addmul(res, res);

  return x;
}
