#ifndef TRANDOM_DOUBLE_H
#define TRANDOM_DOUBLE_H

#include <random>
#include <ctime>
#include <WriteToLog.h>

////Инициализация генератора случайных чисел с учетом одновременности запуска нескольких процессов
////Здесь для случайных чисел в контексте вычислительного потока
//unsigned long seed = time(0) + (
//+ThreadNumber*18181 + ThreadNumber
//#ifndef _NOT_USING_MPI
//+ParentDetectorSimulator->ProcRank*17471 + ParentDetectorSimulator->ProcRank
//#endif
//)%(16661);
//
//srand(seed);
//TRandomDouble * rd = &(TRandomDouble::Instance());
//rd->Initialize(seed);

///Синглтон - генератор случайных даблов
class TRandomDouble {
private:
  std::mt19937_64 RE;
  std::uniform_real<double> * URD;

  TRandomDouble() {URD = 0;}                           // Private constructor
  ~TRandomDouble() {}
  TRandomDouble(const TRandomDouble&);                 // Prevent copy-construction
  TRandomDouble& operator=(const TRandomDouble&);      // Prevent assignment
public:
  static TRandomDouble& Instance()
  {
    static TRandomDouble random_double;
    return random_double;
  }

  void Initialize(unsigned long int s = 0)
  {
		if (s != 0) RE.seed(s); else RE.seed(time(0));
		if (URD == 0) URD = new std::uniform_real<double>(0., 1.);
  };

  double operator () () {
    if (URD == 0) Initialize();
    return (*URD)(RE);
  }
};

inline double RandomDouble()
{
  TRandomDouble * rd = &(TRandomDouble::Instance());
  return (*rd)();
};

#endif