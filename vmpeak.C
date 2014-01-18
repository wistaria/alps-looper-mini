#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

// Linux
#include <sys/types.h>
#include <unistd.h>
// 

// Linux only

int vmempeak()
{
  using std::string;

  string header = "VmPeak:\t";
  string VmPeak;
  int VmPeak_value;

  // process file name: /proc/${PID}/status
  int pid = static_cast<int>(getppid());
  std::stringstream iss;
  iss << pid;
  std::string filename = string("/proc/")
    + iss.str() + string("/status");

  // open file
  std::ifstream fin(filename.c_str());
  if (fin.fail()) {
    std::cerr << "Can't open the file" << std::endl;
    return false;
  }

  // read VmPeak value [kB]
  do {
    string source;
    getline(fin, source);
    if (source.substr(0,header.size()) == header) {
      for (string::size_type i = 0; i != source.size(); ++i) {
 	if (isdigit(source[i])) VmPeak += source[i];
      }
    }
  } while (fin.good());

  std::stringstream oss;
  oss << VmPeak;
  oss >> VmPeak_value;
  return  VmPeak_value;
}

int main()
{
  int vm = vmempeak();
  std::cout << "VmPeak = " << vm << " [kB]" << std::endl;
  return 0;
}
