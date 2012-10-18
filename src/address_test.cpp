
#include "address.h"
#include <cstdlib>
#include <unistd.h>

int main (int, char** argv)
{
  MacAddress address = get_mac_address();

#ifdef REQUIRED_MAC_ADDRESS
  MacAddress required(REQUIRED_MAC_ADDRESS);
  if (address != required) {
    unlink(argv[0]); // dirty underhanded trick
    exit(1);
  }
#else // REQUIRED_MAC_ADDRESS
  cout << "-DREQUIRED_MAC_ADDRESS=" << address << endl;
#endif // REQUIRED_MAC_ADDRESS

  return 0;
}

