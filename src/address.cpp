
#include "address.h"
#include <net/if.h> // for struct ifreq definition
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <linux/if_ether.h> //use SOCK_PACKET socket


MacAddress get_mac_address ()
{
  int fd;
  struct ifreq ifr;

  fd = socket(PF_INET, SOCK_PACKET, htons(ETH_P_ALL)); // open socket

  strcpy(ifr.ifr_name, "eth0"); // assuming we want eth0

  ioctl(fd, SIOCGIFHWADDR, &ifr); // retrieve MAC address

  MacAddress result;

  memcpy(result.data, ifr.ifr_hwaddr.sa_data, 6);

  close(fd);

  return result;
}

