
#include "common.h"
#include "args.h"
#include <lemon/network_simplex.h>

const char * help_message =
"Usage: lemon_test"
;

int main (int argc, char ** argv)
{
  Args args(argc, argv, help_message);

  // right now, we only test compilation

  return 0;
}

