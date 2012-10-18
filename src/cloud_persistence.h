#ifndef KAZOO_CLOUD_PERSISTENCE_H
#define KAZOO_CLOUD_PERSISTENCE_H

#include "common.h"
#include "eigen.h"

namespace Cloud
{

//----( persistence )---------------------------------------------------------

// TODO switch to json format

struct Persistent
{
  // TODO use relative paths for filestems.
  //   this entails passing around a directory in the read,write methods
  mutable string filestem;

  Persistent () : filestem("default") {}
  virtual ~Persistent () {}

  void save (string filename) const;
  void load (string filename);

  virtual void write (ostream & o) const = 0;
  virtual void read (istream & file) = 0;
};

void write_matrix (const MatrixSf & A, ostream & o);
void read_matrix (MatrixSf & A, istream & file);

} // namespace Cloud

#endif // KAZOO_CLOUD_PERSISTENCE_H
