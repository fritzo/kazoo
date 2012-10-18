
#ifndef KAZOO_CONFIG_H
#define KAZOO_CONFIG_H

#include "common.h"
#include "hash_map.h"
#include <fstream>

// TODO switch to json format: http://www.json.org/

class ConfigParser
{
  typedef std::hash_map<string, string> Dict;
  typedef Dict::const_iterator Auto;
  Dict m_dict;
  const string m_filename;

public:

  ConfigParser (const char * filename)
    : m_filename(filename)
  {
    std::ifstream file(filename);
    ASSERT(file, "failed to open config file " << filename);

    string comment, key, equals, value;
    while (file) {
      int peek = file.peek();
      if (isspace(peek)) {
        file.get();
      } else if (peek == '#') {
        std::getline(file, comment);
      } else {
        file >> key >> equals >> value;
        ASSERT_EQ(equals, "=");
        m_dict[key] = value;
      }
    }
  }

  string operator() (string key, string default_value) const
  {
    Auto i = m_dict.find(key);
    const string & value = (i == m_dict.end()) ? default_value : i->second;
    if (value != default_value) {
      LOG(" " << m_filename << ": " << key << " = " << value
          << " (default = " << default_value << ")");
    }
    return value;
  }

  int operator() (string key, int default_value) const
  {
    Auto i = m_dict.find(key);
    int value = (i == m_dict.end()) ? default_value : atoi(i->second.c_str());
    if (value != default_value) {
      LOG(" " << m_filename << ": " << key << " = " << value
          << " (default = " << default_value << ")");
    }
    return value;
  }

  float operator() (string key, float default_value) const
  {
    Auto i = m_dict.find(key);
    float value = (i == m_dict.end()) ? default_value : atof(i->second.c_str());
    if (value != default_value) {
      LOG(" " << m_filename << ": " << key << " = " << value
          << " (default = " << default_value << ")");
    }
    return value;
  }
};

#endif // KAZOO_CONFIG_H

