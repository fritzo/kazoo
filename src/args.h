#ifndef KAZOO_ARGS_H
#define KAZOO_ARGS_H

#include <stdint.h>
#include <iostream>
#include <string>
#include <vector>
#include <map>

struct Args
{
  int argc;
  char ** argv;
  const char * help;

  class Switch
  {
  public:

    typedef void (* Function)(Args &);
    class Action
    {
      enum Type { NONE, CALL, INT, FLOAT, SIZE_T, STRING, BOOL };
      Type m_type;

      // TODO a union would take less space
      Function m_fun;
      int * m_int;
      float * m_float;
      size_t * m_size_t;
      std::string * m_string;
      bool * m_bool;

    public:

      Action ()           : m_type(NONE) {}
      Action (Function f) : m_type(CALL), m_fun(f) {}
      Action (int & i)    : m_type(INT), m_int(& i) {}
      Action (float & f)  : m_type(FLOAT), m_float(& f) {}
      Action (size_t & s) : m_type(SIZE_T), m_size_t(& s) {}
      Action (std::string & s) : m_type(STRING), m_string(& s) {}
      Action (bool & b)   : m_type(BOOL), m_bool(& b) {}

      void operator() (Args & args)
      {
        switch (m_type) {
          case NONE: break;
          case CALL: m_fun(args); break;
          case INT: * m_int = atoi(args.pop()); break;
          case FLOAT: * m_float = atof(args.pop()); break;
          case SIZE_T: * m_size_t = atoi(args.pop()); break;
          case STRING: * m_string = args.pop(); break;
          case BOOL: * m_bool = atoi(args.pop()); break;
        }
      }
    };

  private:

    typedef std::map<std::string, Action> Cases;
    typedef Cases::iterator Case;

    Args & m_args;
    Cases m_cases;
    void print_error ();

  public:

    Switch (Args & args);
    ~Switch () {}

    Switch & case_ (std::string key, Action action);
    void default_keep (Action default_action);
    void default_ (Action default_action);
    void default_error ();
    void default_break_else_repeat ();
    void default_all ();
  };

  const char * top () { return * argv; }

public:

  Args (int c, char ** v, const char * h) : argc(c-1), argv(v+1), help(h) {}
  ~Args ();

  size_t size () const { return argc; }

  const char * pop ()
  {
    if (not argc--) {
      std::cout << help << std::endl;
      std::cout << "ERROR too few arguments" << std::endl;
      exit(1);
    }
    return *(argv++);
  }

  const char * pop (const char * default_value)
  {
    if (not argc) return default_value;
    --argc;
    return *(argv++);
  }
  int pop (int default_value)
  {
    if (not argc) return default_value;
    --argc;
    return atoi(*(argv++));
  }
  float pop (float default_value)
  {
    if (not argc) return default_value;
    --argc;
    return atof(*(argv++));
  }

  std::vector<std::string> pop_all ()
  {
    std::vector<std::string> result;
    while (argc) result.push_back(pop());
    return result;
  }
  std::vector<std::string> pop_all (std::string default_value)
  {
    return argc ? pop_all() : std::vector<std::string>(1, default_value);
  }

  void done ()
  {
    if (argc) {
      std::cout << help << std::endl;
      std::cout << "WARNING too many arguments" << std::endl;
    }
  }

  Switch case_ (std::string key, Switch::Action action);
};

inline std::ostream & operator<< (std::ostream & o, const Args & args)
{
  for (int i = 0; i < args.argc; ++i) {
    o << "\n  " << args.argv[i];
  }
  return o << std::flush;
}

#endif // KAZOO_ARGS_H
