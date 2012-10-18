
#include "args.h"
#include <sstream>

using std::cout;
using std::endl;
using std::string;

Args::Switch::Switch (Args & args)
  : m_args(args),
    m_cases()
{}

Args::Switch & Args::Switch::case_ (
    string key,
    Args::Switch::Action action)
{
  m_cases[key] = action;
  return * this;
}

void Args::Switch::default_keep (Args::Switch::Action default_action)
{
  Action action = default_action;

  if (m_args.size()) {
    string arg = m_args.top();
    Case i = m_cases.find(arg);
    if (i != m_cases.end()) {
      action = i->second;
    }
  }

  action(m_args);
  //delete this;
}

void Args::Switch::default_ (Args::Switch::Action default_action)
{
  Action action = default_action;

  if (m_args.size()) {
    string arg = m_args.top();
    Case i = m_cases.find(arg);
    if (i != m_cases.end()) {
      m_args.pop();
      action = i->second;
    }
  }

  action(m_args);
  //delete this;
}

void Args::Switch::default_error ()
{
  if (not m_args.size()) {
    cout << m_args.help << "\nERROR missing command.";
    cout << "\ntry one of:";
    for (Case i = m_cases.begin(); i != m_cases.end(); ++i) {
      cout << " " << i->first;
    }
    cout << endl;
    exit(1);
  }
  string arg = m_args.pop();
  Case i = m_cases.find(arg);
  if (i != m_cases.end()) {
    i->second(m_args);
  } else {
    cout << m_args.help << "\nERROR unknown command: " << arg;
    cout << "\ntry one of:";
    for (Case i = m_cases.begin(); i != m_cases.end(); ++i) {
      cout << " " << i->first;
    }
    cout << endl;
    exit(1);
  }

  //delete this;
}

void Args::Switch::default_break_else_repeat ()
{
  while (m_args.size()) {
    string arg = m_args.top();
    Case i = m_cases.find(arg);
    if (i == m_cases.end()) break;
    m_args.pop();
    i->second(m_args);
  }

  //delete this;
}

void Args::Switch::default_all ()
{
  if (not m_args.size()) {
    for (Case i = m_cases.begin(); i != m_cases.end(); ++i) {
      i->second(m_args);
    }
  } else {
    string arg = m_args.pop();
    Case i = m_cases.find(arg);
    if (i != m_cases.end()) {
      i->second(m_args);
    } else {
      cout << m_args.help << "\nERROR unknown command: " << arg;
      cout << "\ntry one of:";
      for (Case i = m_cases.begin(); i != m_cases.end(); ++i) {
        cout << " " << i->first;
      }
      cout << endl;
      exit(1);
    }
  }

  //delete this;
}

Args::Switch Args::case_ (string key, Args::Switch::Action action)
{
  Switch * result = new Switch(* this);
  return result->case_(key, action);
}

Args::~Args ()
{
  if (size()) {
    std::ostringstream s;
    s << "WARNING the following arguments were not used:";
    while (size()) s << " " << pop();
    cout << s.str() << endl;
  }
}

