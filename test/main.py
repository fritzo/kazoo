
import os, sys, inspect

__commands = []
def command (fun):
  'decorator for main commands'

  args,vargs,kwds,defaults = inspect.getargspec(fun)
  if defaults is None: defaults = ()
  types = [str] * (len(args) - len(defaults)) + [d.__class__ for d in defaults]
  def parser (*args,**kwds):
    assert not kwds, 'TODO parse keyword arguments'
    types_etc = types + [str] * (len(args) - len(types)) # for vargs
    fun(*tuple(t(a) for a,t in zip(args,types_etc)))

  assert fun.__doc__, 'missing docstring for %s' % name
  name = fun.__name__.replace('_','-')
  __commands.append((name,(fun,parser)))

  return fun

def at_top (extra_depth = 0):
  'returns whether calling location is top-level main command'

  depth = len(inspect.stack())
  assert depth >= 5
  return depth == 5 + extra_depth

def main (args = None):
  'parses arguments to call a main command'

  if args is None:
    args = sys.argv[1:]

  if not args:
    print('Usage: %s COMMAND [ARGS] [KWDS]' % os.path.split(sys.argv[0])[-1])
    for name,(fun,_) in  __commands:
      print('\n%s %s\n  %s' % (
          name,
          inspect.formatargspec(*inspect.getargspec(fun)),
          fun.__doc__.strip(),
          ))
    sys.exit(1)

  cmd,args,kwds = args[0],args[1:],{}
  while args and '=' in args[-1]:
    key,val = args.pop().split('=',1)
    kwds[key] = value
  dict(__commands)[cmd.replace('_','-')][1](*args,**kwds)

